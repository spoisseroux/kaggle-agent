"""Kaggle agent — Windows system tray app.

Runs natively on Windows (NOT inside WSL2). Polls the kaggle-api FastAPI
service running in WSL2 (WSL2 ports are auto-bridged to localhost on
Windows) and exposes pause / resume / stop in the tray menu.

Icon
----
A simple "K" letter on a coloured rounded square:
    green  → running
    yellow → paused
    gray   → stopped
    blue   → transitioning (a request is in flight)
    red    → API unreachable

Dependencies
------------
    pip install pystray pillow requests

Run
---
Double-click `start_tray.bat`, or:
    pythonw kaggle_tray.py

`pythonw` (instead of `python`) hides the console window.
"""
from __future__ import annotations

import threading
import time
import webbrowser
from typing import Optional

import pystray
import requests
from PIL import Image, ImageDraw, ImageFont

API_BASE = "http://localhost:8765"
DASHBOARD_URL = "https://kaggle.nnaq.net"
POLL_INTERVAL_S = 5
HTTP_TIMEOUT = 4
ACTION_TIMEOUT = 20

ICON_SIZE = 64

COLOURS = {
    "running":       (41, 182, 95),    # green
    "paused":        (246, 195, 77),   # yellow
    "stopped":       (144, 150, 156),  # gray
    "transitioning": (138, 169, 255),  # blue
    "offline":       (228, 80, 80),    # red — API unreachable
}


def _make_icon_image(state: str) -> Image.Image:
    colour = COLOURS.get(state, COLOURS["transitioning"])
    img = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((4, 4, ICON_SIZE - 4, ICON_SIZE - 4), radius=12, fill=colour)
    font = None
    for candidate in ("arialbd.ttf", "arial.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(candidate, 44)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    d.text((ICON_SIZE / 2, ICON_SIZE / 2 + 1), "K",
           fill="white", font=font, anchor="mm")
    return img


class KaggleTray:
    def __init__(self) -> None:
        self.state: str = "transitioning"
        self.since_ts: Optional[float] = None
        self.active_competition: Optional[str] = None
        self._stop = threading.Event()
        self._action_lock = threading.Lock()

        self.icon = pystray.Icon(
            "kaggle-agent",
            icon=_make_icon_image(self.state),
            title=self._tooltip(),
            menu=self._build_menu(),
        )

    # ---------- helpers ----------

    def _tooltip(self) -> str:
        comp = self.active_competition or "no active competition"
        return f"Kaggle Agent — {self.state}  ({comp})"

    def _build_menu(self) -> pystray.Menu:
        def status_label(_):
            label = f"● {self.state.title()}"
            if self.active_competition:
                label += f" · {self.active_competition}"
            return label

        return pystray.Menu(
            pystray.MenuItem(status_label, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Pause", self._do_pause,
                enabled=lambda _: self.state == "running",
            ),
            pystray.MenuItem(
                "Stop", self._do_stop,
                enabled=lambda _: self.state in ("running", "paused"),
            ),
            pystray.MenuItem(
                "Resume", self._do_resume,
                enabled=lambda _: self.state in ("paused", "stopped"),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open Dashboard", self._open_dashboard),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit tray", self._quit),
        )

    def _refresh_icon(self) -> None:
        self.icon.icon = _make_icon_image(self.state)
        self.icon.title = self._tooltip()
        self.icon.update_menu()

    def _set_state(self, state: str, since_ts: Optional[float] = None,
                   active_competition: Optional[str] = None) -> None:
        if (state == self.state
                and since_ts == self.since_ts
                and active_competition == self.active_competition):
            return
        self.state = state
        self.since_ts = since_ts
        self.active_competition = active_competition
        self._refresh_icon()

    # ---------- menu actions ----------

    def _do_pause(self, icon, item) -> None:
        self._call_action("pause")

    def _do_resume(self, icon, item) -> None:
        self._call_action("resume")

    def _do_stop(self, icon, item) -> None:
        self._call_action("stop")

    def _open_dashboard(self, icon, item) -> None:
        webbrowser.open(DASHBOARD_URL)

    def _quit(self, icon, item) -> None:
        self._stop.set()
        self.icon.stop()

    def _call_action(self, action: str) -> None:
        if not self._action_lock.acquire(blocking=False):
            return
        prev_state = self.state
        self._set_state("transitioning",
                        since_ts=self.since_ts,
                        active_competition=self.active_competition)
        try:
            r = requests.post(f"{API_BASE}/system/{action}",
                              timeout=ACTION_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                self._set_state(
                    data.get("state", prev_state),
                    since_ts=data.get("since"),
                    active_competition=data.get("active_competition"),
                )
            else:
                self._set_state(prev_state)
        except Exception:
            self._set_state("offline")
        finally:
            self._action_lock.release()

    # ---------- poller ----------

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                r = requests.get(f"{API_BASE}/system/state",
                                 timeout=HTTP_TIMEOUT)
                if r.status_code == 200:
                    data = r.json()
                    new_state = data.get("state", "transitioning")
                    # Don't clobber 'transitioning' while an action is in flight.
                    if self._action_lock.locked():
                        return
                    self._set_state(
                        new_state,
                        since_ts=data.get("since"),
                        active_competition=data.get("active_competition"),
                    )
                else:
                    self._set_state("offline")
            except Exception:
                self._set_state("offline")
            self._stop.wait(POLL_INTERVAL_S)

    # ---------- entrypoint ----------

    def run(self) -> None:
        threading.Thread(target=self._poll, daemon=True).start()
        self.icon.run()


if __name__ == "__main__":
    KaggleTray().run()
