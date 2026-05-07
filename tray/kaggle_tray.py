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
Use tray_launcher.py (auto-restarts on file change), or directly:
    pythonw kaggle_tray.py
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
import webbrowser
from typing import Optional

import pystray
import requests
from PIL import Image, ImageDraw, ImageFont

API_BASE = "http://localhost:8765"
DASHBOARD_URL = "https://kaggle-ui.nnaq.net"
POLL_INTERVAL_S = 5
HTTP_TIMEOUT = 4
ACTION_TIMEOUT = 20

ICON_SIZE = 64

# Name of the Windows Task Scheduler task that runs wsl_startup.sh at logon.
# Change this to match whatever name was used when the task was created.
STARTUP_TASK_NAME = "Kaggle Agent"

# WSL distro name (used when opening a terminal)
WSL_DISTRO = "Ubuntu-24.04"

AVAILABLE_MODELS = [
    ("Haiku 4.5 · fast/cheap", "claude-haiku-4-5-20251001"),
    ("Sonnet 4.6 · balanced",  "claude-sonnet-4-6"),
    ("Opus 4.7 · powerful",    "claude-opus-4-7"),
]

COLOURS = {
    "running":       (41, 182, 95),    # green
    "paused":        (246, 195, 77),   # yellow
    "stopped":       (144, 150, 156),  # gray
    "transitioning": (138, 169, 255),  # blue
    "offline":       (228, 80, 80),    # red — API unreachable
}

STAGE_LABELS = {
    "idle":                 "",
    "downloading":          "⬇ Downloading",
    "eda":                  "🔍 EDA",
    "training":             "🏋 Training",
    "generating_submission":"📄 Generating",
    "submitting":           "📤 Submitting",
    "done":                 "✓ Done",
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


def _schtasks_query(task_name: str) -> Optional[bool]:
    """Return True if task is enabled, False if disabled, None if not found."""
    try:
        r = subprocess.run(
            ["schtasks", "/Query", "/TN", task_name, "/FO", "LIST"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return None
        # Look for "Status:" or "Scheduled Task State:" lines
        for line in r.stdout.splitlines():
            low = line.lower()
            if "status" in low and ("enabled" in low or "disabled" in low):
                return "disabled" not in low
        return True  # found but status unclear → assume enabled
    except Exception:
        return None


def _schtasks_set(task_name: str, enable: bool) -> bool:
    flag = "/Enable" if enable else "/Disable"
    try:
        r = subprocess.run(
            ["schtasks", "/Change", "/TN", task_name, flag],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _open_tmux_window() -> None:
    """Open a Windows Terminal (or cmd) attached to the kaggle-agent tmux session.

    Uses attach-session with -x (detach other clients) and exact session name.
    Falls back to an informative shell if the session doesn't exist yet.
    Never creates a bogus grouped session via 'new-session -t'.
    """
    bash_cmd = (
        "tmux attach-session -t kaggle-agent 2>/dev/null || "
        "{ echo 'kaggle-agent session not running — start it via wsl_startup.sh'; "
        "cd /home/keehar/kaggle-agent && exec bash; }"
    )
    # Try Windows Terminal first (no -d flag — avoids UNC/Windows path issues)
    try:
        subprocess.Popen(["wt", "wsl", "-d", WSL_DISTRO, "-e", "bash", "-c", bash_cmd])
        return
    except FileNotFoundError:
        pass
    subprocess.Popen(
        ["cmd", "/c", "start", "cmd", "/k",
         f"wsl -d {WSL_DISTRO} -e bash -c \"{bash_cmd}\""],
    )


class KaggleTray:
    def __init__(self) -> None:
        self.state: str = "transitioning"
        self.since_ts: Optional[float] = None
        self.active_competition: Optional[str] = None
        self.run_stage: Optional[str] = None
        self.run_stage_detail: Optional[str] = None
        self.run_eta_seconds: Optional[int] = None
        self.current_model: str = "claude-sonnet-4-6"
        self.startup_enabled: Optional[bool] = None  # None = task not found

        self._stop = threading.Event()
        self._action_lock = threading.Lock()

        # Load startup state and current model in background to avoid blocking
        threading.Thread(target=self._load_initial_state, daemon=True).start()

        self.icon = pystray.Icon(
            "kaggle-agent",
            icon=_make_icon_image(self.state),
            title=self._tooltip(),
            menu=self._build_menu(),
        )

    # ---------- helpers ----------

    def _tooltip(self) -> str:
        comp = self.active_competition or "no active competition"
        base = f"Kaggle Agent — {self.state}  ({comp})"
        if self.state == "running" and self.run_stage and self.run_stage != "idle":
            label = STAGE_LABELS.get(self.run_stage, self.run_stage)
            if self.run_stage_detail:
                label += f" {self.run_stage_detail}"
            if self.run_eta_seconds is not None:
                mins = self.run_eta_seconds // 60
                label += f", ~{mins}m left" if mins > 0 else f", ~{self.run_eta_seconds}s left"
            base += f"\n{label}"
        return base

    def _build_menu(self) -> pystray.Menu:
        def status_label(_):
            label = f"● {self.state.title()}"
            if self.active_competition:
                label += f" · {self.active_competition}"
            if self.state == "running" and self.run_stage and self.run_stage != "idle":
                stage_txt = STAGE_LABELS.get(self.run_stage, self.run_stage)
                if self.run_stage_detail:
                    stage_txt += f" {self.run_stage_detail}"
                label += f"  [{stage_txt}]"
            return label

        # Model submenu — checkmarks (radio=True omitted for pystray compat)
        # Use factory functions so pystray sees exactly 2-arg actions.
        def _model_action(m):
            def action(icon, item):
                self._set_model(m)
            return action

        def _model_check(m):
            def checked(item):
                return self.current_model == m
            return checked

        model_items = []
        for label, model_id in AVAILABLE_MODELS:
            model_items.append(
                pystray.MenuItem(
                    label,
                    _model_action(model_id),
                    checked=_model_check(model_id),
                )
            )

        # Startup toggle
        def startup_label(_):
            if self.startup_enabled is None:
                return "Start on startup  (task not found)"
            return "Start on startup"

        def startup_checked(_):
            return bool(self.startup_enabled)

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
            pystray.MenuItem("Open tmux session", self._open_tmux),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Model", pystray.Menu(*model_items)),
            pystray.MenuItem(
                startup_label,
                self._toggle_startup,
                checked=startup_checked,
                enabled=lambda _: self.startup_enabled is not None,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit tray", self._quit),
        )

    def _refresh_icon(self) -> None:
        self.icon.icon = _make_icon_image(self.state)
        self.icon.title = self._tooltip()
        self.icon.update_menu()

    def _set_state(
        self,
        state: str,
        since_ts: Optional[float] = None,
        active_competition: Optional[str] = None,
        run_stage: Optional[str] = None,
        run_stage_detail: Optional[str] = None,
        run_eta_seconds: Optional[int] = None,
    ) -> None:
        changed = (
            state != self.state
            or since_ts != self.since_ts
            or active_competition != self.active_competition
            or run_stage != self.run_stage
            or run_stage_detail != self.run_stage_detail
            or run_eta_seconds != self.run_eta_seconds
        )
        if not changed:
            return
        self.state = state
        self.since_ts = since_ts
        self.active_competition = active_competition
        self.run_stage = run_stage
        self.run_stage_detail = run_stage_detail
        self.run_eta_seconds = run_eta_seconds
        self._refresh_icon()

    def _load_initial_state(self) -> None:
        self.startup_enabled = _schtasks_query(STARTUP_TASK_NAME)
        try:
            r = requests.get(f"{API_BASE}/system/model", timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                self.current_model = r.json().get("model", self.current_model)
        except Exception:
            pass

    # ---------- menu actions ----------

    def _do_pause(self, icon, item) -> None:
        self._call_action("pause")

    def _do_resume(self, icon, item) -> None:
        self._call_action("resume")

    def _do_stop(self, icon, item) -> None:
        self._call_action("stop")

    def _open_dashboard(self, icon, item) -> None:
        webbrowser.open(DASHBOARD_URL)

    def _open_tmux(self, icon, item) -> None:
        threading.Thread(target=_open_tmux_window, daemon=True).start()

    def _toggle_startup(self, icon, item) -> None:
        if self.startup_enabled is None:
            return
        new_state = not self.startup_enabled
        ok = _schtasks_set(STARTUP_TASK_NAME, new_state)
        if ok:
            self.startup_enabled = new_state
            self._refresh_icon()

    def _set_model(self, model_id: str) -> None:
        try:
            r = requests.post(
                f"{API_BASE}/system/model",
                json={"model": model_id},
                timeout=HTTP_TIMEOUT,
            )
            if r.status_code == 200:
                self.current_model = model_id
                self._refresh_icon()
        except Exception:
            pass

    def _quit(self, icon, item) -> None:
        self._stop.set()
        self.icon.stop()
        # Exit with code 0 so tray_launcher.py knows this was intentional
        # and does NOT restart us. Non-zero exit = crash = launcher restarts.
        import os
        os._exit(0)

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
                    run_stage=data.get("run_stage"),
                    run_stage_detail=data.get("run_stage_detail"),
                    run_eta_seconds=data.get("run_eta_seconds"),
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
                        self._stop.wait(POLL_INTERVAL_S)
                        continue
                    self._set_state(
                        new_state,
                        since_ts=data.get("since"),
                        active_competition=data.get("active_competition"),
                        run_stage=data.get("run_stage"),
                        run_stage_detail=data.get("run_stage_detail"),
                        run_eta_seconds=data.get("run_eta_seconds"),
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
