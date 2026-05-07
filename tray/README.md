# Kaggle Agent — Windows tray app

Runs **on Windows**, not inside WSL2. Talks to the FastAPI service that's
running inside WSL2 over `localhost:8765` (WSL2 bridges its ports to
Windows automatically).

## Icon
A bold "K" on a coloured rounded square:

| Colour | Meaning |
|---|---|
| 🟢 green | running |
| 🟡 yellow | paused |
| ⚪ gray | stopped |
| 🔵 blue | a request is in flight (transitioning) |
| 🔴 red | API unreachable (WSL2 down? `kaggle-api` service stopped?) |

## Menu

```
● Running · <active competition>      (status, non-clickable)
─────────────────────────────────────
Pause                                  (only when running)
Stop                                   (when running or paused)
Resume                                 (when paused or stopped)
─────────────────────────────────────
Open Dashboard                         → https://kaggle.nnaq.net
─────────────────────────────────────
Quit tray                              (does NOT stop the agent)
```

## Install (one-time, from a Windows PowerShell or CMD)

```
pip install pystray pillow requests
```

The tray uses Python on **Windows** — not Python in WSL2. You probably
want Python 3.10+ from python.org. Don't use the Microsoft Store Python:
its sandbox blocks `pythonw` from sitting in the tray reliably.

## Run

Double-click `start_tray.bat`, or from a shell:

```
pythonw kaggle_tray.py
```

`pythonw` (note the `w`) is the windowless Python entrypoint — there is
no console window, just the tray icon.

## Auto-start on login

The simplest way is to put a shortcut to `start_tray.bat` in the
**Startup** folder:

1. Press `Win+R`, type `shell:startup`, press Enter — File Explorer opens
   the Startup folder.
2. Right-click → New → Shortcut.
3. Target: `\\wsl.localhost\Ubuntu-24.04\home\keehar\kaggle-agent\tray\start_tray.bat`
   (or copy the folder onto Windows first if you don't want it living on
   the WSL2 filesystem).
4. Name it "Kaggle Agent Tray".

For a more reliable trigger (runs even if Explorer hasn't fully started),
use Task Scheduler:

```
Action:    Start a program
Program:   C:\Path\To\pythonw.exe
Arguments: "C:\Path\To\kaggle_tray.py"
Trigger:   At log on of <your user>
```

## Notes
- "Quit tray" only closes the tray; the agent itself keeps running. Use
  "Stop" if you really want to stop the agent.
- `localhost:8765` in WSL2 = `localhost:8765` in Windows because WSL2's
  `localhostForwarding=true` is on by default. If you've explicitly
  turned that off in `.wslconfig`, point the tray at the WSL2 box's
  Tailscale name instead by editing `API_BASE` at the top of
  `kaggle_tray.py` — e.g. `http://kaggle:8765`.
- The dashboard URL `https://kaggle.nnaq.net` assumes the Cloudflare
  Tunnel is configured (see `../HANDOFF.md`). Until then, point it at
  whatever URL the agency-platform UI is reachable on.
