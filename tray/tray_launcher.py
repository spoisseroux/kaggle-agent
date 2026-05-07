"""Launcher that watches kaggle_tray.py for changes and auto-restarts it.

Use start_tray.bat to launch this (it hides the console with pythonw).
Run this script directly (python tray_launcher.py) for visible output.

Behaviour
---------
- Enforces single-instance via a Windows named mutex — a second launch exits
  immediately rather than creating a duplicate tray icon.
- Starts kaggle_tray.py as a child pythonw process.
- Polls kaggle_tray.py mtime every 2 seconds.
- On change: gracefully terminates the child, waits up to 5s, then restarts.
- If the child crashes on its own: restarts after a 3s cooldown.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

_MUTEX_NAME = "KaggleTrayLauncher_SingleInstance"

def _acquire_single_instance_mutex() -> object:
    """Create a named mutex. If it already exists, another instance is running — exit."""
    kernel32 = ctypes.windll.kernel32
    mutex = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    if kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        print("Kaggle tray is already running — exiting.", flush=True)
        sys.exit(0)
    return mutex  # keep reference alive for process lifetime

TRAY_SCRIPT = Path(__file__).resolve().parent / "kaggle_tray.py"
POLL_INTERVAL_S = 2
RESTART_COOLDOWN_S = 3
GRACEFUL_SHUTDOWN_S = 5

# Use pythonw so the restarted tray has no console window.
# Fall back to the same interpreter that launched this script.
def _python_exe() -> str:
    exe = sys.executable
    if exe.lower().endswith("python.exe"):
        candidate = exe[:-10] + "pythonw.exe"
        if Path(candidate).exists():
            return candidate
    return exe


def _start() -> subprocess.Popen:
    return subprocess.Popen(
        [_python_exe(), str(TRAY_SCRIPT)],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def main() -> None:
    print(f"Tray launcher watching: {TRAY_SCRIPT}", flush=True)
    mtime = TRAY_SCRIPT.stat().st_mtime
    proc = _start()
    print(f"Started tray pid={proc.pid}", flush=True)

    while True:
        time.sleep(POLL_INTERVAL_S)

        # Check for file change
        try:
            new_mtime = TRAY_SCRIPT.stat().st_mtime
        except FileNotFoundError:
            continue  # file temporarily missing (save-in-progress)

        if new_mtime != mtime:
            print(f"kaggle_tray.py changed — restarting tray", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=GRACEFUL_SHUTDOWN_S)
            except subprocess.TimeoutExpired:
                proc.kill()
            mtime = new_mtime
            time.sleep(0.5)  # brief pause for any editor finalisation
            proc = _start()
            print(f"Restarted tray pid={proc.pid}", flush=True)
            continue

        # Check if child exited unexpectedly
        if proc.poll() is not None:
            print(f"Tray exited (rc={proc.returncode}) — restarting in {RESTART_COOLDOWN_S}s",
                  flush=True)
            time.sleep(RESTART_COOLDOWN_S)
            mtime = TRAY_SCRIPT.stat().st_mtime
            proc = _start()
            print(f"Restarted tray pid={proc.pid}", flush=True)


if __name__ == "__main__":
    _mutex = _acquire_single_instance_mutex()
    main()
