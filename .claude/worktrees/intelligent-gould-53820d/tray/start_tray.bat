@echo off
REM Kaggle Agent system tray launcher (Windows).
REM Starts tray_launcher.py with pythonw (no console window).
REM tray_launcher.py watches kaggle_tray.py for changes and auto-restarts the
REM tray whenever the source file is saved — no manual restart needed.

setlocal
pushd "%~dp0"

REM Use pythonw if available (no console window); fall back to python.
where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw tray_launcher.py
) else (
    start "" python tray_launcher.py
)

popd
endlocal
