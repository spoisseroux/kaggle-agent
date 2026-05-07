@echo off
REM Kaggle Agent system tray launcher (Windows).
REM Uses a hardcoded UNC path so this bat works when run from the startup
REM folder, a desktop shortcut, or any other location — cmd.exe cannot
REM set a UNC path as the working directory, so relative paths break.

set "SCRIPT=\\wsl.localhost\Ubuntu-24.04\home\keehar\kaggle-agent\tray\tray_launcher.py"

where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw "%SCRIPT%"
) else (
    start "" python "%SCRIPT%"
)
