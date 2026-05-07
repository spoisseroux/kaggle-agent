@echo off
REM Kaggle Agent system tray launcher (Windows).
REM Kills any existing tray instances first, then starts exactly one.
REM Hardcoded UNC path so this works from startup folder / shortcuts.

set "SCRIPT=\\wsl.localhost\Ubuntu-24.04\home\keehar\kaggle-agent\tray\tray_launcher.py"

REM Kill any running tray launcher or tray instances (precise — command-line match)
powershell -WindowStyle Hidden -Command ^
  "Get-WmiObject Win32_Process ^| Where-Object {$_.CommandLine -like '*kaggle_tray*' -or $_.CommandLine -like '*tray_launcher*'} ^| ForEach-Object {Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue}"

REM Brief pause so killed processes release the mutex before we start
timeout /t 1 /nobreak >nul

where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw "%SCRIPT%"
) else (
    start "" python "%SCRIPT%"
)
