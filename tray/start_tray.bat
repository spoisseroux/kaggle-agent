@echo off
REM Kaggle Agent system tray launcher (Windows).
REM Hides the console by using pythonw. Works from any CWD.

setlocal
pushd "%~dp0"

REM Use pythonw if available (no console window); fall back to python.
where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw kaggle_tray.py
) else (
    start "" python kaggle_tray.py
)

popd
endlocal
