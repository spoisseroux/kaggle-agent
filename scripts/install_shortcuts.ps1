# install_shortcuts.ps1 -- create Desktop + Start Menu + Startup shortcuts
# for the Kaggle Agent tray. Run once after a fresh setup.
#
# Usage from any Windows PowerShell (no admin needed):
#   powershell -ExecutionPolicy Bypass -File "\\wsl.localhost\Ubuntu-24.04\home\keehar\kaggle-agent\scripts\install_shortcuts.ps1"

$ErrorActionPreference = "Stop"

$batPath = "\\wsl.localhost\Ubuntu-24.04\home\keehar\kaggle-agent\tray\start_tray.bat"

if (-not (Test-Path $batPath)) {
    Write-Error "Cannot find $batPath -- is WSL running and the repo in place?"
    exit 1
}

$shell = New-Object -ComObject WScript.Shell
$workingDir = Split-Path $batPath -Parent
$desc = "Start the Kaggle Agent tray and all WSL services"

$targets = @(
    [Environment]::GetFolderPath("Desktop"),
    (Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs"),
    [Environment]::GetFolderPath("Startup")
)

Write-Host "Installing Kaggle Agent shortcuts..."

foreach ($dir in $targets) {
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    $lnkPath = Join-Path $dir "Kaggle Agent.lnk"
    $sc = $shell.CreateShortcut($lnkPath)
    $sc.TargetPath = $batPath
    $sc.WorkingDirectory = $workingDir
    $sc.WindowStyle = 7
    $sc.Description = $desc
    $sc.Save()
    Write-Host "  Created: $lnkPath"
}

Write-Host ""
Write-Host "Done. You can now:"
Write-Host "  - Double-click 'Kaggle Agent' on the Desktop to start everything"
Write-Host "  - Find it in Start Menu under 'Kaggle Agent'"
Write-Host "  - It auto-launches on every Windows login"
Write-Host ""
Write-Host "To uninstall, delete the three Kaggle Agent.lnk files manually."
