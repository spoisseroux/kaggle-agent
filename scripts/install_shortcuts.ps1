# install_shortcuts.ps1 — create Desktop + Start Menu shortcuts for the
# Kaggle Agent tray. Run this once after a fresh setup.
#
# Usage (from Windows PowerShell, no admin needed):
#   powershell -ExecutionPolicy Bypass -File `
#     "\\wsl.localhost\Ubuntu-24.04\home\keehar\kaggle-agent\scripts\install_shortcuts.ps1"
#
# Or from any drive: cd to the script's folder, then:
#   .\install_shortcuts.ps1

$ErrorActionPreference = "Stop"

$batPath = "\\wsl.localhost\Ubuntu-24.04\home\keehar\kaggle-agent\tray\start_tray.bat"
$iconPath = "\\wsl.localhost\Ubuntu-24.04\home\keehar\kaggle-agent\tray\icon.png"

if (-not (Test-Path $batPath)) {
    Write-Error "Cannot find $batPath — is WSL running and the repo in place?"
    exit 1
}

$shell = New-Object -ComObject WScript.Shell

function New-KaggleShortcut {
    param([string]$Path, [string]$Description)

    $sc = $shell.CreateShortcut($Path)
    $sc.TargetPath = $batPath
    $sc.WorkingDirectory = Split-Path $batPath -Parent
    $sc.WindowStyle = 7   # Minimized
    $sc.Description = $Description
    # PNG can't be used as a shortcut icon directly, but we keep this hint
    # for now — once we ship a real .ico via PyInstaller, point this at it.
    # $sc.IconLocation = $iconPath
    $sc.Save()
    Write-Host "  Created: $Path"
}

Write-Host "Installing Kaggle Agent shortcuts..."

# Desktop
$desktop = [Environment]::GetFolderPath("Desktop")
New-KaggleShortcut `
    -Path (Join-Path $desktop "Kaggle Agent.lnk") `
    -Description "Start the Kaggle Agent tray + all WSL services"

# Start Menu (user-level, no admin)
$startMenu = [Environment]::GetFolderPath("StartMenu")
$programsDir = Join-Path $startMenu "Programs"
if (-not (Test-Path $programsDir)) { New-Item -ItemType Directory -Path $programsDir | Out-Null }
New-KaggleShortcut `
    -Path (Join-Path $programsDir "Kaggle Agent.lnk") `
    -Description "Start the Kaggle Agent tray + all WSL services"

# Startup folder (replaces the manual shortcut you had)
$startup = [Environment]::GetFolderPath("Startup")
New-KaggleShortcut `
    -Path (Join-Path $startup "Kaggle Agent.lnk") `
    -Description "Auto-start Kaggle Agent tray at login"

Write-Host ""
Write-Host "Done. You can now:"
Write-Host "  - Double-click 'Kaggle Agent' on the Desktop to start everything"
Write-Host "  - Find it in Start Menu under 'Kaggle Agent'"
Write-Host "  - It auto-launches on every Windows login (Startup folder shortcut)"
Write-Host ""
Write-Host "To uninstall, delete these three .lnk files manually."
