# Kaggle Agent — Packaging & Distribution Roadmap

Plan to go from "Python script + .bat in Startup folder" to a signed
GitHub-released installer, without breaking the current working setup.

## Current State

- Tray app: `tray/kaggle_tray.py` (Python + pystray)
- Launched at boot via `start_tray.bat` in Windows Startup folder
- Triggers a SmartScreen / UAC prompt every boot — manual "Yes" required
- Icon: `tray/icon.png` (hardcoded path)
- Distribution: clone repo + install Python deps manually

## Goals

1. Zero prompts at boot — tray launches silently
2. Single `.exe` with embedded icon
3. GitHub Release with a `Setup.exe` installer anyone can double-click
4. Icon trivially swappable (drop in new SVG → rebuild)
5. Preserve current behavior — agent / WSL / tmux flow unchanged

## Non-Goals (for now)

- Cross-platform (Mac/Linux) — Windows only
- Built-in auto-update — defer to Phase 5
- Buying an EV code-signing cert ($300+/yr) — defer to Phase 4
- Installing WSL or Python for the user — assume those exist

## Why the Boot Prompt Happens

Two Windows mechanisms can trigger it:
- **SmartScreen** — flags any `.exe` / `.bat` without a known publisher signature
- **UAC** — fires if the batch tries to do anything privileged

`start_tray.bat` is unsigned and downloaded (carries the
`Zone.Identifier` flag), so SmartScreen flags it. Two fixes:
- **Quick fix:** move the trigger from Startup folder → Task Scheduler.
  Tasks scheduled via Task Scheduler don't go through SmartScreen.
- **Real fix:** code-sign the final `.exe` (Phase 4).

---

## Phased Plan

### Phase 0 — Kill the boot prompt today (no code changes)

Replace the Startup-folder shortcut with a Task Scheduler entry.

```powershell
# Run once as your normal user (no admin needed for "On log on" trigger)
$action = New-ScheduledTaskAction `
    -Execute "C:\path\to\kaggle-agent\tray\start_tray.bat"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERNAME"
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable -DontStopOnIdleEnd
Register-ScheduledTask -TaskName "KaggleAgentTray" `
    -Action $action -Trigger $trigger -Settings $settings
```

Then delete the Startup folder shortcut. No more prompts.

**Effort:** 5 minutes. **Result:** problem solved while we build the
real version.

---

### Phase 1 — Build a real `.exe` with PyInstaller

Bundle Python + tray code + icon into a single `KaggleAgent.exe`.

**New files:**
- `build/assets/icon.svg` — source icon (already added: `tray/assets/icon.svg`)
- `build/build_icon.ps1` — converts SVG → multi-resolution `.ico`
- `build/kaggle_tray.spec` — PyInstaller spec with `--windowed --onefile --icon`
- `build/build_tray.ps1` — full build script

**Build command:**
```powershell
pwsh build/build_tray.ps1
# → dist/KaggleAgent.exe (~15 MB, single file, custom icon)
```

**Icon swap workflow:**
1. Replace `build/assets/icon.svg`
2. Run `build/build_tray.ps1`
3. Done — new icon embedded in the next build

**Tooling:** PyInstaller (free), Inkscape or ImageMagick for SVG→ICO.

---

### Phase 2 — Inno Setup installer

Wrap the `.exe` in a proper Windows installer.

**New file:** `build/installer.iss` (Inno Setup script)

**Installer features:**
- Installs `KaggleAgent.exe` to `%LOCALAPPDATA%\KaggleAgent\`
- Creates Start Menu shortcut
- Optional checkbox: "Launch at startup" → registers Task Scheduler entry
- Uninstaller included
- Outputs: `KaggleAgentSetup-vX.Y.Z.exe` (~20 MB)

**User experience:**
1. Download `KaggleAgentSetup-v1.0.0.exe` from GitHub Releases
2. Double-click → wizard → "Install"
3. Tray icon appears, launches at every boot from then on

**Tooling:** Inno Setup (free, scriptable, industry standard for ~20 yrs).

---

### Phase 3 — GitHub Actions release pipeline

Auto-build on tag push.

**New file:** `.github/workflows/release.yml`

**Trigger:** push a tag matching `v*` (e.g. `v1.0.0`)

**Steps:**
1. Spin up a Windows GitHub runner
2. Install Python, PyInstaller, Inno Setup
3. Run `build/build_tray.ps1` → `KaggleAgent.exe`
4. Run Inno Setup → `KaggleAgentSetup-v1.0.0.exe`
5. Create GitHub Release with both files attached
6. Auto-generated release notes from commits since last tag

**Release flow from then on:**
```bash
git tag v1.0.0
git push --tags
# 5 minutes later, release is live with downloadable installer
```

**Cost:** Free for public repos on GitHub Actions.

---

### Phase 4 — Code signing (eliminates SmartScreen warnings)

The installer will still show "Unknown publisher" until signed.

Three options ranked by cost:

| Option | Cost | Notes |
|---|---|---|
| **SignPath.io** | Free for OSS | Cleanest. Requires public repo & approval. |
| **Azure Trusted Signing** | ~$10/mo | Microsoft-managed. Easiest for solo devs. |
| **DigiCert / Sectigo OV cert** | $200-400/yr | Old-school. Hardware token required. |
| **Self-signed cert** | Free | Still shows SmartScreen — pointless for distribution. |

**Recommendation:** Start with SignPath if going open-source; otherwise
Azure Trusted Signing.

Integrate into the Phase 3 GitHub Action — signs after build, before
release upload.

---

### Phase 5 — Auto-update (future)

Once Phase 3 is live, add an in-tray version check:
- On startup, tray hits `GET /repos/spoisseroux/kaggle-agent/releases/latest`
- If newer version exists, show menu item: "Update available — v1.2.0"
- Click → opens browser to release page

Not Sparkle/Squirrel level — just a notification. User downloads the
new installer themselves. Keeps complexity low.

---

## Icon Asset Pipeline

**Source of truth:** `tray/assets/icon.svg` (already added)

**Build process generates:**
- `tray/icon.png` — 64×64 PNG for the running tray app (unchanged behavior)
- `build/dist/icon.ico` — multi-resolution ICO (16/32/48/64/128/256) for the `.exe`

**To swap the icon later:**
- Replace `tray/assets/icon.svg`
- Re-run `build/build_tray.ps1` (or just commit + push a tag for GitHub Actions to do it)

The build script handles both PNG and ICO generation so the running
tray and the packaged exe stay in sync.

---

## Risk: Don't Break What's Working

The current setup is stable. To keep it that way:

- **Phase 0** is the only thing affecting the running system right now.
  It's reversible — delete the Task Scheduler entry, put the shortcut back.
- **Phases 1-3** all happen in a new `build/` directory. They produce
  artifacts but don't touch `tray/kaggle_tray.py` or the agent code.
- The packaged `.exe` is functionally identical to running `python tray/kaggle_tray.py`.
- We can ship Phase 1-3 as "experimental" while you keep using the .bat
  until you're confident.

---

## Recommended Order

1. **Today (5 min):** Do Phase 0 to stop the boot prompt
2. **Next afternoon (2-3 hrs):** Phases 1 + 2 together — `.exe` + installer locally
3. **Following session (1 hr):** Phase 3 — GitHub Actions
4. **When traction matters:** Phase 4 — code signing
5. **Later, optional:** Phase 5 — auto-update

---

## Open Questions Before We Start

1. **Repo visibility** — public or private? Affects free signing options
   (SignPath needs public) and GitHub Actions minutes (private = limited free tier)
2. **Distribution audience** — just you, or anyone? If just you, Phase 4 doesn't matter
3. **Update philosophy** — when you change `tray/kaggle_tray.py`, do users need
   the new code at runtime, or only on next install? Affects Phase 5 design.
4. **Versioning scheme** — semver (`v1.0.0`)? CalVer (`v2026.5.13`)?

Pick answers and we start with Phase 0.
