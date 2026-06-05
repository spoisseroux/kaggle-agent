# Kaggle Agent

Autonomous Kaggle competition agent running locally on Windows + WSL2 + RTX
5070. Persistent tmux session with Claude Code at the wheel, talking to a
FastAPI backend, a Windows tray app, and you via Telegram and a web UI.

This README is the **disaster-recovery guide** — everything you need to
rebuild the system from a fresh Windows install. Follow top-to-bottom.

---

## What's running where

```
┌─ Windows host ───────────────────────────────────────────────────┐
│                                                                  │
│   System tray (pystray)  →  shows agent status, menu, dashboard  │
│   Tailscale              →  remote access via keehars-desktop    │
│   OpenSSH server         →  optional, for SSH-into-WSL flow      │
│                                                                  │
│   ┌─ WSL2 (Ubuntu-24.04) ───────────────────────────────────┐    │
│   │                                                         │    │
│   │   tmux session "kaggle-agent"                           │    │
│   │     └─ Claude Code (start_agent.sh) ←─ the agent        │    │
│   │                                                         │    │
│   │   systemd services:                                     │    │
│   │     • kaggle-api               FastAPI on :8765         │    │
│   │     • kaggle-api-watch         restarts api on edit     │    │
│   │     • telegram-bot             user → agent bridge      │    │
│   │     • telegram-enforcer        forces notify to fire    │    │
│   │     • telegram-auto-submit     submission gating        │    │
│   │     • mlflow                   experiment tracker       │    │
│   │     • ollama                   local LLM (qwen3:14b)    │    │
│   │     • kaggle-leaderboard-monitor.timer  hourly LB poll  │    │
│   │     • kaggle-shutdown-notify   Telegram on shutdown     │    │
│   │     • kaggle-message-responder bus → agent              │    │
│   │                                                         │    │
│   │   Python venv: /home/keehar/kaggle-venv                 │    │
│   │   Repo:        /home/keehar/kaggle-agent                │    │
│   │   Claude:      ~/.claude/.credentials.json              │    │
│   └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
         │
         │  Cloudflare Tunnel
         ▼
   kaggle.nnaq.net  (FastAPI public face)
   app.nnaq.net     (web frontend)
         │
         ▼
   Telegram bot ←→ you (mobile / desktop)
```

The agent runs continuously in tmux. The tray shows status. The web UI
and Telegram are both UIs onto the same SQLite message bus.

---

## Quick rebuild (when you lose your machine)

If you're rebuilding on a fresh Windows install, do these in order. Each
section below has detailed steps.

1. **Windows prep** — WSL2, Tailscale, OpenSSH, Python
2. **WSL prep** — Ubuntu, tmux, system packages
3. **Clone repo + create venv**
4. **Restore secrets** — `.env`, credentials
5. **Bootstrap services** — install systemd units, enable on boot
6. **Cloudflare tunnel** — restore tunnel config
7. **Tray app** — copy to Startup, registry trust fix
8. **Verify** — kick off the agent, confirm Telegram

Total time on a clean machine: ~2 hours, mostly waiting for downloads.

---

## 1. Windows host setup

### 1a. WSL2 + Ubuntu-24.04

```powershell
# In an Admin PowerShell
wsl --install -d Ubuntu-24.04
# Reboot if prompted, then set username/password when Ubuntu launches
```

Verify:
```powershell
wsl -l -v
# Should show: Ubuntu-24.04   Running   2
```

### 1b. Tailscale

Install Tailscale on Windows (NOT in WSL — WSL shares the Windows network
stack and inherits the Tailscale connection automatically).

- Download: <https://tailscale.com/download/windows>
- Sign in with your usual account
- Confirm hostname is `keehars-desktop` (or update notes accordingly)

### 1c. OpenSSH Server (optional, for remote tmux access)

```powershell
# In an Admin PowerShell — uses local install, much faster than Add-WindowsCapability
DISM /Online /Add-Capability /CapabilityName:OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
```

Then from any Tailscale device:
```bash
ssh skunk@keehars-desktop -t "wsl.exe -d Ubuntu-24.04 -- tmux attach -t kaggle-agent"
```

### 1d. Python on Windows (for the tray app)

Install Python 3.10+ from <https://python.org>. **Do not use Microsoft Store
Python** — its sandbox breaks `pythonw` sitting in the tray.

```powershell
pip install pystray pillow requests
```

### 1e. Trust UNC paths from WSL (kills the Startup security prompt)

```powershell
$path = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings\ZoneMap\Domains\wsl.localhost"
New-Item -Path $path -Force | Out-Null
New-ItemProperty -Path $path -Name "*" -Value 1 -PropertyType DWord -Force
```

This adds `\\wsl.localhost` to the Local Intranet zone (zone 1). Without
this, every boot prompts "Are you sure you want to run this file?" for
the tray batch. One-time fix, no admin needed.

### 1f. NVIDIA drivers + CUDA in WSL

Install the NVIDIA driver for Windows that includes WSL2 GPU passthrough.
After that, inside WSL `nvidia-smi` should show your RTX 5070.

If you reinstall: <https://developer.nvidia.com/cuda/wsl>

---

## 2. WSL setup

### 2a. System packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  tmux git curl wget build-essential \
  python3 python3-pip python3-venv \
  postgresql postgresql-contrib \
  jq htop nvtop
```

### 2b. Allow passwordless sudo for the agent services

The startup script calls `sudo -n systemctl start ...` — needs NOPASSWD on
those specific commands. Edit:

```bash
sudo visudo
```

Add (replace `keehar` with your user if different):
```
keehar ALL=(ALL) NOPASSWD: /bin/systemctl, /usr/bin/tailscale
```

### 2c. Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:14b
```

### 2d. Postgres (for agent memory)

```bash
sudo -u postgres psql -c "CREATE USER keehar WITH PASSWORD '<from .env>' SUPERUSER;"
sudo -u postgres createdb keehar_kaggle
# Tables get created automatically on first agent run
```

### 2e. Qdrant (vector store)

```bash
docker run -d --name qdrant --restart unless-stopped \
  -p 6333:6333 -p 6334:6334 \
  -v ~/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

---

## 3. Clone repo + venv

```bash
cd ~
git clone git@github.com:spoisseroux/kaggle-agent.git
cd kaggle-agent

python3 -m venv ~/kaggle-venv
source ~/kaggle-venv/bin/activate
pip install -r requirements.txt
```

If `requirements.txt` is stale, regenerate after install:
```bash
pip freeze > requirements.txt
```

---

## 4. Restore secrets

### 4a. `.env` (root of repo)

Create `.env` with these keys. **Never commit this file.**

```bash
# Telegram bot — create via @BotFather, your numeric chat ID via @userinfobot
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TELEGRAM_AUTHORIZED_USERS=<your-username>
TELEGRAM_REQUIRE_MENTION=false

# Kaggle — from kaggle.com/settings → Create New API Token
KAGGLE_USERNAME=...
KAGGLE_KEY=...
KAGGLE_API_TOKEN=...

# Anthropic / Claude
ANTHROPIC_MODEL=claude-sonnet-4-5
CLAUDE_BACKEND=cli

# Tailscale / network
TAILSCALE_HOSTNAME=keehars-desktop
TAILSCALE_MEMORY_HOST=<wsl-tailscale-ip>
CLOUDFLARE_TUNNEL_URL=https://kaggle.nnaq.net
ALLOWED_ORIGINS=https://app.nnaq.net,http://localhost:5173

# Local services
POSTGRES_PASSWORD=...
POSTGRES_DSN=postgresql://keehar:<password>@localhost:5432/keehar_kaggle
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:14b
MLFLOW_TRACKING_URI=http://localhost:5000
API_PORT=8765

# Optional
HF_TOKEN=...
HF_HOME=/home/keehar/.cache/huggingface
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
MCP_URL=...
MCP_BEARER_TOKEN=...
```

Restore from your password manager / backup. **The agent will not start
without `.env`.**

### 4b. Claude Code authentication

```bash
claude
# In the Claude prompt:
/login
# Open the URL in a browser, log in with your Claude Max account
```

This writes `~/.claude/.credentials.json`. The agent reads it on every
startup via `scripts/check_claude_auth.sh`. If the token is stale,
Claude Code refreshes it internally on first run.

### 4c. Kaggle CLI

```bash
mkdir -p ~/.kaggle
# The agent uses KAGGLE_USERNAME + KAGGLE_KEY env vars from .env
# Do NOT use ~/.kaggle/kaggle.json — agent expects env vars only
```

---

## 5. Bootstrap systemd services

The repo ships unit files in `systemd/`. Install them:

```bash
cd ~/kaggle-agent
sudo cp systemd/*.service systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload

# Enable + start each
for unit in kaggle-api kaggle-api-watch telegram-bot telegram-enforcer \
            telegram-auto-submit mlflow ollama \
            kaggle-shutdown-notify kaggle-message-responder; do
    sudo systemctl enable --now $unit.service
done

# The leaderboard monitor is a timer
sudo systemctl enable --now kaggle-leaderboard-monitor.timer
```

Verify all green:
```bash
systemctl status kaggle-api kaggle-api-watch telegram-bot \
    telegram-enforcer mlflow ollama --no-pager | grep -E 'Active:|●'
```

### What each service does

| Service | What it does |
|---|---|
| `kaggle-api` | FastAPI backend on `:8765` — talks to tray, web, Telegram bot |
| `kaggle-api-watch` | Restarts `kaggle-api` when source files change |
| `telegram-bot` | Polls Telegram, pushes messages into the SQLite bus |
| `telegram-enforcer` | Validates the agent actually sends notifications (no silent stops) |
| `telegram-auto-submit` | Submission-approval gate logic |
| `mlflow` | Experiment tracking UI on `:5000` |
| `ollama` | Local LLM (qwen3:14b) for cheap reasoning |
| `kaggle-leaderboard-monitor.timer` | Hourly poll, alerts on rank change |
| `kaggle-shutdown-notify` | "Agent offline" Telegram message on shutdown |
| `kaggle-message-responder` | Bridges user messages from bus to agent |

---

## 6. Cloudflare tunnel

The tunnel serves `kaggle.nnaq.net` (FastAPI) and `app.nnaq.net` (web UI)
from your local machine through Cloudflare.

```bash
# Install
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Authenticate (opens browser)
cloudflared tunnel login

# Restore tunnel config — keep your tunnel UUID in a password manager
# Config lives at ~/.cloudflared/config.yml — back this up!
cloudflared tunnel run <tunnel-name>

# Or run as a service
sudo cloudflared service install
```

DNS routes are configured in the Cloudflare dashboard, not here.

---

## 7. Tray app on Windows

### 7a. Install Python deps on Windows

```powershell
pip install pystray pillow requests
```

### 7b. Schedule it on boot

The simplest method is a shortcut in the Startup folder pointing to the
WSL-hosted batch:

1. Open `shell:startup` (Run dialog)
2. Right-click → New → Shortcut
3. Target: `\\wsl.localhost\Ubuntu-24.04\home\keehar\kaggle-agent\tray\start_tray.bat`
4. Name it: `Kaggle Agent Tray`

The registry trust fix from §1e prevents the boot prompt.

### 7c. WSL Task Scheduler entry for headless startup

The WSL services + tmux session should come up when you log in, even
without opening a terminal. Use Windows Task Scheduler:

1. Open **Task Scheduler**
2. Action → Create Task
3. **General:** Name = `KaggleAgentWSLStartup`, "Run only when user is logged on", "Run with highest privileges"
4. **Triggers:** "At log on" of your user
5. **Actions:** Start a program
   - Program: `wsl.exe`
   - Arguments: `-d Ubuntu-24.04 -- bash -c "/home/keehar/kaggle-agent/scripts/wsl_startup.sh"`
6. **Settings:** Uncheck "Stop if runs longer than"

`wsl_startup.sh` brings up Tailscale, all systemd services, the tmux
session running the agent, and sends a Telegram "agent online" ping.

---

## 8. Verify everything

After all the above, restart Windows. Within ~30 seconds of login you
should see:

1. Tray icon appears (green dot = running)
2. Telegram message: `🟢 Agent online — all services started`
3. <https://kaggle.nnaq.net> responds (Cloudflare → FastAPI)
4. `tmux attach -t kaggle-agent` (in WSL) shows the agent running

If anything's missing, check `~/kaggle-agent/logs/startup.log`.

---

## Daily operation

- **Watch progress:** Telegram, or <https://app.nnaq.net>
- **Attach to terminal:** `wsl -d Ubuntu-24.04 -- tmux attach -t kaggle-agent`
- **Restart agent:** `bash ~/kaggle-agent/scripts/restart_agent.sh`
- **Pause/resume:** Use the tray menu or `POST /system/pause` / `/system/resume`
- **Logs:** `~/kaggle-agent/logs/`
- **MLflow UI:** <http://localhost:5000>

Agent instructions live in `CLAUDE.md` (root) and
`competitions/active/<slug>/CLAUDE.md` (per-competition). Tray-specific
docs are in `tray/README.md`.

---

## Architecture notes

- **Message bus:** SQLite at `~/.kaggle-agent.db`. Telegram bot + web UI
  + agent all read/write to the same table. Pagination, unread counts,
  ask/reply matching all happen here.
- **Worktrees:** Claude Code uses git worktrees under `.claude/worktrees/`
  for parallel agent sessions. Each worktree has its own
  `.claude/settings.json` that must include the same permissions block as
  root or you'll get permission prompts.
- **Hooks:** `Stop`, `PreCompact`, `PostToolUse` hooks in
  `.claude/settings.json` fire scripts under `scripts/hooks/`. The Stop
  hook has a 90-second cooldown to prevent duplicate "Agent paused"
  notifications.
- **Auth check:** `scripts/check_claude_auth.sh` runs before every agent
  start. It tries an OAuth refresh that always fails with 403 (PKCE
  verifier isn't stored) — that's fine, Claude refreshes internally on
  startup. The script exits 0 either way so it never blocks.
- **Download safety:** `core/download_guard.py` gates all downloads.
  Checks effective free space (`min(WSL free, Windows C: free)` because
  WSL2's VHDX lives on C:), asks via Telegram before anything >10 GB,
  refuses if <20 GB effective free.

---

## Troubleshooting

### "Agent paused" repeats / agent goes silent
Check the Stop hook fired: `tail ~/kaggle-agent/logs/*.log`. The hook has
a 90s cooldown — multiple stops within 90s show only once.

### Telegram messages not arriving
Test the bot: `curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe"`
should return your bot info. If it 401s, the token is wrong/revoked.

### Telegram messages garbled (missing chars in filenames)
Old bug — `parse_mode: Markdown` ate underscores. Fixed in `core/notify.py`
by removing parse_mode. If it recurs, check git history of that file.

### Permission prompts when agent edits files
Worktree settings missing the permissions block. Each worktree under
`.claude/worktrees/*/` needs its own `.claude/settings.json` with
`Write(*)`, `Edit(*)`, etc.

### Token expired, can't refresh
Run `claude /login` in tmux. The auto-refresh always 403s (PKCE) — that's
expected. Only a fresh `/login` produces working credentials.

### WSL disk fills up
WSL2 VHDX lives on Windows C:. `df /` in WSL shows up to 1 TB free even
when C: only has 50 GB. `download_guard` accounts for this. If you've
already filled the VHDX, shrink it with `wsl --shutdown` followed by
Optimize-VHD in PowerShell.

### Cloudflare tunnel down
```bash
sudo systemctl status cloudflared
sudo journalctl -u cloudflared -n 50
```
Tunnel UUID + cert live in `~/.cloudflared/`. Back this directory up.

### Tray shows red dot
FastAPI is unreachable. `sudo systemctl restart kaggle-api` and check
`journalctl -u kaggle-api -n 50`.

---

## File layout

```
kaggle-agent/
├── CLAUDE.md              # Master instructions for the agent
├── README.md              # This file
├── .env                   # Secrets (NOT committed)
├── requirements.txt       # Python deps
│
├── api/                   # FastAPI backend (kaggle-api service)
├── core/                  # Shared utilities
│   ├── notify.py          # Telegram + bus delivery
│   ├── ask_human.py       # Blocking ask-and-wait
│   ├── message_bus.py     # SQLite chat bus
│   ├── download_guard.py  # Disk-aware download gate
│   └── vram_manager.py    # GPU memory arbitration
├── scripts/               # Operational scripts
│   ├── start_agent.sh     # Launches Claude Code with competition prompt
│   ├── restart_agent.sh   # Idempotent restart
│   ├── wsl_startup.sh     # Boot-time bootstrapper
│   ├── check_claude_auth.sh # OAuth check (non-blocking)
│   └── hooks/             # Claude Code hooks (Stop, PreCompact, PostBash)
├── tray/                  # Windows tray app
│   ├── kaggle_tray.py     # pystray icon + menu
│   ├── start_tray.bat     # Launcher (referenced by Startup shortcut)
│   └── assets/icon.svg    # Source-of-truth icon
├── systemd/               # systemd unit files (copied to /etc/systemd/system/)
├── competitions/          # Per-competition state, registry, CLAUDE.md files
├── agents/                # Sub-agent definitions
├── docs/                  # Project docs (incl. packaging-roadmap.md)
├── docker/                # Optional docker-compose stack
└── .claude/               # Claude Code state (settings, worktrees, hooks)
```

---

## Backup checklist

Things you can't easily reconstruct — back these up:

1. `.env` — all your secrets and tokens
2. `~/.cloudflared/` — tunnel UUID and cert
3. `~/.claude/.credentials.json` — Claude OAuth tokens (you can also
   just run `claude /login` again)
4. Postgres dump — `pg_dump keehar_kaggle > backup.sql` periodically
5. Qdrant — `~/qdrant_storage/` volume

The repo itself is on GitHub, no backup needed.

---

## See also

- `CLAUDE.md` — instructions Claude reads at every agent start
- `tray/README.md` — tray app details, icon scheme, menu
- `docs/packaging-roadmap.md` — future plan to ship as a signed
  installer (not active — kept for reference)
- `docs/changelog.md` — what changed when
- `docs/troubleshooting.md` — recurring bug fixes
