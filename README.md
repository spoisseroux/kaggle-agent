# Kaggle Agent

> An autonomous AI agent that competes in Kaggle ML competitions on its own —
> running 24/7 on a local GPU, controllable from anywhere via Telegram or a
> web dashboard, surviving reboots, auth expirations, and the operator's
> long absences.

Built around Claude Code as the reasoning engine, with a stack of services
(FastAPI, Postgres, Qdrant, MLflow, Ollama, Telegram) that turn an
interactive AI assistant into a persistent, monitorable, recoverable
worker.

---

## What it does

The agent is given an active Kaggle competition (e.g. ARC-AGI-2) and runs
the full ML lifecycle autonomously:

1. **Researches** the competition — pulls top public notebooks, scrapes
   discussion threads, builds an "insights database" in a Qdrant vector
   store before writing any code
2. **Explores the data** — EDA, distribution checks, missing-value
   analysis, feature-type inference
3. **Builds a pipeline** — refine, enrich, synthetic augmentation, fuzz
4. **Iterates on models** — LightGBM / XGBoost / CatBoost baselines,
   Optuna tuning, ensembling
5. **Tracks every experiment** in MLflow with full reproducibility metadata
6. **Asks the operator before any submission** — never burns a daily
   submission slot without explicit human approval

Most of the *implementation* work (boilerplate, training loops, feature
engineering) is delegated to a local Ollama model (qwen3:14b) to keep
operating costs near zero. Claude is reserved for strategy, interpretation,
and architecture decisions.

## Why it's interesting

This isn't a notebook — it's a distributed system with seven independent
services that have to cooperate without a central scheduler. The interesting
problems aren't ML; they're systems engineering on top of an AI core:

- **Multi-process state coordination** between a Windows tray app, a
  Linux daemon stack, and an external messaging API, all sharing a
  single SQLite message bus as the source of truth
- **Auth recovery via Telegram** — when the OAuth refresh token expires,
  the bot drives a `claude /login` flow inside a separate tmux session,
  extracts the auth URL from the TUI by parsing wrapped lines, sends it
  to the operator, and pipes their reply back as a paste buffer
- **Hot model swapping** — selecting a different Claude model from the
  tray or via `/model opus` in Telegram triggers a clean agent restart,
  resuming context from a Postgres checkpoint
- **Survives WSL2's broken localhost forwarding** — the tray auto-discovers
  the live WSL IP when `localhost:8765` stops resolving (a known WSL bug)
- **Disk-aware downloads** — the agent computes effective free space as
  `min(WSL VHDX free, Windows C: free)` because WSL2's virtual disk lives
  on the host filesystem; refuses any download that would fill C:
- **Resilient hooks** — Stop / PreCompact / PostToolUse hooks fire on
  every Claude action, with a cooldown to prevent notification spam and
  graceful degradation if a downstream service is offline

## Architecture

```
┌─ Windows host ───────────────────────────────────────────────────┐
│                                                                  │
│   System tray (pystray)  ←  status, model switch, shutdown menu  │
│   Tailscale              ←  remote tmux access via SSH           │
│                                                                  │
│   ┌─ WSL2 (Ubuntu) ────────────────────────────────────────┐    │
│   │                                                         │    │
│   │   tmux session "kaggle-agent"                           │    │
│   │     └─ Claude Code (the reasoning engine)               │    │
│   │                                                         │    │
│   │   systemd services:                                     │    │
│   │     • kaggle-api          FastAPI :8765                 │    │
│   │     • telegram-bot        slash commands + chat bridge  │    │
│   │     • telegram-enforcer   guarantees delivery to user   │    │
│   │     • mlflow              experiment tracker            │    │
│   │     • ollama              local LLM (qwen3:14b)         │    │
│   │     • leaderboard-monitor hourly rank-change alerts     │    │
│   │     • kaggle-api-watch    hot-reloads API on edits      │    │
│   │                                                         │    │
│   │   Postgres + Qdrant (Tailscale-attached docker host)    │    │
│   └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
         │                          ▲
         │ Cloudflare Tunnel        │ python-telegram-bot
         ▼                          │
   Public web dashboard      Telegram bot ←→ operator's phone
```

The unifying primitive is a **SQLite message bus** that all three
operator-facing UIs (Telegram, web, tray) read from and write to. The
agent polls it as part of its loop. This eliminates the usual
"how does the chat client talk to the running model" problem — every
component just inserts rows.

## Tech stack

| Layer | Tools |
|---|---|
| **AI / reasoning** | Anthropic Claude (via Claude Code CLI), Ollama (qwen3:14b) |
| **Backend** | Python 3.12, FastAPI, Pydantic, httpx |
| **Real-time UI** | python-telegram-bot, pystray (Windows tray) |
| **Storage** | SQLite (message bus), Postgres (state + memory), Qdrant (vector store) |
| **Experiment tracking** | MLflow |
| **Orchestration** | systemd, tmux, bash, PowerShell |
| **Networking** | Cloudflare Tunnel, Tailscale, WSL2 |
| **ML libs** | LightGBM, XGBoost, CatBoost, Optuna, scikit-learn, PyTorch |

## Feature highlights

### Telegram control surface

Eight slash commands, all auth-gated by chat ID:

| Command | What it does |
|---|---|
| `/status` | Agent state and active competition |
| `/config` | Full diagnostic — model, services, account, OAuth token expiry |
| `/model` | Inline keyboard with available Claude models (auto-restarts agent on change) |
| `/login` | Drives `claude /login` via Telegram when the OAuth refresh token dies |
| `/restart` | Restart the agent process |
| `/pause`, `/resume`, `/stop` | Agent state controls |
| `/help` | Command list |

Plus file uploads — any document sent to the chat gets saved to an
`inbox/` directory and the agent is notified with the file path and a
content preview (for small text files).

### Tray app

- Live status dot (running / paused / stopped / transitioning / offline)
  drawn with a soft glow effect over the project logo
- Model swap submenu — switching models auto-restarts the agent
- "Shutdown everything & quit" — stops agent + services + Ollama (frees
  VRAM) in one click
- Auto-discovers the API URL — tries `localhost:8765` first, falls back
  to the live WSL IP via `wsl hostname -I` when WSL's localhost
  forwarding breaks (a known glitch on long sessions)

### Self-healing on boot

A single Task Scheduler entry runs `wsl_startup.sh` at login, which
brings up Tailscale, all systemd services, and a fresh tmux session
running the agent. The tray launcher (`start_tray.bat`) independently
calls `ensure_running.sh` so just opening the tray from a cold state
also brings the stack up.

### Dynamic model catalog

The list of available Claude models lives in one place
(`core/models_catalog.py`). When Anthropic ships a new model, a daily
background check against OpenRouter compares their catalog against the
local list and sends a Telegram alert if a new tier (Haiku / Sonnet /
Opus) version exists. The operator updates one constant — both the
tray menu and the Telegram bot pick up the new model on next restart.

### Auth recovery flow

The most complex single feature. When the OAuth refresh token expires:

1. Operator sends `/login` in Telegram
2. Bot spawns a `kaggle-login` tmux session, runs `claude /login`
3. Waits for the 3-option login menu, presses Enter to select "Claude
   subscription"
4. Polls the pane for the OAuth URL, joining hard-wrapped lines
   from the TUI's fixed-width box
5. Sends the assembled URL via Telegram
6. Operator logs in on browser, copies the callback code, pastes it
   back in Telegram
7. Bot injects the code via `tmux paste-buffer` (not `send-keys` —
   the `#` separator in the code breaks `send-keys` parsing), watches
   `~/.claude/.credentials.json` for an mtime change, kills the login
   session, restarts the agent

Total recovery time: ~30 seconds, from anywhere with a Telegram client.

## Setup

The system targets Windows 11 + WSL2 + an NVIDIA GPU. See
[`docs/recovery.md`](docs/recovery.md) for a detailed step-by-step
rebuild guide. The high-level sequence:

1. Install WSL2 + Ubuntu + Tailscale + Python on Windows
2. Install repo dependencies inside WSL (`pip install -r requirements.txt`)
3. Restore `.env` (Telegram bot token, Kaggle API key, Postgres DSN, etc.)
4. Run `claude /login` for Claude Max OAuth
5. Install systemd unit files from `systemd/` to `/etc/systemd/system/`
6. Run the one-time PowerShell shortcut installer:
   ```powershell
   powershell -ExecutionPolicy Bypass -File `
     "\\wsl.localhost\Ubuntu-24.04\<repo-path>\scripts\install_shortcuts.ps1"
   ```

A clean rebuild from a blank Windows install takes about two hours,
mostly download time.

## File layout

```
kaggle-agent/
├── api/                   FastAPI backend
│   └── main.py            All HTTP endpoints (/system/*, /chat/*, /experiments/*)
├── core/                  Shared logic
│   ├── notify.py          Telegram outbound + bus delivery
│   ├── ask_human.py       Blocking ask-and-wait for human input
│   ├── message_bus.py     SQLite chat bus
│   ├── telegram_bot.py    Slash commands + chat ingest
│   ├── models_catalog.py  Single source of truth for Claude models
│   ├── download_guard.py  Disk-aware download gate (WSL VHDX-aware)
│   ├── vram_manager.py    GPU memory arbitration
│   └── memory.py          Postgres + Qdrant memory layer
├── scripts/
│   ├── start_agent.sh     Launches Claude Code with current settings
│   ├── restart_agent.sh   Hard restart (kills claude + recreates tmux)
│   ├── ensure_running.sh  Idempotent — starts only what isn't already up
│   ├── shutdown_all.sh    Stops agent + services + Ollama
│   ├── wsl_startup.sh     Boot-time bootstrapper (Task Scheduler entry)
│   ├── install_shortcuts.ps1   Creates Desktop/Start Menu/Startup shortcuts
│   └── hooks/             Claude Code event hooks
├── tray/
│   ├── kaggle_tray.py     pystray icon + menu
│   ├── start_tray.bat     Launcher (calls ensure_running.sh)
│   └── assets/icon.svg    Source-of-truth icon
├── systemd/               Unit files for all background services
├── competitions/          Per-competition state, registry, instructions
├── docs/
│   ├── recovery.md        Disaster-recovery rebuild guide
│   ├── packaging-roadmap.md  Future plan to ship as a signed installer
│   └── troubleshooting.md
└── CLAUDE.md              Agent operating instructions (read at every start)
```

## Engineering challenges worth a story

A few of the more interesting bugs and fixes from the build, in case
they're useful as interview talking points:

- **WSL2 silently drops localhost forwarding** under load, breaking the
  tray's `localhost:8765` calls. Fixed by auto-discovering the live WSL
  IP via `wsl.exe hostname -I` and probing both candidates on tray
  startup.
- **Telegram silently mangles underscores in Markdown mode**, so
  filenames like `xgb_v2_fixed_lags.csv` arrive as `xgbv2fixedlags.cs`.
  Removed `parse_mode: Markdown` entirely — plain text is reliable and
  the operator can still read it.
- **Claude Code's Stop hook fires twice in quick succession** on certain
  exits, sending duplicate "Agent paused" pings. Added a 90-second
  cooldown file at `/tmp/kaggle_agent_stop_notified`.
- **`tmux send-keys` mis-interprets `#` in OAuth codes**. Switched the
  `/login` flow to `tmux set-buffer` + `paste-buffer`.
- **OAuth URL hard-wraps across 6 TUI rows** — even with `capture-pane
  -J`, tmux returns separate lines. Wrote a line-joining parser that
  detects URL-shape continuations.
- **`restart_agent.sh` was silently spawning nested Claude processes** —
  the old version sent Ctrl-C and a bash command via `send-keys`, but
  Claude traps SIGINT and the bash command got typed into Claude's
  prompt as a query, spawning a new Claude inside the old one. Replaced
  with a clean pkill + tmux kill-session + fresh create.

## Status

Actively used as the operator's daily-driver for Kaggle competitions.
The agent currently:

- Runs on a workstation with an RTX 5070
- Manages 2+ concurrent competitions
- Submits with explicit human approval only
- Auto-recovers from auth expiry, service crashes, and OS reboots
- Survives ~24/7 uptime with periodic operator interactions via phone

## License

MIT — see [LICENSE](LICENSE) if present, otherwise default MIT terms
apply. If you fork this, the `core/models_catalog.py` constants and the
systemd unit user names will need to be adjusted for your environment.

## Acknowledgements

Built on top of [Claude Code](https://claude.com/claude-code) (Anthropic),
with significant Claude-assisted code generation throughout the
development process — many of the operational reliability fixes
documented above were debugged in collaboration with Claude itself,
which seemed appropriate.
