# Kaggle Agent

An autonomous Kaggle competition agent. It runs on a workstation with an
RTX 5070, picks up a competition, and works through the ML lifecycle
without supervision. I control it from my phone through Telegram.

The reasoning core is Claude Code. Around it sits a stack of services
(FastAPI, Postgres, Qdrant, MLflow, Ollama, a Telegram bot, a tray app)
that turn an interactive CLI assistant into something closer to a
24/7 worker process.

## What it does

Give it an active Kaggle competition (right now: ARC-AGI-2) and it
handles the whole loop:

1. Reads the competition. Pulls top public notebooks, scrapes the
   discussion forum, builds an "insights" collection in a Qdrant vector
   store before writing any code.
2. EDA. Distribution checks, missing values, type inference.
3. Builds a data pipeline (refine, enrich, synthetic augmentation, fuzz).
4. Iterates on models. LightGBM, XGBoost, CatBoost baselines. Optuna
   tuning. Ensembling on the top-N CV.
5. Tracks every experiment in MLflow with the config that produced it.
6. Asks me before any Kaggle submission. One approval, one submission.

Most of the *implementation* work (training loops, sklearn pipelines,
docstrings, boilerplate features) is routed to a local Ollama model
(qwen3:14b) so the API bill stays near zero. Claude is reserved for
strategy, surprising results, and architecture calls.

## Why I built it this way

The agent itself is a thin layer over Claude. The actually hard part was
making it survive its environment: WSL2 quirks, OAuth tokens expiring at
3am, the operator being on the subway with no laptop, services crashing
on reboot. Most of what's in this repo is the surrounding scaffolding
that keeps the agent alive and reachable.

A few decisions worth flagging:

* All operator-facing surfaces (Telegram, the web dashboard, the tray
  menu) read and write the same SQLite message bus. The agent polls it.
  No separate websocket layer, no message broker. One table is enough
  when the producer and consumer are both inside one machine.
* The agent runs inside `tmux`, not as a daemon. This was a pragmatic
  choice: when I SSH in I can attach and watch it think.
* Claude is launched with `--dangerously-skip-permissions` because every
  permission prompt requires a human at the keyboard. The agent has
  guardrails elsewhere (download size gates, submission approval, VRAM
  arbitration) so this trade is workable.

## Architecture

```
+- Windows host --------------------------------------------------+
|                                                                 |
|   System tray (pystray)    status + model swap + shutdown       |
|   Tailscale                remote SSH to the box                |
|                                                                 |
|   +- WSL2 (Ubuntu) -------------------------------------+       |
|   |                                                     |       |
|   |   tmux session "kaggle-agent"                       |       |
|   |     -> Claude Code  (the reasoning loop)            |       |
|   |                                                     |       |
|   |   systemd services:                                 |       |
|   |     - kaggle-api          FastAPI on :8765          |       |
|   |     - telegram-bot        slash cmds + chat ingest  |       |
|   |     - telegram-enforcer   guarantees delivery       |       |
|   |     - mlflow              experiment tracker        |       |
|   |     - ollama              local LLM (qwen3:14b)     |       |
|   |     - leaderboard-monitor hourly rank-change pings  |       |
|   |     - kaggle-api-watch    hot-reloads API on edits  |       |
|   |                                                     |       |
|   |   Postgres + Qdrant on a Tailscale-attached host    |       |
|   +-----------------------------------------------------+       |
|                                                                 |
+-----------------------------------------------------------------+
         |                       ^
         | Cloudflare Tunnel     | python-telegram-bot
         v                       |
   Public web dashboard    Telegram bot <-> me
```

## Tech stack

| Layer | Tools |
|---|---|
| AI / reasoning | Claude (via Claude Code CLI), Ollama (qwen3:14b) |
| Backend | Python 3.12, FastAPI, Pydantic, httpx |
| Operator UI | python-telegram-bot, pystray (Windows tray) |
| Storage | SQLite (message bus), Postgres (state + memory), Qdrant (vector store) |
| Experiment tracking | MLflow |
| Orchestration | systemd, tmux, bash, PowerShell |
| Networking | Cloudflare Tunnel, Tailscale, WSL2 |
| ML libs | LightGBM, XGBoost, CatBoost, Optuna, scikit-learn, PyTorch |

## Telegram surface

Slash commands handled locally in the bot, not forwarded to the agent
(so checking disk space doesn't burn Claude tokens):

| Command | Behavior |
|---|---|
| `/status` | Agent state + active competition |
| `/config` | Diagnostic: model, services, account, OAuth expiry |
| `/model` | Inline keyboard of available Claude models. Picking one restarts the agent. |
| `/login` | Drives `claude /login` over Telegram when the OAuth refresh token dies |
| `/restart` | Restart the agent process |
| `/pause`, `/resume`, `/stop` | State controls |
| `/help` | Command list |

Plus file uploads. Drop a .md, .py, .csv, .json into the chat and it
lands in an `inbox/` directory with the agent told about it. Small text
files get their content inlined so the agent has context without
needing to call Read separately.

## Tray app

Lives in the Windows system tray, talks to the FastAPI backend across
the WSL boundary. The icon shows a colored dot over the project logo:
green (running), yellow (paused), grey (stopped), blue (transitioning),
red (API unreachable).

Right-click gives me:

* Pause / Resume / Stop
* Open Dashboard (Cloudflare-tunneled web UI)
* Open tmux session (spawns Windows Terminal attached to the agent)
* Model picker submenu
* Shutdown everything & quit (stops the agent, all services, and
  Ollama, which frees ~7GB of VRAM)
* Quit tray, leave the agent running

## Self-healing

A Windows Task Scheduler entry runs `wsl_startup.sh` at login. That
brings up Tailscale, all systemd services, and a fresh tmux session
with the agent in it. The tray's launcher (`start_tray.bat`)
independently calls `ensure_running.sh`, so just opening the tray from
a cold state also brings the stack up. The two paths are independent on
purpose, in case one fails.

`ensure_running.sh` is idempotent. It checks `pgrep` for a running
Claude process before deciding whether to restart, so it can run on
every tray launch without disrupting an active agent.

## Auth recovery flow

The most complex single feature. Claude Code's OAuth refresh token
eventually expires, and there's no way to refresh it programmatically
without the PKCE verifier (which Claude Code doesn't store). The
solution is to drive a full `claude /login` flow over Telegram:

1. I send `/login` in Telegram.
2. The bot spawns a `kaggle-login` tmux session, runs `claude /login`,
   waits for the 3-option login menu, presses Enter to pick
   subscription auth.
3. It polls the tmux pane for the OAuth URL, joining hard-wrapped lines
   (the TUI breaks the URL across ~6 rows of a fixed-width box).
4. Sends the assembled URL to me.
5. I open it on my phone, log in, copy the callback code.
6. I paste the code as my next Telegram message.
7. The bot injects it via `tmux paste-buffer` (using `send-keys`
   doesn't work; the `#` in the code gets mis-interpreted as a tmux
   key reference).
8. Watches `~/.claude/.credentials.json` for an mtime change, then
   restarts the agent.

Total recovery time is around 30 seconds, from anywhere with a phone.

## Setup

The system targets Windows 11 + WSL2 + an NVIDIA GPU. The full rebuild
guide is in [`docs/recovery.md`](docs/recovery.md). At a high level:

1. Install WSL2, Ubuntu, Tailscale, and Python on Windows
2. Install Python deps inside WSL (`pip install -r requirements.txt`)
3. Restore `.env` (Telegram bot token, Kaggle API key, Postgres DSN, etc.)
4. Run `claude /login` once
5. Install systemd unit files from `systemd/` into `/etc/systemd/system/`
6. Run the PowerShell shortcut installer once:
   ```powershell
   powershell -ExecutionPolicy Bypass -File `
     "\\wsl.localhost\Ubuntu-24.04\<repo-path>\scripts\install_shortcuts.ps1"
   ```

A clean rebuild from a blank Windows install takes about two hours,
mostly waiting on downloads.

## File layout

```
kaggle-agent/
+-- api/                   FastAPI backend
|   +-- main.py            All HTTP endpoints
+-- core/                  Shared logic
|   +-- notify.py          Telegram outbound + bus delivery
|   +-- ask_human.py       Blocking ask-and-wait
|   +-- message_bus.py     SQLite chat bus
|   +-- telegram_bot.py    Slash commands + chat ingest
|   +-- models_catalog.py  Source of truth for Claude models
|   +-- download_guard.py  Disk-aware download gate
|   +-- vram_manager.py    GPU memory arbitration
|   +-- memory.py          Postgres + Qdrant memory layer
+-- scripts/
|   +-- start_agent.sh     Launch Claude with current settings
|   +-- restart_agent.sh   Hard restart (kills claude, recreates tmux)
|   +-- ensure_running.sh  Idempotent boot; starts only what's missing
|   +-- shutdown_all.sh    Stops agent + services + Ollama
|   +-- wsl_startup.sh     Boot bootstrapper (Task Scheduler target)
|   +-- install_shortcuts.ps1   Creates Desktop / Start Menu shortcuts
|   +-- hooks/             Claude Code event hooks
+-- tray/
|   +-- kaggle_tray.py     pystray icon + menu
|   +-- start_tray.bat     Launcher (calls ensure_running.sh)
|   +-- assets/icon.svg    Source icon
+-- systemd/               Unit files for all background services
+-- competitions/          Per-competition state + registry
+-- docs/
|   +-- recovery.md        Disaster-recovery rebuild guide
|   +-- packaging-roadmap.md   Future plan to ship as a signed installer
|   +-- troubleshooting.md
+-- CLAUDE.md              Agent operating instructions
```

## Bugs I fixed that might be useful to know about

A short list of operational gotchas, with the fix:

* **WSL2 silently drops localhost port forwarding** under load. The tray
  loses its connection to `localhost:8765` even though the API is
  perfectly healthy. Fixed by having the tray fall back to the live WSL
  IP, discovered via `wsl.exe hostname -I`.
* **Telegram's Markdown parser eats underscores**, so filenames like
  `xgb_v2_fixed_lags.csv` arrive as `xgbv2fixedlags.cs`. Dropped
  `parse_mode: Markdown` and use plain text everywhere.
* **Claude Code's Stop hook can fire twice in quick succession**,
  sending duplicate "Agent paused" pings. Added a 90s cooldown file at
  `/tmp/kaggle_agent_stop_notified`.
* **`tmux send-keys` mis-interprets `#` in OAuth codes** as a key
  reference. The `/login` flow switched to `tmux set-buffer` +
  `paste-buffer`.
* **OAuth URLs hard-wrap across 6 TUI rows.** Even with `capture-pane
  -J`, tmux returns separate lines because the wraps are hard newlines
  in a fixed-width box, not soft wraps. Wrote a line-joining parser
  that recognizes URL-shape continuations.
* **`restart_agent.sh` was silently spawning nested Claude processes.**
  The original sent Ctrl-C and a bash command via `send-keys`, but
  Claude traps SIGINT, so the bash command just got typed into Claude's
  prompt as a query. Replaced with `pkill` + `tmux kill-session` +
  fresh create.

## Status

Daily-driver. The agent's been running on my workstation for ~3 months,
across reboots, OS updates, OAuth expiries, and WSL outages. It
currently manages one active competition (ARC-AGI-2) and a couple of
completed ones in archive state. Submissions require my explicit
approval and the system makes that hard to fat-finger.

## License

MIT. If you fork this, the systemd unit user names, paths in `core/
models_catalog.py`, and a few hardcoded references to my WSL distro
name will need adjusting.
