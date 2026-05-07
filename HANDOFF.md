# Kaggle Agent — Handoff

This file documents the **as-built** state (Phase 5 + Phase 6 updates). Where
it diverges from `kaggle-agent-PRD-final.md`, this file wins. Pass this to the
agency-platform Web UI session along with PRD §11 and §17 for the UI spec.

---

## API base URL

| Environment | URL |
|---|---|
| Local (this WSL2) | `http://localhost:8765` |
| Tailscale peer | `http://kaggle:8765` (peers on the same tailnet) |
| Public (Cloudflare Tunnel) | **NOT YET CONFIGURED** — see *Known issues / TODOs* |

Until the tunnel is wired up, the web UI must run on the same Tailscale
network and use `http://kaggle:8765` (or `http://<tailscale-ip>:8765`).

## Authentication
- **Web UI → API:** open. CORS is `*` unless `ALLOWED_ORIGINS` is set in
  `.env`. Set it before exposing publicly.
- **API → memory stack:** Postgres uses the credentials in `.env`
  (`POSTGRES_DSN`); MCP uses `MCP_BEARER_TOKEN` from `.env`.

---

## Confirmed working endpoints

### `GET /health`
Live response captured at end of build:
```json
{
  "status": "online",
  "services": {
    "postgres": true, "qdrant": true, "mcp": true,
    "ollama": true, "telegram": true, "mlflow": true
  },
  "active_competition": null,
  "gpu": {
    "name": "NVIDIA GeForce RTX 5070",
    "vram_total_mb": 12227, "vram_used_mb": 3042, "vram_free_mb": 9184,
    "gpu_util_pct": 0, "mem_util_pct": 19, "temp_c": 38, "power_w": 8.903,
    "process_pids": [], "ts": 1778124392.36
  },
  "uptime_s": 184,
  "last_updated": "2026-05-07T03:26:32Z"
}
```

---

### State machine — pause / resume / stop / checkpoint

Four endpoints control the agent's lifecycle without ever touching the
FastAPI service or Telegram bot themselves (those must always stay up so
the user can issue commands).

| Method | Path | Purpose |
|---|---|---|
| GET | `/system/state` | current state + run stage + active competition |
| POST | `/system/pause` | freeze agent processes via cgroupv2 + stop Ollama |
| POST | `/system/resume` | thaw / recreate agent + start Ollama |
| POST | `/system/stop` | SIGTERM agent processes + kill tmux + stop Ollama |
| POST | `/system/checkpoint` | signal agent to save mid-run state before stopping |

States: `running`, `paused`, `stopped` (plus `transitioning` shown
purely client-side while a request is in flight). Persisted to
`state.json` in the repo root and reconciled on read.

Implementation detail: pause uses cgroupv2 `cgroup.freeze` rather than
`SIGSTOP`. On WSL2 with systemd, SIGSTOP delivered cross-cgroup is
silently ignored. cgroup.freeze works reliably and is owned by the user.

#### `GET /system/state` — full schema

```json
{
  "state": "running",
  "since": 1778125260.9,
  "active_competition": "titanic",

  "run_stage": "training",
  "run_stage_detail": "epoch 14/50",
  "run_eta_seconds": 840,
  "run_started_at": "2026-05-07T05:30:00Z"
}
```

`run_stage` is only present when the agent has set it. Possible values:

| `run_stage` | Meaning |
|---|---|
| `idle` | nothing running (or key absent) |
| `downloading` | fetching competition data |
| `eda` | exploratory data analysis |
| `training` | model training loop |
| `generating_submission` | creating submission file |
| `submitting` | uploading to Kaggle |
| `done` | run completed |

UI: show a progress banner when `state == "running"` and `run_stage` is
present and not `idle`/`done`. Format ETA as "~14m left". Strip the
banner when state transitions away from running.

When the agent crashes or is killed externally:
```json
{ "state": "stopped", "since": 1778125261.0, "note": "session_missing", "active_competition": null }
```

#### `POST /system/checkpoint`
Signals the agent to save mid-run state (useful before issuing Stop).
Also forwards a wake-key to the tmux session so the agent sees it quickly.
```json
{ "state": "running", "checkpoint_requested": true, "checkpoint_requested_at": "2026-05-07T12:00:00Z" }
```
UI: show a toast "Checkpoint requested" with no further action needed.

---

### Model selection

| Method | Path | Purpose |
|---|---|---|
| GET | `/system/model` | current model + available list |
| POST | `/system/model` | switch model (takes effect on next agent restart) |

```json
// GET /system/model
{ "model": "claude-sonnet-4-6", "available": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"] }

// POST /system/model  body: {"model": "claude-opus-4-7"}
{ "model": "claude-opus-4-7", "restart_required": true, "note": "Model takes effect when the agent session restarts (stop + resume)" }
```

Model is persisted to `.claude/kaggle_settings.json` and read by
`scripts/start_agent.sh` at startup via `claude --model <id>`. The model
cannot change mid-session — the agent must be stopped and resumed.

UI: show a dropdown/radio with the three models. After selecting, call
POST and show "Restart agent to apply" notice if `restart_required: true`.

---

### Chat — message behaviour (Phase 6 update)

**Immediate acknowledgment:** when a human message arrives (from Telegram
or web), the system posts `"Got it, thinking..."` to the message bus
*before* the agent has time to respond. The WS `/chat/ws` broadcasts this
within ~1 s — no extra UI work needed.

**Agent wake:** alongside the ack, the backend sends the message text to
Claude Code's stdin via `tmux send-keys`. This means messages no longer
sit unread in the SQLite queue when the agent is idle at its `>` prompt.

| Method | Path | Notes |
|---|---|---|
| GET | `/chat/messages?limit=50` | recent messages, oldest first |
| POST | `/chat/send` | body: `{text}` — posts ack, wakes agent, forwards to Telegram |
| WS | `/chat/ws` | initial dump of last 50, then live; send `{"text":"..."}`, get `{"ack":"<id>"}` |

Chat message schema:
```json
{
  "id": "...",
  "role": "agent" | "human",
  "source": "telegram" | "web" | "agent" | "system",
  "text": "...",
  "status": "new" | "claimed" | "consumed" | "delivered",
  "ask_id": null,
  "created_at": 1778125260.9
}
```

---

### Terminal WebSocket

`WS /terminal/ws` — streams live tmux pane output; also accepts keystrokes.

**Server → client:**
```json
{ "type": "output", "data": "<pane content>", "ts": "2026-05-07T12:00:00Z" }
{ "type": "status", "data": "session offline", "ts": "..." }
{ "type": "error",  "data": "<error message>", "ts": "..." }
```

**Client → server (keyboard input):**
```json
{ "type": "input", "data": "<keystrokes>" }
```
The server calls `tmux send-keys -t kaggle-agent <data>` — no newline is
appended automatically. To submit a command, include `\n` in the data string.

Note: the `/terminal/ws` path is a WebSocket endpoint. A plain HTTP GET to
it will return 404 from FastAPI — this is expected behaviour. Connect via
the WebSocket protocol (`ws://` or `wss://`).

---

### Other endpoints — same shapes as PRD §10

| Method | Path | Notes |
|---|---|---|
| GET  | `/system/ssh-info` | tailscale hostname, ssh + tmux attach commands |
| GET  | `/competitions` | falls back to local registry.json if Postgres has no rows |
| GET  | `/competitions/active` | reads `competitions/registry.json` for the active slug |
| GET  | `/competitions/active/trajectory` | 404 if no active competition |
| POST | `/competitions/new` | body: `{slug, metric, deadline, name?, higher_better?, submissions_max?}` |
| POST | `/competitions/switch` | body: `{slug}` |
| GET  | `/experiments?competition=<slug>&limit=50` | rows from `kaggle_experiments` |
| GET  | `/submissions?competition=<slug>` | rows from `kaggle_submissions` |
| GET  | `/leaderboard/{slug}` | shells out to Kaggle CLI |
| GET  | `/logs?lines=200` | tails every `logs/*.log` |
| GET  | `/gpu/stream` | Server-Sent Events, GPU snapshot every 3 s |
| GET  | `/ollama/status` | `{ok, models}` |
| GET  | `/memory/status` | Postgres/Qdrant/MCP/Tailscale latency and status |

---

## Standalone kaggle-ui repository spec

**Repository:** separate Next.js 14+ App Router project deploying to Vercel
**Production URL:** `https://kaggle-ui.nnaq.net`
**Design:** black and white monospace throughout (IBM Plex Mono, Roboto Mono, or JetBrains Mono)
**Terminal:** xterm.js for the `/terminal` page, full-screen black background with white text

### Environment variables (Vercel)

```bash
NEXT_PUBLIC_KAGGLE_API_URL=https://kaggle-ui.nnaq.net
NEXT_PUBLIC_KAGGLE_WS_URL=wss://kaggle-ui.nnaq.net
PASSWORD=your-secure-password-here
```

For local development:
```bash
NEXT_PUBLIC_KAGGLE_API_URL=http://localhost:8765
NEXT_PUBLIC_KAGGLE_WS_URL=ws://localhost:8765
PASSWORD=dev
```

### Authentication

Simple middleware password check using `PASSWORD` env var. On first visit, show a
centered password prompt (black background, white monospace input). Store the
password in a secure httpOnly cookie after validation. Middleware checks the
cookie on every request. No user accounts, no database — single shared password.

### Pages

| Route | Purpose |
|-------|---------|
| `/` or `/dashboard` | Overview: system state with run-stage progress banner, active competition, GPU snapshot, quick controls (Pause/Resume/Stop/Checkpoint), model selector |
| `/terminal` | Full-screen xterm.js terminal connected to `WS /terminal/ws`, sends keystrokes back to tmux, auto-reconnect |
| `/chat` | Chat interface connected to `WS /chat/ws`, message history, send box at bottom |
| `/experiments` | Table from `GET /experiments`, filterable by competition, sortable by CV score |
| `/gpu` | Real-time GPU monitor from `GET /gpu/stream` (SSE), live charts |
| `/competitions` | List from `GET /competitions`, show active, switch via `POST /competitions/switch` |
| `/logs` | Tail of agent logs from `GET /logs?lines=200`, auto-refresh every 5 s |
| `/submissions` | Table from `GET /submissions`, history per competition |
| `/leaderboard` | From `GET /leaderboard/{slug}` |
| `/memory` | Memory stack status from `GET /memory/status` |
| `/settings` | Model selector (GET/POST `/system/model`), API info |

### Dashboard — run stage banner

Poll `GET /system/state` every 5 s. When `state == "running"` and
`run_stage` is present and not `idle`/`done`, show a banner:

```
🏋 Training  epoch 14/50  ~14m left       [Checkpoint]
```

Clear the banner when `run_stage` is absent, `idle`, or `done`, or when
state is not `running`. The `[Checkpoint]` button calls `POST /system/checkpoint`.

### Dashboard — controls

| Button | API call | Enabled when |
|---|---|---|
| Pause | POST /system/pause | state == running |
| Resume | POST /system/resume | state == paused or stopped |
| Stop | POST /system/stop | state == running or paused |
| Checkpoint | POST /system/checkpoint | state == running and run_stage not idle |

Show confirmation dialog before Stop (destructive — kills Claude Code session).
Optimistic UI: set badge to "transitioning" immediately on click.

### Terminal page implementation

Use xterm.js with `fit` addon:
1. Connect to `WS /terminal/ws`
2. On `type: "output"` frames: call `terminal.write(data)` — the pane content
   includes ANSI escape codes since tmux is captured with `-e`
3. On `type: "status"` with `data: "session offline"`: show an overlay banner
4. On `type: "error"`: log to console, show brief toast
5. On user keystrokes: send `{"type": "input", "data": event.key}` back to server
6. Auto-reconnect with exponential backoff (1 s, 2 s, 4 s, 8 s max)

### Model selector UI

Dropdown or radio group showing:
- Haiku 4.5 · fast/cheap
- Sonnet 4.6 · balanced ← default
- Opus 4.7 · powerful

After selecting: call `POST /system/model`, then show a dismissable notice
"Model updated — stop and resume the agent to apply."

### Design guidelines

- **Typography:** monospace only (IBM Plex Mono, Roboto Mono, or JetBrains Mono)
- **Colors:** pure black (#000) background, pure white (#FFF) text, gray (#666, #999) for secondary
- **Layout:** sidebar nav on left (~200px fixed), main content area on right
- **Tables:** simple borders, alternating row backgrounds (#111 vs #000), sortable columns
- **Terminal:** xterm.js full-screen, black bg, white text, 80×24 default, fit-addon to resize
- **Charts (GPU page):** Chart.js dark theme or plain ASCII charts
- **Status indicators:** colored dots (green/yellow/red/gray/blue) matching tray app colours
- **Buttons:** simple bordered rectangles, white border, white text, hover inverts
- **No animations** except loading spinners (simple CSS keyframes)

### Key implementation notes

1. **Poll `/health` every 10 s** (3 s timeout) to drive global status indicator in sidebar.
2. **Poll `/system/state` every 5 s** separately — lighter than health, needed for run-stage banner.
3. **Auto-reconnect WebSockets** with exponential backoff (1s → 2s → 4s → 8s max).
4. **Terminal xterm.js:** use `terminal.write()` not `terminal.writeln()` — tmux output already has `\r\n`.
5. **Chat page:** initial history dump on connect, append new messages as they arrive, auto-scroll to bottom.
6. **GPU page:** use `EventSource`, parse `data:` lines as JSON, update charts live.
7. **"Got it, thinking..."** messages arrive as `role: "agent", source: "system"` — render with a distinct style (e.g., italic gray) to distinguish from real agent responses.

### Deploy to Vercel

```bash
vercel env add NEXT_PUBLIC_KAGGLE_API_URL
vercel env add NEXT_PUBLIC_KAGGLE_WS_URL
vercel env add PASSWORD
vercel --prod
```

Custom domain `kaggle-ui.nnaq.net` → Vercel. API backend (`http://kaggle:8765` on Tailscale
or Cloudflare Tunnel) has CORS for `https://*.vercel.app` and `https://kaggle-ui.nnaq.net`.

---

## Memory stack (Tailscale VM 'docker')

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | `claude_memory` database with kaggle_* tables |
| Qdrant | 6333 | Vector collections: kaggle_features, kaggle_experiments, kaggle_insights, kaggle_errors, claude_memory |
| MCP server | 8000 | Memory MCP server with bearer token auth |

```bash
TAILSCALE_MEMORY_HOST=docker
POSTGRES_DSN=postgresql://claude:PASSWORD@docker:5432/claude_memory
QDRANT_URL=http://docker:6333
MCP_URL=http://docker:8000
MCP_BEARER_TOKEN=...
```

---

## Confirmed SSH commands
```bash
ssh keehar@kaggle
ssh keehar@kaggle -t tmux attach -t kaggle-agent
```

---

## Windows tray app (Phase 6)

Native Windows app in `tray/`. **Run `tray/start_tray.bat`** — now uses
`tray_launcher.py` which watches `kaggle_tray.py` for saves and auto-restarts
the tray. No manual restart needed after code changes.

**Icon colours:**

| Colour | Meaning |
|---|---|
| 🟢 green | running |
| 🟡 yellow | paused |
| ⚪ gray | stopped |
| 🔵 blue | transitioning |
| 🔴 red | API unreachable |

**Tooltip when running** shows the run stage and ETA:
```
Kaggle Agent — running  (titanic)
🏋 Training epoch 14/50, ~14m left
```

**Menu:**
- Status label with run-stage badge
- Pause / Stop / Resume
- Open Dashboard (browser)
- Open tmux session (opens Windows Terminal or cmd with `wsl tmux attach`)
- **Model** submenu — radio checkmarks for Haiku/Sonnet/Opus; calls `POST /system/model`
- **Start on startup** — checked checkbox; toggles the `"Kaggle Agent"` Task Scheduler task via `schtasks`
- Quit tray

**Task Scheduler task** (`"Kaggle Agent"`): runs at logon, executes
`wsl.exe -d Ubuntu-24.04 -- bash /home/keehar/kaggle-agent/scripts/wsl_startup.sh`.

**Auto-restart:** `tray_launcher.py` polls `kaggle_tray.py` mtime every 2 s.
On file change: terminates the old tray process, starts a new one. This means
editing and saving `kaggle_tray.py` on the Windows side is enough — the tray
picks up the change within ~3 s.

---

## Backend auto-restart on code changes (Phase 6)

`systemd/kaggle-api-watch.service` runs `scripts/watch_backend.sh`. This
script uses `inotifywait` to watch `api/` and `core/` for `.py` file changes.
On change: 1 s debounce, then `systemctl restart kaggle-api`. Installed and
enabled by the deploy commands; starts automatically at boot.

To verify it's running:
```bash
sudo systemctl status kaggle-api-watch
```

inotify-tools (`inotifywait`) must be installed: `sudo apt-get install -y inotify-tools`.
Already confirmed installed on this machine.

---

## Startup flow (Phase 6)

`scripts/wsl_startup.sh` is designed to run **headlessly** — all output goes
to `logs/startup.log`. The WSL terminal window started by Task Scheduler can
be closed immediately; all processes run in detached tmux or systemd services.

Flow:
1. Tailscale up (idempotent)
2. `systemctl start telegram-bot kaggle-api mlflow ollama`
3. 5 s wait → `POST /system/resume` (sets state = running)
4. `tmux new-session -d -s kaggle-agent` running `scripts/start_agent.sh`
5. `systemctl start kaggle-api-watch`

The agent session is a detached tmux session — closing any terminal window
has no effect. Access it at any time via the tray's "Open tmux session" or:
```bash
wsl -d Ubuntu-24.04 -e bash -c "tmux attach -t kaggle-agent"
```

---

## Model selection (Phase 6)

Agent model is stored in `.claude/kaggle_settings.json`:
```json
{ "model": "claude-sonnet-4-6" }
```

`scripts/start_agent.sh` reads this and passes `--model <id>` to `claude`.
Change via:
- Tray → Model submenu
- `POST /system/model {"model": "claude-opus-4-7"}`
- Direct edit of `.claude/kaggle_settings.json`

Then Stop + Resume the agent session to apply.

---

## Deviations from PRD

| What | PRD | Built |
|---|---|---|
| `TRAINING_MIN_FREE_MB` | 9500 | **9000** — Windows reserves ~3 GB on this WSL2 install |
| Ollama install location | `/usr/local/bin/ollama` | `~/.local/bin/ollama` |
| Cloudflare Tunnel | configured Phase 4 | **Deferred** |
| `huggingface_hub` API | `list_datasets(tags=...)` | `list_datasets(filter=...)` |

---

## Known issues / TODOs

1. **Cloudflare Tunnel not yet wired up.** Run once:
   ```bash
   sudo dpkg -i cloudflared.deb
   cloudflared tunnel login
   cloudflared tunnel create kaggle
   cloudflared tunnel route dns kaggle kaggle.<your-domain>
   sudo cloudflared service install
   ```

2. **Kaggle auth:** `.env` must have `KAGGLE_KEY` and `KAGGLE_USERNAME`. Never use `kaggle.json`.

3. **`pynvml` deprecation warning.** Cosmetic — suppress with `PYTHONWARNINGS=ignore::FutureWarning`.

4. **Task Scheduler "Kaggle Agent" task** was created with admin rights; the tray app can toggle
   it enabled/disabled but cannot recreate it from scratch without admin PowerShell.

5. **MLflow served on `127.0.0.1:5000`** — not exposed via API. Add a proxy if needed.

6. **Per-competition CLAUDE.md is a stub.** Agent should expand it after registration.

---

## Quick smoke test
```bash
curl -sS http://localhost:8765/health   | jq .status
curl -sS http://localhost:8765/system/state | jq .
curl -sS http://localhost:8765/system/model | jq .
python core/notify.py "handoff smoke test"
```

## How to start the agent
```bash
# Attach to the running tmux session:
tmux attach -t kaggle-agent

# Or start fresh:
bash scripts/wsl_startup.sh
```
