# Kaggle Agent — Handoff

This file documents the **as-built** state at the end of Phase 5. Where it
diverges from `kaggle-agent-PRD-final.md`, this file wins. Pass this to the
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

### State machine — pause / resume / stop

Four endpoints control the agent's lifecycle without ever touching the
FastAPI service or Telegram bot themselves (those must always stay up so
the user can issue commands).

| Method | Path | Purpose |
|---|---|---|
| GET | `/system/state` | current state + when it was set |
| POST | `/system/pause` | freeze agent processes via cgroupv2 + stop Ollama |
| POST | `/system/resume` | thaw / recreate agent + start Ollama |
| POST | `/system/stop` | SIGTERM agent processes + kill tmux + stop Ollama |

States are: `running`, `paused`, `stopped` (plus `transitioning` shown
purely client-side while a request is in flight). Persisted to
`state.json` in the repo root and reconciled on read — if `state.json`
says `running` but the tmux session is gone, the API reports `stopped`
with `note: "session_missing"`. **No state-changing endpoint clears the
note**; only fresh transitions do.

Implementation detail worth knowing for the UI: pause uses cgroupv2
`cgroup.freeze` rather than `SIGSTOP`. On WSL2 with systemd, SIGSTOP
delivered cross-cgroup is silently ignored (a CPU-bound python keeps
running at 100% after `kill -STOP` returns success). cgroup.freeze works
reliably and is owned by the user, so no sudo is needed.

#### `GET /system/state` — example
```json
{
  "state": "running",
  "since": 1778125260.9,
  "active_competition": null
}
```
When the agent has crashed or been killed externally:
```json
{
  "state": "stopped",
  "since": 1778125261.0,
  "note": "session_missing",
  "active_competition": null
}
```

#### `POST /system/pause` — example response
```json
{
  "state": "paused",
  "since": 1778125257.7,
  "paused_pids": 1,
  "frozen_scopes": 1,
  "active_competition": null
}
```
Side effects: every PID in the `kaggle-agent` tmux session has its cgroup
frozen; Ollama service is stopped; a Telegram message `⏸ Agent paused.`
is sent.

#### `POST /system/resume` — example response
```json
{
  "state": "running",
  "since": 1778125260.9,
  "resumed_pids": 1,
  "thawed_scopes": 1,
  "active_competition": null
}
```
If the tmux session is missing (we were `stopped`), this endpoint runs
`tmux new-session -d -s kaggle-agent ... bash scripts/start_agent.sh`
to recreate it, then thaws the scope and starts Ollama. Telegram
notification: `▶ Agent resumed.`

#### `POST /system/stop` — example response
```json
{
  "state": "stopped",
  "since": 1778125274.6,
  "terminated_pids": 1,
  "active_competition": null
}
```
SIGCONT (in case it was paused) → SIGTERM → 1 second grace → SIGKILL →
`tmux kill-session` → `systemctl stop ollama`. Telegram notification:
`■ Agent stopped.`

### `GET /system/ssh-info`
```json
{
  "tailscale_hostname": "kaggle",
  "ssh_command": "ssh keehar@kaggle",
  "tmux_command": "ssh keehar@kaggle -t tmux attach -t kaggle-agent",
  "tailscale_connected": true
}
```

### `GET /memory/status`
```json
{
  "postgres": { "status": "online", "host": "docker", "db": "claude_memory",
                "tables_count": 5, "latency_ms": 146 },
  "qdrant":   { "status": "online", "host": "docker",
                "collections": [
                  {"name": "kaggle_features", "vectors_count": 0},
                  {"name": "claude_memory",   "vectors_count": 106},
                  {"name": "kaggle_insights", "vectors_count": 0},
                  {"name": "kaggle_errors",   "vectors_count": 0},
                  {"name": "kaggle_experiments","vectors_count": 0}
                ] },
  "mcp":       { "status": "online", "host": "docker", "latency_ms": 112 },
  "tailscale": { "connected": true, "latency_ms": 146 }
}
```

### `GET /ollama/status`
```json
{ "ok": true, "models": [{"name": "qwen3:14b", ...}] }
```

### Other endpoints — same shapes as PRD §10
| Method | Path | Notes |
|---|---|---|
| GET  | `/competitions` | falls back to local registry.json if Postgres has no rows |
| GET  | `/competitions/active` | reads `competitions/registry.json` for the active slug |
| GET  | `/competitions/active/trajectory` | 404 if no active competition |
| POST | `/competitions/new` | body: `{slug, metric, deadline, name?, higher_better?, submissions_max?}` |
| POST | `/competitions/switch` | body: `{slug}` |
| GET  | `/experiments?competition=<slug>&limit=50` | rows from `kaggle_experiments` |
| GET  | `/submissions?competition=<slug>` | rows from `kaggle_submissions` |
| GET  | `/leaderboard/{slug}` | shells out to the Kaggle CLI; needs `KAGGLE_KEY` + `KAGGLE_USERNAME` in env |
| GET  | `/logs?lines=200` | tails every `logs/*.log` |
| GET  | `/chat/messages?limit=50` | recent messages (oldest first) |
| POST | `/chat/send` | body: `{text}` — also forwards to Telegram as `[web] <text>` |

### Streaming endpoints

| Endpoint | Protocol | Cadence |
|---|---|---|
| `GET /gpu/stream` | Server-Sent Events (`text/event-stream`) | snapshot every 3 s |
| `WS  /chat/ws` | WebSocket — JSON messages | initial dump of last 50, then live |

WebSocket client framing — incoming events from server look like
`{ "id": ..., "role": "agent"|"human", "source": "telegram"|"web"|"agent",
"text": ..., "status": ..., "ask_id": ..., "created_at": ... }`. Outgoing
client messages must be `{"text": "..."}`. Server replies with
`{"ack": "<msg_id>"}` after persisting.

---

## Frontend env vars
```
NEXT_PUBLIC_KAGGLE_API_URL=http://kaggle:8765        # while on Tailscale
# When the Cloudflare Tunnel is up, switch to:
# NEXT_PUBLIC_KAGGLE_API_URL=https://kaggle.<your-domain>
```
The UI should poll `/health` every 10 s with a 3 s timeout to drive the
sidebar status dot (online / degraded / offline). On WebSocket disconnect,
go offline immediately without waiting for the next poll.

---

## Confirmed SSH commands
On any device on the same Tailscale tailnet:
```bash
ssh keehar@kaggle
ssh keehar@kaggle -t tmux attach -t kaggle-agent
```
Tailscale SSH is enabled (`tailscale up --ssh`). The `kaggle-agent` tmux
session is created by `scripts/wsl_startup.sh` at WSL boot.

---

## Deviations from PRD

| What | PRD | Built |
|---|---|---|
| `TRAINING_MIN_FREE_MB` | 9500 | **9000** — Windows reserves ~3 GB on this WSL2 install rather than the PRD's assumed 1.5 GB, so post-Ollama-stop free is ~9.2 GB. |
| Ollama install location | `/usr/local/bin/ollama` (via the Ollama install script) | `~/.local/bin/ollama → ~/.local/lib/ollama/bin/ollama` — the install script wants broad sudo we don't grant. systemd unit points at the user-local path. |
| Cloudflare Tunnel | configured during Phase 4 | **Deferred** — `cloudflared.deb` is in the repo but `cloudflared tunnel login` is an interactive browser flow. See *Known issues / TODOs*. |
| `huggingface_hub` API | `list_datasets(tags=...)` | `list_datasets(filter=...)` — `tags` was renamed in 1.x. The signature of `core.hf_search.search_relevant_datasets(query, tags=...)` is unchanged. |
| `competitions/registry.json` | starts empty | Same — initialised to `{"active": null, "competitions": {}}`. |

---

## Known issues / TODOs

1. **Cloudflare Tunnel not yet wired up.** Run once on this box:
   ```bash
   sudo dpkg -i cloudflared.deb
   cloudflared tunnel login
   cloudflared tunnel create kaggle
   cloudflared tunnel route dns kaggle kaggle.<your-domain>
   sudo cloudflared service install
   ```
   Then update `.env`:
   ```
   CLOUDFLARE_TUNNEL_URL=https://kaggle.<your-domain>
   ALLOWED_ORIGINS=https://<your-agency-platform-host>
   ```
   and `sudo systemctl restart kaggle-api`.

2. **Kaggle auth:** `.env` must have `KAGGLE_KEY` and `KAGGLE_USERNAME`.
   The Kaggle Python library requires these exact variable names. Never use
   `kaggle.json` — environment variables only.

3. **`pynvml` deprecation warning.** Cosmetic — package functions are
   identical to `nvidia-ml-py`. Suppress with
   `PYTHONWARNINGS=ignore::FutureWarning` if it's noisy in logs.

4. **Per-competition CLAUDE.md is a stub.** `competition_manager.py new`
   writes a 4-line file. The agent should expand it once a competition
   is registered (target column, evaluation metric details, EDA notes).

5. **HF dataset enrichment requires explicit join keys.** The `enrich`
   stage is a no-op if `hf_dataset_ids` is empty — by design — but it
   also does nothing if your join keys aren't present in both sides. We
   don't auto-discover them.

6. **MLflow served on `127.0.0.1:5000`** (not exposed via the API).
   Add a reverse-proxy endpoint or a separate Cloudflare hostname if you
   want to browse runs from the web UI directly.

---

## Web UI work needed (Session 2)

The `/system/*` endpoints exist server-side; the agency-platform UI still
needs to surface them:

1. **Sidebar state badge** — next to the existing health status dot, render
   a small badge showing the lifecycle state from `GET /system/state`:
   `Running` (green), `Paused` (yellow), `Stopped` (gray). When the dot is
   already red (API offline) the badge is hidden.
2. **Dashboard buttons** — `Pause`, `Resume`, `Stop`. Enabled state mirrors
   the tray app:
   - Pause: enabled only when state is `running`
   - Stop: enabled when `running` or `paused`
   - Resume: enabled when `paused` or `stopped`
3. **Confirmation dialog on Stop** — Stop is destructive (kills the
   Claude Code tmux session and any active training). Prompt before POST.
4. **Optimistic UI** — set the badge to a transitioning style as soon as
   the user clicks; reconcile when the response arrives.
5. **Polling** — bump `/system/state` polling cadence to match the
   existing 10 s `/health` poll, or piggyback on a single combined call.

## Windows tray app

A native Windows tray app lives in `tray/` and exposes the same controls
without opening the browser. Install once on Windows (not WSL2):

```
pip install pystray pillow requests
```

Run by double-clicking `tray/start_tray.bat` (uses `pythonw` so no console
window appears).

Auto-start on login: drop a shortcut to `tray/start_tray.bat` into
`shell:startup`, or create a Task Scheduler entry "At log on" pointing at
`pythonw kaggle_tray.py`. Full instructions in `tray/README.md`.

The icon is a bold "K" on a coloured rounded square:

| Colour | Meaning |
|---|---|
| 🟢 green | running |
| 🟡 yellow | paused |
| ⚪ gray | stopped |
| 🔵 blue | transitioning (request in flight) |
| 🔴 red | API unreachable |

The tray polls `GET /system/state` every 5 s. The dashboard URL it opens
is hard-coded to `https://kaggle.nnaq.net`; change `DASHBOARD_URL` at the
top of `kaggle_tray.py` if the Cloudflare host differs.

## How to start the agent
```bash
cd ~/kaggle-agent
bash scripts/start_agent.sh        # runs Claude Code in this terminal
# or, for the daemonised path:
bash scripts/wsl_startup.sh        # creates tmux session 'kaggle-agent'
tmux attach -t kaggle-agent
```

The agent reads `CLAUDE.md` for instructions, queries the MCP at
`http://docker:8000` for context, and calls
`core.message_bus.get_pending_instructions()` at the top of each loop
iteration so you can redirect it from Telegram or the web chat at any time.

## Quick smoke test from a fresh shell
```bash
curl -sS http://localhost:8765/health   | jq .status
curl -sS http://localhost:8765/memory/status | jq '{pg:.postgres.status,qd:.qdrant.status,mcp:.mcp.status}'
python core/notify.py "handoff smoke test"
```

If those three commands all succeed, the agent is fully wired up.
