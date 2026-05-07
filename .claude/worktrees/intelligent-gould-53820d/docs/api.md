# API

FastAPI bridge listens on `:8765` (override with `API_PORT`). Full docs in `HANDOFF.md`.

## All endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | overall status, services, GPU, active competition |
| GET  | `/system/state` | lifecycle state + run stage + active competition |
| POST | `/system/pause` | freeze agent via cgroupv2 + stop Ollama |
| POST | `/system/resume` | thaw / recreate agent + start Ollama |
| POST | `/system/stop` | SIGTERM + kill tmux + stop Ollama |
| POST | `/system/checkpoint` | signal agent to save mid-run state |
| GET  | `/system/model` | current Claude model + available list |
| POST | `/system/model` | switch model (takes effect on next restart) |
| GET  | `/system/ssh-info` | Tailscale hostname + ssh / tmux commands |
| GET  | `/competitions` | all competitions |
| GET  | `/competitions/active` | current active competition |
| GET  | `/competitions/active/trajectory` | CV/LB history + trend slope |
| POST | `/competitions/new` | `{slug, metric, deadline, …}` |
| POST | `/competitions/switch` | `{slug}` |
| GET  | `/experiments?competition=<slug>&limit=50` | MLflow rows |
| GET  | `/submissions?competition=<slug>` | submission history |
| GET  | `/leaderboard/{slug}` | best-effort via Kaggle CLI |
| GET  | `/logs?lines=200` | tail of every `logs/*.log` |
| GET  | `/gpu/stream` | SSE, GPU snapshot every 3 s |
| GET  | `/chat/messages?limit=50` | recent bus messages, oldest first |
| POST | `/chat/send` | `{text}` — posts ack, wakes agent, forwards to Telegram |
| WS   | `/chat/ws` | live chat; send `{"text":"..."}`, receive message objects |
| WS   | `/terminal/ws` | tmux pane output + accepts `{type:"input",data:"..."}` |
| GET  | `/ollama/status` | `{ok, models}` |
| GET  | `/memory/status` | Postgres + Qdrant + MCP + Tailscale |

CORS is open to `*` unless `ALLOWED_ORIGINS` is set in `.env`.

## `GET /system/state` shape

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

`run_stage` is absent when idle. Stages: `idle | downloading | eda | training | generating_submission | submitting | done`

Agent updates stage by calling `system_control.set_run_stage(stage, detail, eta_seconds)`.

## `GET /system/model` shape

```json
{ "model": "claude-sonnet-4-6", "available": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"] }
```

Model stored in `.claude/kaggle_settings.json`, applied via `--model` flag at agent startup.

## Health response shape
```json
{
  "status": "online",
  "services": {
    "postgres": true, "qdrant": true, "mcp": true,
    "ollama": true, "telegram": true, "mlflow": false
  },
  "active_competition": null,
  "gpu": { "vram_total_mb": 12227, "vram_free_mb": 9123, "gpu_util_pct": 0, ... },
  "uptime_s": 12,
  "last_updated": "2026-05-07T02:00:00Z"
}
```

## WebSocket protocols

**`/chat/ws`**
- Server → client: `{id, role, source, text, status, ask_id, created_at}`
- Client → server: `{"text": "your message"}`
- Server ack: `{"ack": "<msg_id>"}`
- On new human message: backend immediately posts `"Got it, thinking..."` agent message to bus

**`/terminal/ws`**
- Server → client: `{type: "output"|"status"|"error", data: "...", ts: "..."}`
- Client → server: `{type: "input", data: "<keystrokes>"}` — forwarded to tmux via `send-keys`
- `status` with `data: "session offline"` means no active tmux session
