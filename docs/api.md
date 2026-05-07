# API

FastAPI bridge listens on `:8765` (override with `API_PORT`). The web UI in
agency-platform talks to it via the Cloudflare Tunnel URL.

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | overall status, services, gpu, active competition |
| GET  | `/system/ssh-info` | tailscale hostname + ssh / tmux commands |
| GET  | `/competitions` | all competitions known to memory |
| GET  | `/competitions/active` | current active competition |
| GET  | `/competitions/active/trajectory` | CV/LB history + trend slope |
| POST | `/competitions/new` | `{slug, metric, deadline, …}` |
| POST | `/competitions/switch` | `{slug}` |
| GET  | `/experiments?competition=<slug>&limit=50` | MLflow rows |
| GET  | `/submissions?competition=<slug>` | submission history |
| GET  | `/leaderboard/{slug}` | best-effort via Kaggle CLI |
| GET  | `/logs?lines=200` | tail of every `logs/*.log` |
| GET  | `/gpu/stream` | SSE, 3-second snapshots |
| GET  | `/chat/messages?limit=50` | recent bus messages |
| POST | `/chat/send` | `{text}` |
| WS   | `/chat/ws` | live chat |
| GET  | `/ollama/status` | `{ok, models}` |
| GET  | `/memory/status` | postgres + qdrant + mcp + tailscale |

CORS is open to `*` unless `ALLOWED_ORIGINS` is set in `.env`.

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
