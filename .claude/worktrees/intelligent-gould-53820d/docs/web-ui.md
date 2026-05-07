# Web UI

Owned by the **agency-platform** repo (Next.js). This file documents the
contract from the kaggle-agent side; full UI spec lives in PRD §11 / §17.

## Sidebar
```
▼ Kaggle (status dot: green / yellow / red)
    Dashboard
    Chat
    Experiments
    GPU
    Competitions
    Logs
    Submissions
    Leaderboard
    Memory
```

The status dot polls `GET /health` every 10 s with a 3 s timeout:
- **green** — `services.postgres && services.ollama && services.telegram`
- **yellow** — any service down except all three (e.g. Ollama stopped
  during training is *expected*)
- **red** — `/health` did not respond within timeout

When `/health` is red, all sub-tools are greyed and a banner reads
`● Kaggle agent offline · last seen X minutes ago`.

## Pages → endpoints
| Page | Endpoint(s) |
|---|---|
| Dashboard | `/health`, `/competitions/active/trajectory`, `/system/ssh-info` |
| Chat | `WS /chat/ws`, `GET /chat/messages`, `POST /chat/send` |
| Experiments | `GET /experiments?competition=<slug>` |
| GPU | `SSE /gpu/stream` |
| Competitions | `GET /competitions`, `POST /competitions/new`, `POST /competitions/switch` |
| Logs | `GET /logs?lines=200` |
| Submissions | `GET /submissions?competition=<slug>` |
| Leaderboard | `GET /leaderboard/{slug}` |
| Memory | `GET /memory/status` |

## Frontend env var
```
NEXT_PUBLIC_KAGGLE_API_URL=https://kaggle.<your-domain>
```
