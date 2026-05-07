# Kaggle Agent — operator docs

This is a local Kaggle agent that runs on a single RTX 5070 box and shares
state with a homelab memory stack (Postgres + Qdrant + MCP) over Tailscale.
The agent loop is driven by Claude Code in a persistent tmux session; the
agency-platform web UI talks to a small FastAPI bridge (port 8765).

| File | Topic |
|---|---|
| `setup.md` | one-time install, Windows Task Scheduler, sudoers |
| `competitions.md` | new/switch/archive workflow |
| `data-pipeline.md` | refine → enrich → synthetic → fuzz |
| `memory.md` | Postgres tables, Qdrant collections, MCP |
| `vram-manager.md` | how training cohabits with Ollama |
| `telegram.md` | bot setup, ask-human, unsolicited messages |
| `api.md` | endpoint reference |
| `web-ui.md` | sidebar, dashboard, chat, GPU, memory pages |
| `hooks.md` | Claude Code hooks |
| `troubleshooting.md` | known issues and fixes |
| `changelog.md` | dated build log |

The single source of truth for the system spec is
[`../kaggle-agent-PRD-final.md`](../kaggle-agent-PRD-final.md).
