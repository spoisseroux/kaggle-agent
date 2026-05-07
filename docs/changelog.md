# Changelog

## 2026-05-06 — initial build (Phases 1–5)

### Phase 1 — connectivity + message bus
- `.env` populated with KAGGLE_TOKEN (never KAGGLE_KEY), POSTGRES_DSN,
  MCP_BEARER_TOKEN, HF_TOKEN
- `.claude/mcp_config.json` points at the Tailscale memory MCP at
  `http://docker:8000` with bearer auth
- DB migration applied to `claude_memory` on `docker:5432`:
  `kaggle_competitions`, `kaggle_experiments`, `kaggle_features`,
  `kaggle_submissions`, `kaggle_datasets`
- `core/message_bus.py` — SQLite chat bus (`create_ask` /
  `wait_for_reply` / `get_pending_instructions`)
- `core/notify.py` (fire-and-forget) and `core/ask_human.py` (blocking)
- `core/telegram_bot.py` runs as `systemd/telegram-bot.service`
- Verified Telegram roundtrip: outbound delivered, inbound `HELLLOOOO`
  ingested into bus.

### Phase 2 — VRAM manager + Ollama
- Ollama 0.23.1 installed under `~/.local` (no global sudo install)
- `systemd/ollama.service` enabled and active
- `qwen3:14b` pulled (~8.8 GB Q4_K_M)
- `core/vram_manager.py` — `request_training_vram` /
  `release_training_vram`, threshold lowered to 9000 MB to match WSL2
  headroom (PRD §7 assumed 9500 MB but Windows reserves ~3 GB on this
  box, not 1.5 GB)
- `core/ollama_client.py` — generate / stream / health, supports
  Qwen3 think mode via `/think` prefix
- Verified full cycle: VRAM 383 MB → 9204 MB after stop, Ollama
  restarted automatically on release.

### Phase 3 — data pipeline + HuggingFace
- `core/hf_search.py` — wraps `HfApi.list_datasets` (uses 1.x `filter`
  arg, not the deprecated `tags`)
- `competitions/template/src/data_pipeline.py` — `refine` / `enrich` /
  `synthetic` / `fuzz` with versioned parquet output and
  `kaggle_datasets` row per stage; auto-loads `.env`
- `scripts/pre_train.sh` and `scripts/post_train.sh` wrap
  `request_training_vram` / `release_training_vram` for shell use
- Verified: refine clips 1e6 outlier to 5σ, coerces numeric-looking
  strings, fuzz preserves target column, SMOTE rebalances class 1
  118→482, postgres rows recorded.

### Phase 4 — competition manager + MLflow + API
- `core/competition_manager.py` CLI with new/switch/archive/list/status
  + Postgres mirror via `upsert_competition`
- `core/experiment_tracker.py` — MLflow `run()` context manager that
  also writes a row to `kaggle_experiments` (FK to `kaggle_competitions`)
- `core/memory.py` — Postgres + Qdrant + MCP read/write helpers, lazy
  Qdrant collection creation, never raises on outage
- `core/gpu_monitor.py` — pynvml snapshot wrapper
- `api/main.py` — every endpoint from PRD §10/§17 plus `/memory/status`,
  WebSocket chat, SSE GPU stream, CORS
- `systemd/mlflow.service` and `systemd/kaggle-api.service` installed
  and active
- Verified: `/health` reports all 6 services online; `chat/send` +
  `chat/messages` roundtrip; experiment logged to MLflow and mirrored
  into Postgres.

### Phase 5 — hooks + startup + docs + handoff
- `.claude/settings.json` Stop + PostToolUse hooks
- `scripts/hooks/post_bash.py` fires notify on CV / OOM / submission /
  fold-1 / epoch-1 events
- `scripts/start_agent.sh` boots Claude Code with MCP and compaction
- `scripts/wsl_startup.sh` brings up Tailscale, all services, and the
  `kaggle-agent` tmux session
- All `docs/*.md` populated
- `HANDOFF.md` generated with confirmed endpoints and SSH commands
