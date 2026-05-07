# Memory stack

The shared memory stack runs on the Tailscale VM `docker`:

- **Postgres** at `docker:5432`, database `claude_memory`
- **Qdrant** at `docker:6333`
- **MCP server** at `docker:8000` (FastAPI wrapper around the above)

Claude Code reads/writes via the MCP server using the bearer token in
`.env` → `MCP_BEARER_TOKEN`. Backend code reads/writes Postgres directly
through `core/memory.py`.

## Postgres tables
| Table | Purpose |
|---|---|
| `kaggle_competitions` | active/archived competitions, best CV/LB |
| `kaggle_experiments` | one row per MLflow run, includes config JSONB |
| `kaggle_features` | feature engineering snippets and CV deltas |
| `kaggle_submissions` | submission filename, CV vs LB |
| `kaggle_datasets` | versioned data pipeline outputs |

Re-run the migration with `python scripts/migrate_kaggle_tables.py` — it
uses `CREATE TABLE IF NOT EXISTS` and is safe to re-run.

## Qdrant collections
Created on first call to `core.memory.ensure_collections()`:

| Collection | Vector size | Purpose |
|---|---|---|
| `kaggle_experiments` | 1536 | retrieve similar past approaches |
| `kaggle_features` | 1536 | retrieve relevant feature code |
| `kaggle_errors` | 1536 | retrieve fixes for similar errors |
| `kaggle_insights` | 1536 | general wisdom across competitions |

## Working without memory
If the docker VM is offline, `core.memory.*` returns empty results / logs a
warning rather than raising. The agent should detect the outage from
`/memory/status` (or via `get_pending_instructions()` if the user pings it)
and continue working without memory storage until the VM is back.
