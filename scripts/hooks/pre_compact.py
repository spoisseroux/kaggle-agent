"""PreCompact hook — runs before Claude Code compacts the context window.

Saves a snapshot of current working state to Postgres memory so the agent
can recover full context after compaction without re-reading everything.

Called automatically by Claude Code via settings.json PreCompact hook.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _pg_save(body: str, title: str) -> bool:
    try:
        import psycopg2  # type: ignore
        from core.notify import _load_env
        import os
        _load_env()
        dsn = os.environ.get("POSTGRES_DSN", "")
        if not dsn:
            return False
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO memories (type, title, body, tags, created_at, updated_at)
            VALUES (%s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT DO NOTHING
        """, ("session_checkpoint", title, body, ["checkpoint", "pre_compact"]))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"pre_compact: pg save failed: {e}", file=sys.stderr)
        return False


def main() -> None:
    registry = _read_json(REPO_ROOT / "competitions" / "registry.json")
    active = registry.get("active", "unknown")

    state = _read_json(REPO_ROOT / "state.json")
    run_stage = state.get("run_stage", "idle")

    # Read last few experiments from MLflow if available
    experiments_summary = ""
    try:
        import mlflow  # type: ignore
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(active)
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=5,
            )
            lines = []
            for r in runs:
                cv = r.data.metrics.get("cv_score", r.data.metrics.get("cv", "?"))
                lines.append(f"  {r.info.run_name}: cv={cv}")
            experiments_summary = "Recent experiments:\n" + "\n".join(lines)
    except Exception:
        pass

    ts = datetime.now(timezone.utc).isoformat()
    title = f"session_checkpoint_{active}_{ts}"
    body = f"""Context compaction checkpoint — {ts}

Competition: {active}
Run stage: {run_stage}
Agent state: {state.get('state', 'unknown')}

{experiments_summary}

Resume instructions:
- Competition is {active}, read competitions/active/{active}/CLAUDE.md
- Last run stage was: {run_stage}
- Check get_pending_instructions() for any queued human messages
- Continue from where you left off — do not restart the full workflow
""".strip()

    saved = _pg_save(body, title)
    print(f"pre_compact: checkpoint saved={saved} competition={active} stage={run_stage}")


if __name__ == "__main__":
    main()
