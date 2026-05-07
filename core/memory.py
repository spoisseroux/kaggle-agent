"""Read/write helpers for the shared memory stack on the docker Tailscale VM.

Postgres tables (created by scripts/migrate_kaggle_tables.py):
  kaggle_competitions, kaggle_experiments, kaggle_features,
  kaggle_submissions, kaggle_datasets

Qdrant collections (created lazily on startup):
  kaggle_experiments, kaggle_features, kaggle_errors, kaggle_insights

All operations are best-effort — never raise on a memory outage; the agent
should keep working without memory if the docker VM is offline.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)


def _ensure_env() -> None:
    """Load .env once if POSTGRES_DSN/MCP_URL aren't already set."""
    if os.environ.get("POSTGRES_DSN") and os.environ.get("MCP_URL"):
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


_ensure_env()

QDRANT_COLLECTIONS = {
    "kaggle_experiments": 1536,
    "kaggle_features": 1536,
    "kaggle_errors": 1536,
    "kaggle_insights": 1536,
}


# ---------- Postgres ----------

@contextmanager
def pg_conn():
    import psycopg2
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("POSTGRES_DSN not set")
    conn = psycopg2.connect(dsn, connect_timeout=5)
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()


def upsert_competition(slug: str, **fields: Any) -> None:
    cols = ["slug"] + list(fields.keys())
    vals = [slug] + list(fields.values())
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "slug")
    sql = (
        f"INSERT INTO kaggle_competitions ({', '.join(cols)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (slug) DO UPDATE SET {updates}, updated_at=NOW()"
        if updates else
        f"INSERT INTO kaggle_competitions ({', '.join(cols)}) "
        f"VALUES ({placeholders}) ON CONFLICT (slug) DO NOTHING"
    )
    try:
        with pg_conn() as c, c.cursor() as cur:
            cur.execute(sql, vals)
    except Exception as e:
        log.warning("upsert_competition failed: %s", e)


def list_competitions() -> list[dict]:
    try:
        with pg_conn() as c, c.cursor() as cur:
            cur.execute(
                "SELECT slug, name, metric, higher_better, deadline, status, "
                "best_cv, best_lb, lb_rank, total_teams, submissions_used, submissions_max, "
                "created_at, updated_at "
                "FROM kaggle_competitions ORDER BY updated_at DESC"
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as e:
        log.warning("list_competitions failed: %s", e)
        return []


def get_competition(slug: str) -> dict | None:
    try:
        with pg_conn() as c, c.cursor() as cur:
            cur.execute(
                "SELECT slug, name, metric, higher_better, deadline, status, "
                "best_cv, best_lb, lb_rank, total_teams, submissions_used, submissions_max, "
                "created_at, updated_at "
                "FROM kaggle_competitions WHERE slug=%s", (slug,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
    except Exception as e:
        log.warning("get_competition failed: %s", e)
        return None


def insert_experiment(slug: str, *, mlflow_run_id: str | None = None,
                      model_type: str | None = None, feature_set: str | None = None,
                      data_version: str | None = None, config: dict | None = None,
                      cv_mean: float | None = None, cv_std: float | None = None,
                      lb_score: float | None = None, training_time_s: int | None = None,
                      notes: str | None = None) -> int | None:
    import json
    try:
        with pg_conn() as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO kaggle_experiments "
                "(competition_slug, mlflow_run_id, model_type, feature_set, data_version, "
                "config, cv_mean, cv_std, lb_score, training_time_s, notes) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (slug, mlflow_run_id, model_type, feature_set, data_version,
                 json.dumps(config or {}), cv_mean, cv_std, lb_score, training_time_s, notes),
            )
            return cur.fetchone()[0]
    except Exception as e:
        log.warning("insert_experiment failed: %s", e)
        return None


def list_experiments(slug: str | None = None, limit: int = 50) -> list[dict]:
    try:
        with pg_conn() as c, c.cursor() as cur:
            if slug:
                cur.execute(
                    "SELECT id, competition_slug, mlflow_run_id, model_type, feature_set, "
                    "data_version, config, cv_mean, cv_std, lb_score, training_time_s, notes, "
                    "created_at "
                    "FROM kaggle_experiments WHERE competition_slug=%s "
                    "ORDER BY created_at DESC LIMIT %s",
                    (slug, limit),
                )
            else:
                cur.execute(
                    "SELECT id, competition_slug, mlflow_run_id, model_type, feature_set, "
                    "data_version, config, cv_mean, cv_std, lb_score, training_time_s, notes, "
                    "created_at "
                    "FROM kaggle_experiments ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as e:
        log.warning("list_experiments failed: %s", e)
        return []


def insert_submission(slug: str, *, filename: str, cv_score: float | None = None,
                      lb_score: float | None = None, lb_rank: int | None = None,
                      total_teams: int | None = None, notes: str | None = None) -> int | None:
    pct = None
    if lb_rank is not None and total_teams:
        pct = 100.0 * (1.0 - (lb_rank - 1) / max(total_teams, 1))
    try:
        with pg_conn() as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO kaggle_submissions "
                "(competition_slug, filename, cv_score, lb_score, lb_rank, total_teams, "
                "percentile, notes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (slug, filename, cv_score, lb_score, lb_rank, total_teams, pct, notes),
            )
            return cur.fetchone()[0]
    except Exception as e:
        log.warning("insert_submission failed: %s", e)
        return None


def list_submissions(slug: str | None = None, limit: int = 50) -> list[dict]:
    try:
        with pg_conn() as c, c.cursor() as cur:
            if slug:
                cur.execute(
                    "SELECT id, competition_slug, filename, cv_score, lb_score, lb_rank, "
                    "total_teams, percentile, notes, submitted_at "
                    "FROM kaggle_submissions WHERE competition_slug=%s "
                    "ORDER BY submitted_at DESC LIMIT %s", (slug, limit))
            else:
                cur.execute(
                    "SELECT id, competition_slug, filename, cv_score, lb_score, lb_rank, "
                    "total_teams, percentile, notes, submitted_at "
                    "FROM kaggle_submissions ORDER BY submitted_at DESC LIMIT %s", (limit,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as e:
        log.warning("list_submissions failed: %s", e)
        return []


def postgres_status() -> dict:
    try:
        t0 = time.time()
        with pg_conn() as c, c.cursor() as cur:
            cur.execute("SELECT current_database()")
            db = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name LIKE 'kaggle_%'"
            )
            tables_count = cur.fetchone()[0]
        latency_ms = int((time.time() - t0) * 1000)
        return {"status": "online", "host": os.environ.get("TAILSCALE_MEMORY_HOST", "docker"),
                "db": db, "tables_count": tables_count, "latency_ms": latency_ms}
    except Exception as e:
        return {"status": "offline", "error": str(e)[:200]}


# ---------- Qdrant ----------

def qdrant_client():
    from qdrant_client import QdrantClient
    url = os.environ.get("QDRANT_URL", "http://docker:6333")
    return QdrantClient(url=url, timeout=5)


def ensure_collections(sizes: dict[str, int] = QDRANT_COLLECTIONS) -> list[str]:
    """Create kaggle_* collections in Qdrant if missing. Returns names ensured."""
    try:
        from qdrant_client.http.models import VectorParams, Distance
        cli = qdrant_client()
        existing = {c.name for c in cli.get_collections().collections}
        ensured = []
        for name, size in sizes.items():
            if name not in existing:
                cli.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=size, distance=Distance.COSINE),
                )
            ensured.append(name)
        return ensured
    except Exception as e:
        log.warning("ensure_collections failed: %s", e)
        return []


def qdrant_status() -> dict:
    try:
        t0 = time.time()
        cli = qdrant_client()
        cols = cli.get_collections().collections
        out = []
        for c in cols:
            try:
                info = cli.get_collection(collection_name=c.name)
                count = info.points_count if info.points_count is not None else 0
                out.append({"name": c.name, "vectors_count": int(count)})
            except Exception:
                out.append({"name": c.name, "vectors_count": None})
        return {"status": "online", "host": os.environ.get("TAILSCALE_MEMORY_HOST", "docker"),
                "collections": out, "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"status": "offline", "error": str(e)[:200]}


def store_vector(collection: str, vector: Iterable[float], payload: dict | None = None) -> str | None:
    try:
        from qdrant_client.http.models import PointStruct
        cli = qdrant_client()
        point_id = str(uuid.uuid4())
        cli.upsert(
            collection_name=collection,
            points=[PointStruct(id=point_id, vector=list(vector), payload=payload or {})],
        )
        return point_id
    except Exception as e:
        log.warning("store_vector failed: %s", e)
        return None


# ---------- MCP ----------

def mcp_status() -> dict:
    import httpx
    try:
        url = os.environ.get("MCP_URL", "http://docker:8000")
        t0 = time.time()
        r = httpx.get(f"{url}/health", timeout=3)
        ok = r.status_code == 200
        return {"status": "online" if ok else "degraded",
                "host": os.environ.get("TAILSCALE_MEMORY_HOST", "docker"),
                "latency_ms": int((time.time() - t0) * 1000),
                "version": "unknown"}
    except Exception as e:
        return {"status": "offline", "error": str(e)[:200]}


if __name__ == "__main__":
    import json
    print(json.dumps({
        "postgres": postgres_status(),
        "qdrant": qdrant_status(),
        "mcp": mcp_status(),
    }, indent=2, default=str))
