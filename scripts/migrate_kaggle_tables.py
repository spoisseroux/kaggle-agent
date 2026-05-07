"""Create kaggle_* tables in the shared `claude_memory` Postgres on docker.

Idempotent: uses CREATE TABLE IF NOT EXISTS. Safe to re-run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.notify import _load_env  # noqa: E402

_load_env()

import psycopg2  # noqa: E402

SCHEMA = """
CREATE TABLE IF NOT EXISTS kaggle_competitions (
  id            SERIAL PRIMARY KEY,
  slug          TEXT UNIQUE NOT NULL,
  name          TEXT,
  metric        TEXT,
  higher_better BOOLEAN DEFAULT TRUE,
  deadline      TIMESTAMP,
  status        TEXT DEFAULT 'active',
  best_cv       FLOAT,
  best_lb       FLOAT,
  lb_rank       INTEGER,
  total_teams   INTEGER,
  submissions_used INTEGER DEFAULT 0,
  submissions_max  INTEGER,
  created_at    TIMESTAMP DEFAULT NOW(),
  updated_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kaggle_experiments (
  id              SERIAL PRIMARY KEY,
  competition_slug TEXT REFERENCES kaggle_competitions(slug),
  mlflow_run_id   TEXT,
  model_type      TEXT,
  feature_set     TEXT,
  data_version    TEXT,
  config          JSONB,
  cv_mean         FLOAT,
  cv_std          FLOAT,
  lb_score        FLOAT,
  training_time_s INTEGER,
  notes           TEXT,
  created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kaggle_features (
  id              SERIAL PRIMARY KEY,
  name            TEXT,
  competition_slug TEXT,
  code_snippet    TEXT,
  cv_delta        FLOAT,
  description     TEXT,
  tags            TEXT[],
  created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kaggle_submissions (
  id              SERIAL PRIMARY KEY,
  competition_slug TEXT REFERENCES kaggle_competitions(slug),
  filename        TEXT,
  cv_score        FLOAT,
  lb_score        FLOAT,
  lb_rank         INTEGER,
  total_teams     INTEGER,
  percentile      FLOAT,
  notes           TEXT,
  submitted_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kaggle_datasets (
  id              SERIAL PRIMARY KEY,
  competition_slug TEXT,
  version         TEXT,
  row_count       INTEGER,
  col_count       INTEGER,
  description     TEXT,
  hf_datasets_used TEXT[],
  created_at      TIMESTAMP DEFAULT NOW()
);
"""


def main() -> int:
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        print("POSTGRES_DSN not set", file=sys.stderr)
        return 2
    conn = psycopg2.connect(dsn, connect_timeout=10)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name LIKE 'kaggle_%' "
            "ORDER BY table_name"
        )
        tables = [r[0] for r in cur.fetchall()]
    print("kaggle tables now present:")
    for t in tables:
        print(f"  - {t}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
