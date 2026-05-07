"""Four-stage data pipeline. Run before any model training.

    Stage 1: Refine     — clean, fix errors, handle outliers
    Stage 2: Enrich     — merge HuggingFace external datasets
    Stage 3: Synthetic  — generate additional samples (SMOTE / SDV)
    Stage 4: Fuzz       — add controlled noise to prevent overfitting
                          numeric: Gaussian noise sigma=0.01*std
                          categorical: swap 2% of values randomly
                          NEVER fuzz the target column
                          NEVER fuzz the test set

Each stage produces a versioned parquet under data/<slug>/processed/<vN_stage>/
and an entry in the kaggle_datasets table.

This module is intentionally generic; the per-competition CLAUDE.md should
specialise the cleaning rules and choose which stages to run.

Usage:
    python -m competitions.active.<slug>.src.data_pipeline \\
        --slug <slug> --stages refine,enrich,synthetic,fuzz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Walk upward to find the repo root (contains .env). Works whether this file
# lives in competitions/template/src/ or competitions/active/<slug>/src/.
_p = Path(__file__).resolve()
for _ in range(8):
    if (_p / ".env").exists() or (_p / "kaggle-agent-PRD-final.md").exists():
        REPO_ROOT = _p
        break
    _p = _p.parent
else:
    REPO_ROOT = Path.cwd()


def _ensure_env() -> None:
    if os.environ.get("POSTGRES_DSN"):
        return
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


_ensure_env()


# ---------- Stage 1: refine ----------

def refine(df: pd.DataFrame, *, target: str | None = None) -> pd.DataFrame:
    """Cleaning that is safe for any tabular dataset.

    - Drop fully-empty rows / cols
    - Strip whitespace from string columns
    - Coerce numeric-looking object columns
    - Clip extreme outliers in numeric cols (>5 sigma) by winsorising
    """
    out = df.copy()
    out = out.dropna(axis=0, how="all").dropna(axis=1, how="all")
    for c in out.select_dtypes(include="object").columns:
        out[c] = out[c].astype(str).str.strip()
        coerced = pd.to_numeric(out[c], errors="coerce")
        if coerced.notna().mean() > 0.95:
            out[c] = coerced
    for c in out.select_dtypes(include=[np.number]).columns:
        if c == target:
            continue
        s = out[c]
        if s.notna().sum() < 30:
            continue
        mu, sd = s.mean(), s.std()
        if sd and not np.isnan(sd):
            lo, hi = mu - 5 * sd, mu + 5 * sd
            out[c] = s.clip(lo, hi)
    return out


# ---------- Stage 2: enrich ----------

def enrich(df: pd.DataFrame, *, hf_dataset_ids: Iterable[str] = (), join_keys: list[str] | None = None) -> pd.DataFrame:
    """Merge external HuggingFace datasets.

    Caller specifies which datasets and the join keys; this function loads
    them and left-joins. If hf_dataset_ids is empty, the dataframe is returned
    unchanged — this is the common case until enrichment is configured.
    """
    if not hf_dataset_ids:
        return df
    from datasets import load_dataset
    out = df.copy()
    for ds_id in hf_dataset_ids:
        ds = load_dataset(ds_id, split="train")
        ext = ds.to_pandas()
        if join_keys:
            keys = [k for k in join_keys if k in out.columns and k in ext.columns]
            if keys:
                out = out.merge(ext, on=keys, how="left", suffixes=("", f"_{ds_id.split('/')[-1]}"))
    return out


# ---------- Stage 3: synthetic ----------

def synthetic(df: pd.DataFrame, *, target: str, method: str = "smote", n_samples: int | None = None) -> pd.DataFrame:
    """Generate extra training rows.

    - method='smote' for class imbalance (classification only)
    - method='sdv'   for general tabular synthetic data
    """
    if method == "smote":
        from imblearn.over_sampling import SMOTE
        y = df[target]
        X = df.drop(columns=[target])
        non_num = X.select_dtypes(exclude=[np.number]).columns
        if len(non_num):
            return df  # SMOTE needs purely-numeric features; caller should encode first
        sm = SMOTE(random_state=0)
        X_res, y_res = sm.fit_resample(X, y)
        out = pd.concat([X_res.reset_index(drop=True),
                         pd.Series(y_res, name=target).reset_index(drop=True)], axis=1)
        return out
    if method == "sdv":
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer
        meta = SingleTableMetadata()
        meta.detect_from_dataframe(df)
        synth = GaussianCopulaSynthesizer(meta)
        synth.fit(df)
        n = n_samples or len(df)
        return pd.concat([df, synth.sample(num_rows=n)], ignore_index=True)
    raise ValueError(f"unknown synthetic method: {method}")


# ---------- Stage 4: fuzz ----------

def fuzz(
    df: pd.DataFrame,
    *,
    target: str,
    numeric_sigma_frac: float = 0.01,
    categorical_swap_frac: float = 0.02,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Add controlled noise to prevent overfitting.

    - numeric: Gaussian noise N(0, sigma * std) added per column
    - categorical: swap categorical_swap_frac of values with another random value
    - Never touch the target column
    """
    rng = np.random.default_rng(rng_seed)
    out = df.copy()
    for c in out.columns:
        if c == target:
            continue
        col = out[c]
        if pd.api.types.is_numeric_dtype(col):
            sd = col.std()
            if sd and not np.isnan(sd):
                noise = rng.normal(0.0, numeric_sigma_frac * sd, size=len(col))
                out[c] = col + noise
        else:
            n = len(col)
            k = max(1, int(categorical_swap_frac * n))
            idx = rng.choice(n, size=k, replace=False)
            values = col.dropna().unique()
            if len(values) > 1:
                replacements = rng.choice(values, size=k)
                col = col.copy()
                col.iloc[idx] = replacements
                out[c] = col
    return out


# ---------- registry I/O ----------

@dataclass
class StageResult:
    stage: str
    version: str
    rows: int
    cols: int
    out_path: Path
    extra: dict = field(default_factory=dict)


def _write_versioned(df: pd.DataFrame, slug: str, stage: str, version: str) -> Path:
    out_dir = Path(f"data/{slug}/processed/{version}_{stage}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stage}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def log_dataset_to_postgres(slug: str, version: str, rows: int, cols: int,
                            description: str, hf_used: list[str] | None = None) -> None:
    try:
        import psycopg2
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            return
        conn = psycopg2.connect(dsn, connect_timeout=5)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO kaggle_datasets "
                "(competition_slug, version, row_count, col_count, description, hf_datasets_used) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (slug, version, rows, cols, description, hf_used or []),
            )
        conn.close()
    except Exception as e:
        print(f"[warn] could not log dataset to postgres: {e}", file=sys.stderr)


# ---------- driver ----------

def run_pipeline(
    slug: str,
    *,
    raw_train: str,
    target: str,
    stages: list[str],
    hf_dataset_ids: list[str] | None = None,
    join_keys: list[str] | None = None,
    synthetic_method: str = "smote",
    version: str = "v1",
) -> list[StageResult]:
    df = pd.read_csv(raw_train) if raw_train.endswith(".csv") else pd.read_parquet(raw_train)
    results: list[StageResult] = []
    if "refine" in stages:
        df = refine(df, target=target)
        path = _write_versioned(df, slug, "refine", version)
        log_dataset_to_postgres(slug, f"{version}_refine", len(df), df.shape[1], "stage 1 refine")
        results.append(StageResult("refine", version, len(df), df.shape[1], path))
    if "enrich" in stages:
        df = enrich(df, hf_dataset_ids=hf_dataset_ids or [], join_keys=join_keys)
        path = _write_versioned(df, slug, "enrich", version)
        log_dataset_to_postgres(slug, f"{version}_enrich", len(df), df.shape[1],
                                "stage 2 enrich", hf_dataset_ids or [])
        results.append(StageResult("enrich", version, len(df), df.shape[1], path,
                                   extra={"hf": hf_dataset_ids or []}))
    if "synthetic" in stages:
        df = synthetic(df, target=target, method=synthetic_method)
        path = _write_versioned(df, slug, "synthetic", version)
        log_dataset_to_postgres(slug, f"{version}_synthetic", len(df), df.shape[1],
                                f"stage 3 synthetic ({synthetic_method})")
        results.append(StageResult("synthetic", version, len(df), df.shape[1], path,
                                   extra={"method": synthetic_method}))
    if "fuzz" in stages:
        df = fuzz(df, target=target)
        path = _write_versioned(df, slug, "fuzz", version)
        log_dataset_to_postgres(slug, f"{version}_fuzz", len(df), df.shape[1], "stage 4 fuzz")
        results.append(StageResult("fuzz", version, len(df), df.shape[1], path))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--raw-train", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--stages", default="refine,enrich,synthetic,fuzz")
    ap.add_argument("--hf-dataset-ids", default="")
    ap.add_argument("--join-keys", default="")
    ap.add_argument("--synthetic-method", default="smote", choices=["smote", "sdv"])
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    hf = [s.strip() for s in args.hf_dataset_ids.split(",") if s.strip()]
    keys = [s.strip() for s in args.join_keys.split(",") if s.strip()]
    results = run_pipeline(
        args.slug,
        raw_train=args.raw_train,
        target=args.target,
        stages=stages,
        hf_dataset_ids=hf or None,
        join_keys=keys or None,
        synthetic_method=args.synthetic_method,
        version=args.version,
    )
    print(json.dumps([{
        "stage": r.stage, "version": r.version,
        "rows": r.rows, "cols": r.cols, "out": str(r.out_path),
        **r.extra,
    } for r in results], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
