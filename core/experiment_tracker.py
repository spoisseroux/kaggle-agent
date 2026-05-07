"""Thin wrapper around MLflow for the kaggle agent.

Mirrors run-finish events into the kaggle_experiments table on docker so the
web UI can read either source consistently.
"""
from __future__ import annotations

import contextlib
import os
import time
from typing import Any

import mlflow

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _set_uri() -> None:
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)


def get_or_create_experiment(name: str) -> str:
    _set_uri()
    exp = mlflow.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    return mlflow.create_experiment(name)


@contextlib.contextmanager
def run(competition_slug: str, *, model_type: str = "unknown",
        feature_set: str | None = None, data_version: str | None = None,
        config: dict | None = None, notes: str | None = None,
        tags: dict | None = None):
    """Open an MLflow run; on close, log a row into kaggle_experiments."""
    _set_uri()
    exp_id = get_or_create_experiment(competition_slug)
    started = time.time()
    with mlflow.start_run(experiment_id=exp_id) as r:
        run_tags = {"model_type": model_type}
        if feature_set:
            run_tags["feature_set"] = feature_set
        if data_version:
            run_tags["data_version"] = data_version
        if tags:
            run_tags.update(tags)
        mlflow.set_tags(run_tags)
        if config:
            mlflow.log_params({k: str(v) for k, v in config.items()})
        captured: dict[str, Any] = {"cv_mean": None, "cv_std": None, "lb_score": None}
        yield (r, captured)
        elapsed = int(time.time() - started)
        for k in ("cv_mean", "cv_std", "lb_score"):
            v = captured.get(k)
            if v is not None:
                mlflow.log_metric(k, float(v))
        try:
            from core.memory import insert_experiment
            insert_experiment(
                competition_slug,
                mlflow_run_id=r.info.run_id,
                model_type=model_type,
                feature_set=feature_set,
                data_version=data_version,
                config=config,
                cv_mean=captured.get("cv_mean"),
                cv_std=captured.get("cv_std"),
                lb_score=captured.get("lb_score"),
                training_time_s=elapsed,
                notes=notes,
            )
        except Exception:
            pass


def list_runs(competition_slug: str, limit: int = 50) -> list[dict]:
    _set_uri()
    exp = mlflow.get_experiment_by_name(competition_slug)
    if not exp:
        return []
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=limit,
                            order_by=["attributes.start_time DESC"])
    if df.empty:
        return []
    keep = [c for c in ("run_id", "status", "start_time", "end_time",
                        "metrics.cv_mean", "metrics.cv_std", "metrics.lb_score",
                        "tags.model_type", "tags.feature_set", "tags.data_version")
            if c in df.columns]
    return df[keep].to_dict(orient="records")
