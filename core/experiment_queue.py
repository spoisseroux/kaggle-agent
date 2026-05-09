"""
Experiment Queue - Postgres-backed queue for parallel experiment execution.

Manages experiment scheduling with:
- Priority queue
- VRAM-aware scheduling
- Dependency resolution
- MLflow integration
"""
from __future__ import annotations

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import Json, RealDictCursor

log = logging.getLogger(__name__)


class ExperimentQueue:
    """Postgres-backed experiment queue with smart scheduling."""

    def __init__(self, max_parallel: int = 2, postgres_dsn: Optional[str] = None):
        """
        Initialize experiment queue.

        Args:
            max_parallel: Maximum parallel experiments (based on VRAM)
            postgres_dsn: Postgres connection string (default: from env)
        """
        self.max_parallel = max_parallel
        self.postgres_dsn = postgres_dsn or os.environ.get("POSTGRES_DSN")

        if not self.postgres_dsn:
            raise ValueError("POSTGRES_DSN environment variable not set")

        self._ensure_schema()

    def _get_conn(self):
        """Get database connection."""
        return psycopg2.connect(self.postgres_dsn, cursor_factory=RealDictCursor)

    def _ensure_schema(self):
        """Ensure experiment queue tables exist."""
        schema_path = Path(__file__).parent.parent / "scripts" / "experiment_queue_schema.sql"

        # If schema file doesn't exist, tables should already be created
        if not schema_path.exists():
            log.info("Assuming experiment queue schema already exists")
            return

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute(
                    "SELECT EXISTS (SELECT FROM pg_tables WHERE tablename = 'experiment_queue')"
                )
                exists = cur.fetchone()['exists']

                if not exists:
                    log.info("Creating experiment queue schema")
                    with open(schema_path) as f:
                        cur.execute(f.read())
                    conn.commit()

    def submit(
        self,
        competition_slug: str,
        config: Dict[str, Any],
        priority: int = 5,
        gpu_required: bool = True,
        estimated_duration_min: Optional[int] = None,
        depends_on: Optional[List[int]] = None,
    ) -> int:
        """
        Submit experiment to queue.

        Args:
            competition_slug: Competition identifier
            config: Experiment configuration (model, params, features, etc.)
            priority: 1=highest, 10=lowest
            gpu_required: Whether GPU is needed
            estimated_duration_min: Estimated runtime in minutes
            depends_on: List of experiment IDs this depends on

        Returns:
            Experiment ID
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO experiment_queue
                    (competition_slug, config, priority, gpu_required, estimated_duration_min, status)
                    VALUES (%s, %s, %s, %s, %s, 'queued')
                    RETURNING id
                    """,
                    (competition_slug, Json(config), priority, gpu_required, estimated_duration_min)
                )
                exp_id = cur.fetchone()['id']

                # Add dependencies
                if depends_on:
                    for dep_id in depends_on:
                        cur.execute(
                            """
                            INSERT INTO experiment_dependencies (experiment_id, depends_on_id)
                            VALUES (%s, %s)
                            """,
                            (exp_id, dep_id)
                        )

                conn.commit()

        log.info(f"Submitted experiment {exp_id} for {competition_slug} (priority {priority})")
        return exp_id

    def get_next(self) -> Optional[Dict[str, Any]]:
        """
        Get next experiment to run (respects priority, dependencies, parallel limit).

        Returns:
            Experiment dict or None if queue empty or at capacity
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM get_next_experiment(%s, %s)",
                    (self.max_parallel, True)
                )
                result = cur.fetchone()

                if result:
                    exp_id = result['exp_id']

                    # Mark as running
                    cur.execute(
                        """
                        UPDATE experiment_queue
                        SET status = 'running', started_at = NOW()
                        WHERE id = %s
                        """,
                        (exp_id,)
                    )
                    conn.commit()

                    return {
                        "id": exp_id,
                        "competition_slug": result['exp_slug'],
                        "config": result['exp_config'],
                    }

        return None

    def complete(
        self,
        experiment_id: int,
        mlflow_run_id: str,
        results: Dict[str, Any],
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        """
        Mark experiment as completed.

        Args:
            experiment_id: Experiment ID
            mlflow_run_id: MLflow run ID for tracking
            results: Experiment results (CV scores, metrics)
            success: Whether experiment succeeded
            error_message: Error message if failed
        """
        status = "completed" if success else "failed"

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE experiment_queue
                    SET status = %s,
                        completed_at = NOW(),
                        mlflow_run_id = %s,
                        results = %s,
                        error_message = %s
                    WHERE id = %s
                    """,
                    (status, mlflow_run_id, Json(results), error_message, experiment_id)
                )
                conn.commit()

        log.info(f"Experiment {experiment_id} {status}: {results}")

    def get_status(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get experiment status."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM experiment_queue WHERE id = %s",
                    (experiment_id,)
                )
                return cur.fetchone()

    def list_active(self) -> List[Dict[str, Any]]:
        """List all queued or running experiments."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM active_experiments")
                return cur.fetchall()

    def cancel(self, experiment_id: int):
        """Cancel queued experiment."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE experiment_queue
                    SET status = 'cancelled'
                    WHERE id = %s AND status = 'queued'
                    """,
                    (experiment_id,)
                )
                conn.commit()

        log.info(f"Cancelled experiment {experiment_id}")

    def get_results_summary(self, competition_slug: str) -> List[Dict[str, Any]]:
        """Get all completed experiments for competition, ranked by CV score."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        config,
                        mlflow_run_id,
                        results,
                        started_at,
                        completed_at,
                        EXTRACT(EPOCH FROM (completed_at - started_at))/60 as duration_min
                    FROM experiment_queue
                    WHERE competition_slug = %s
                      AND status = 'completed'
                      AND results IS NOT NULL
                    ORDER BY (results->>'cv_score')::float DESC
                    """,
                    (competition_slug,)
                )
                return cur.fetchall()


if __name__ == "__main__":
    # Test the queue
    queue = ExperimentQueue(max_parallel=2)

    # Submit test experiments
    exp1 = queue.submit(
        "titanic",
        {
            "model": "xgboost",
            "params": {"n_estimators": 100, "learning_rate": 0.1},
            "features": ["age", "fare", "sex"],
        },
        priority=1,
        estimated_duration_min=5,
    )

    exp2 = queue.submit(
        "titanic",
        {
            "model": "lightgbm",
            "params": {"n_estimators": 150, "learning_rate": 0.05},
            "features": ["age", "fare", "sex", "pclass"],
        },
        priority=2,
        estimated_duration_min=7,
    )

    print(f"Submitted experiments: {exp1}, {exp2}")

    # List active
    active = queue.list_active()
    print(f"\nActive experiments: {len(active)}")
    for exp in active:
        print(f"  {exp['id']}: {exp['model']} (priority {exp['priority']})")

    # Get next to run
    next_exp = queue.get_next()
    if next_exp:
        print(f"\nNext to run: {next_exp['id']}")
        print(f"  Config: {next_exp['config']}")

        # Simulate completion
        queue.complete(
            next_exp['id'],
            mlflow_run_id="test_run_123",
            results={"cv_score": 0.85, "train_score": 0.92},
        )

    # Summary
    summary = queue.get_results_summary("titanic")
    print(f"\nCompleted experiments: {len(summary)}")
    for exp in summary:
        print(f"  {exp['id']}: CV={exp['results']['cv_score']}, {exp['duration_min']:.1f}min")
