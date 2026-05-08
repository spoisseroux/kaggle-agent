#!/usr/bin/env python3
"""Backfill kaggle_experiments Postgres table from MLflow runs."""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.notify import _load_env
_load_env()

import mlflow
import psycopg2
from psycopg2.extras import Json

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

def ensure_competition_exists(cursor, slug):
    """Ensure competition exists in kaggle_competitions table."""
    cursor.execute(
        "SELECT slug FROM kaggle_competitions WHERE slug = %s",
        (slug,)
    )
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO kaggle_competitions (slug, name, metric, status)
            VALUES (%s, %s, 'rmsle', 'active')
            ON CONFLICT (slug) DO NOTHING
        """, (slug, slug.replace('-', ' ').title()))

def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    # Get all experiments
    experiments = mlflow.search_experiments()
    
    conn = psycopg2.connect(os.environ['POSTGRES_DSN'])
    cursor = conn.cursor()
    
    total_inserted = 0
    
    for exp in experiments:
        comp_slug = exp.name
        print(f"\nProcessing {comp_slug}...")
        
        # Ensure competition exists
        ensure_competition_exists(cursor, comp_slug)
        conn.commit()
        
        # Get all runs for this experiment
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=1000,
            order_by=["start_time DESC"]
        )
        
        if runs.empty:
            print(f"  No runs found")
            continue
            
        print(f"  Found {len(runs)} runs")
        
        for _, run in runs.iterrows():
            run_id = run.get('run_id')
            
            # Check if already in Postgres
            cursor.execute(
                "SELECT id FROM kaggle_experiments WHERE mlflow_run_id = %s",
                (run_id,)
            )
            if cursor.fetchone():
                continue  # Already exists
            
            # Extract data
            model_type = run.get('tags.model_type')
            feature_set = run.get('tags.feature_set')
            data_version = run.get('tags.data_version')
            cv_mean = run.get('metrics.cv_mean') or run.get('metrics.cv_rmsle')
            cv_std = run.get('metrics.cv_std')
            lb_score = run.get('metrics.lb_score')
            
            # Get params as config
            param_cols = [c for c in runs.columns if c.startswith('params.')]
            config = {c.replace('params.', ''): run.get(c) for c in param_cols if run.get(c) is not None}
            
            # Calculate training time
            start = run.get('start_time')
            end = run.get('end_time')
            training_time_s = None
            if start and end:
                training_time_s = int((end - start).total_seconds())
            
            # Insert
            try:
                cursor.execute("""
                    INSERT INTO kaggle_experiments 
                    (competition_slug, mlflow_run_id, model_type, feature_set, data_version,
                     config, cv_mean, cv_std, lb_score, training_time_s, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    comp_slug, run_id, model_type, feature_set, data_version,
                    Json(config) if config else None,
                    cv_mean, cv_std, lb_score, training_time_s,
                    start
                ))
                total_inserted += 1
            except Exception as e:
                print(f"  Error inserting run {run_id[:8]}: {e}")
                conn.rollback()
                cursor = conn.cursor()  # Get new cursor after rollback
                continue
        
        conn.commit()
    
    print(f"\n✅ Total inserted: {total_inserted}")
    conn.close()

if __name__ == "__main__":
    main()
