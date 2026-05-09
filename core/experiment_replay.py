"""Replay past experiments with modifications.

Allows quick A/B testing of hyperparameters, features, and validation strategies
by re-running past experiments with specified changes.
"""
from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

log = logging.getLogger(__name__)


class ExperimentReplayer:
    """Replay experiments with modifications."""

    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        """
        Initialize replayer.

        Args:
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.mlflow_uri = mlflow_tracking_uri

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment details from MLflow.

        Args:
            experiment_id: MLflow run ID or experiment name

        Returns:
            Experiment metadata and config
        """
        try:
            import mlflow
            mlflow.set_tracking_uri(self.mlflow_uri)

            # Try as run ID first
            try:
                run = mlflow.get_run(experiment_id)
            except:
                # Try searching by name
                runs = mlflow.search_runs(
                    filter_string=f"tags.mlflow.runName = '{experiment_id}'",
                    max_results=1
                )
                if runs.empty:
                    log.error(f"Experiment not found: {experiment_id}")
                    return None
                run = mlflow.get_run(runs.iloc[0].run_id)

            # Extract config
            params = dict(run.data.params)
            metrics = dict(run.data.metrics)
            tags = dict(run.data.tags)

            return {
                "run_id": run.info.run_id,
                "run_name": tags.get("mlflow.runName", "unknown"),
                "params": params,
                "metrics": metrics,
                "tags": tags,
                "artifact_uri": run.info.artifact_uri,
            }

        except Exception as e:
            log.error(f"Failed to get experiment: {e}")
            return None

    def replay(
        self,
        experiment_id: str,
        modifications: Dict[str, Any],
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Replay experiment with modifications.

        Args:
            experiment_id: Original experiment ID or name
            modifications: Dict of parameters to modify
            dry_run: If True, show what would change without running

        Returns:
            Results with original vs new metrics
        """
        # Get original experiment
        original = self.get_experiment(experiment_id)
        if not original:
            return {
                "success": False,
                "error": f"Experiment {experiment_id} not found"
            }

        # Merge modifications
        new_config = self._merge_config(original["params"], modifications)

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "original_config": original["params"],
                "new_config": new_config,
                "changes": self._get_changes(original["params"], new_config),
            }

        # Execute replay
        log.info(f"Replaying {original['run_name']} with modifications: {modifications}")

        # NOTE: Actual training execution would happen here
        # For now, return a placeholder showing what would be done
        return {
            "success": True,
            "dry_run": False,
            "original_run_id": original["run_id"],
            "original_run_name": original["run_name"],
            "original_cv": original["metrics"].get("cv_score"),
            "new_run_id": "would_create_new_run",
            "new_cv": None,  # Would be populated after training
            "changes": self._get_changes(original["params"], new_config),
            "config": new_config,
            "message": "Replay configured - actual training execution requires competition-specific training script"
        }

    def _merge_config(
        self,
        original: Dict[str, Any],
        modifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge modifications into original config."""
        config = original.copy()

        for key, value in modifications.items():
            # Handle special cases
            if key == "add_features":
                existing = config.get("features", "").split(",")
                new_features = value if isinstance(value, list) else value.split(",")
                config["features"] = ",".join(existing + new_features)
            elif key == "remove_features":
                existing = config.get("features", "").split(",")
                to_remove = value if isinstance(value, list) else value.split(",")
                config["features"] = ",".join([f for f in existing if f not in to_remove])
            else:
                # Direct replacement
                config[key] = value

        return config

    def _get_changes(
        self,
        original: Dict[str, Any],
        modified: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get list of changes between configs."""
        changes = []

        # Check modified values
        for key, new_val in modified.items():
            old_val = original.get(key)
            if old_val != new_val:
                changes.append({
                    "parameter": key,
                    "old_value": old_val,
                    "new_value": new_val,
                })

        # Check removed values
        for key in original:
            if key not in modified:
                changes.append({
                    "parameter": key,
                    "old_value": original[key],
                    "new_value": None,
                })

        return changes

    def compare_runs(
        self,
        run_id_1: str,
        run_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare two experiment runs.

        Args:
            run_id_1: First run ID
            run_id_2: Second run ID

        Returns:
            Comparison of metrics and params
        """
        exp1 = self.get_experiment(run_id_1)
        exp2 = self.get_experiment(run_id_2)

        if not exp1 or not exp2:
            return {"success": False, "error": "One or both runs not found"}

        # Compare metrics
        metric_comparison = {}
        all_metrics = set(exp1["metrics"].keys()) | set(exp2["metrics"].keys())

        for metric in all_metrics:
            val1 = exp1["metrics"].get(metric)
            val2 = exp2["metrics"].get(metric)

            if val1 is not None and val2 is not None:
                delta = val2 - val1
                delta_pct = (delta / val1 * 100) if val1 != 0 else None
                metric_comparison[metric] = {
                    "run1": val1,
                    "run2": val2,
                    "delta": delta,
                    "delta_pct": delta_pct,
                }

        return {
            "success": True,
            "run1": {"name": exp1["run_name"], "id": run_id_1},
            "run2": {"name": exp2["run_name"], "id": run_id_2},
            "metrics": metric_comparison,
            "param_changes": self._get_changes(exp1["params"], exp2["params"]),
        }


def replay_experiment(
    experiment_id: str,
    modifications: Dict[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Quick helper to replay an experiment."""
    replayer = ExperimentReplayer()
    return replayer.replay(experiment_id, modifications, dry_run=dry_run)


def compare_experiments(run_id_1: str, run_id_2: str) -> Dict[str, Any]:
    """Quick helper to compare two experiments."""
    replayer = ExperimentReplayer()
    return replayer.compare_runs(run_id_1, run_id_2)


# CLI usage
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Replay experiment with modifications")
    parser.add_argument("experiment_id", help="Original experiment ID or name")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--max-depth", type=int, help="Max tree depth")
    parser.add_argument("--subsample", type=float, help="Subsample ratio")
    parser.add_argument("--features", help="Comma-separated feature list")
    parser.add_argument("--validation", help="Validation strategy")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without running")

    args = parser.parse_args()

    # Build modifications dict
    modifications = {}
    if args.lr is not None:
        modifications["learning_rate"] = args.lr
    if args.max_depth is not None:
        modifications["max_depth"] = args.max_depth
    if args.subsample is not None:
        modifications["subsample"] = args.subsample
    if args.features:
        modifications["features"] = args.features
    if args.validation:
        modifications["validation_strategy"] = args.validation

    if not modifications:
        print("Error: No modifications specified")
        parser.print_help()
        sys.exit(1)

    # Replay
    result = replay_experiment(args.experiment_id, modifications, dry_run=args.dry_run)

    # Print results
    print(json.dumps(result, indent=2, default=str))
