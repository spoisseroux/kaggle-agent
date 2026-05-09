#!/usr/bin/env python3
"""Evaluate an experiment before submission.

Usage:
    python scripts/eval_experiment.py --cv 0.350 --baseline 0.367 --validation TimeSeriesSplit
    python scripts/eval_experiment.py --run-id <mlflow_run_id>
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.evals import ExperimentEval
from core.notify import notify
import json


def eval_from_args(args):
    """Evaluate from command-line arguments."""
    evaluator = ExperimentEval()

    config = {
        "validation_strategy": args.validation,
        "features": args.features.split(",") if args.features else [],
        "hyperparameters": {}
    }

    if args.lr:
        config["hyperparameters"]["learning_rate"] = args.lr
    if args.depth:
        config["hyperparameters"]["max_depth"] = args.depth

    result = evaluator.evaluate_experiment(
        experiment_config=config,
        cv_score=args.cv,
        baseline_score=args.baseline,
        competition_type=args.comp_type
    )

    return result


def eval_from_mlflow(run_id):
    """Evaluate from MLflow run."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")

        run = mlflow.get_run(run_id)
        params = run.data.params
        metrics = run.data.metrics

        config = {
            "validation_strategy": params.get("validation", "unknown"),
            "features": params.get("features", "").split(","),
            "hyperparameters": {
                k: float(v) for k, v in params.items()
                if k in ["learning_rate", "max_depth", "subsample", "colsample_bytree"]
            }
        }

        cv_score = metrics.get("cv_rmsle", metrics.get("cv_score", 0))
        baseline_score = 0.367  # Default - should be loaded from config

        evaluator = ExperimentEval()
        result = evaluator.evaluate_experiment(
            experiment_config=config,
            cv_score=cv_score,
            baseline_score=baseline_score
        )

        return result

    except Exception as e:
        return {
            "passed": False,
            "issues": [f"Failed to load MLflow run: {e}"],
            "warnings": [],
            "score": 0.0,
            "recommendation": "Error evaluating experiment"
        }


def format_result(result):
    """Format evaluation result for display."""
    lines = []

    # Header
    if result["passed"]:
        lines.append("✅ EVALUATION PASSED")
    else:
        lines.append("❌ EVALUATION FAILED")

    lines.append("")
    lines.append(f"Score: {result['score']:.2f}/1.0")
    lines.append(f"Recommendation: {result['recommendation']}")
    lines.append("")

    # Issues
    if result["issues"]:
        lines.append("🚨 CRITICAL ISSUES:")
        for issue in result["issues"]:
            lines.append(f"  - {issue}")
        lines.append("")

    # Warnings
    if result["warnings"]:
        lines.append("⚠️ WARNINGS:")
        for warning in result["warnings"]:
            lines.append(f"  - {warning}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate experiment quality")

    # Option 1: Direct args
    parser.add_argument("--cv", type=float, help="CV score")
    parser.add_argument("--baseline", type=float, help="Baseline score")
    parser.add_argument("--validation", default="unknown", help="Validation strategy")
    parser.add_argument("--features", help="Comma-separated feature list")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--depth", type=int, help="Max depth")
    parser.add_argument("--comp-type", default="time-series", help="Competition type")

    # Option 2: MLflow run
    parser.add_argument("--run-id", help="MLflow run ID to evaluate")

    # Output
    parser.add_argument("--telegram", action="store_true", help="Send result to Telegram")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Get evaluation result
    if args.run_id:
        result = eval_from_mlflow(args.run_id)
    elif args.cv and args.baseline:
        result = eval_from_args(args)
    else:
        parser.error("Either provide --cv and --baseline, or --run-id")

    # Output result
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        output = format_result(result)
        print(output)

        if args.telegram:
            notify(output)

    # Exit code based on result
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
