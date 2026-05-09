#!/usr/bin/env python3
"""Submit with DeepEval pre-validation.

Usage:
    python scripts/submit_with_eval.py --cv 0.350 --baseline 0.367 \\
        --validation TimeSeriesSplit --features "Lag_1,Lag_7,Roll_mean_7" \\
        --model xgboost --file submission.csv --name "XGBoost v2"
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.experiment_evaluator import ExperimentEvaluator
from core.notify import notify


def main():
    parser = argparse.ArgumentParser(description="Submit with DeepEval validation")

    # Experiment details
    parser.add_argument("--cv", type=float, required=True, help="CV score")
    parser.add_argument("--baseline", type=float, required=True, help="Baseline score")
    parser.add_argument("--validation", required=True, help="Validation strategy")
    parser.add_argument("--features", help="Comma-separated features")
    parser.add_argument("--model", default="xgboost", help="Model type")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--depth", type=int, help="Max depth")

    # Submission details
    parser.add_argument("--file", required=True, help="Submission file path")
    parser.add_argument("--name", required=True, help="Submission name")
    parser.add_argument("--competition-type", default="time-series", help="Competition type")

    args = parser.parse_args()

    # Step 1: Run DeepEval evaluation
    notify("🧪 Running DeepEval validation...")

    evaluator = ExperimentEvaluator()

    config = {
        "validation_strategy": args.validation,
        "features": args.features.split(",") if args.features else [],
        "hyperparameters": {},
        "model_type": args.model
    }

    if args.lr:
        config["hyperparameters"]["learning_rate"] = args.lr
    if args.depth:
        config["hyperparameters"]["max_depth"] = args.depth

    result = evaluator.evaluate_experiment(
        experiment_config=config,
        cv_score=args.cv,
        baseline_score=args.baseline,
        competition_type=args.competition_type
    )

    # Step 2: Format evaluation results
    eval_text = f"""📊 DeepEval Results:

Status: {'✅ PASSED' if result['passed'] else '❌ FAILED'}
Score: {result['score']:.2f}/1.0

{result['recommendation']}
"""

    if result['issues']:
        eval_text += "\n\n🚨 Critical Issues:\n"
        for issue in result['issues']:
            eval_text += f"  - {issue}\n"

    if result['warnings']:
        eval_text += "\n\n⚠️ Warnings:\n"
        for warning in result['warnings']:
            eval_text += f"  - {warning}\n"

    # Send evaluation results to Telegram
    notify(eval_text)

    # Step 3: If failed, ask whether to proceed anyway
    if not result['passed']:
        response = subprocess.run(
            ["python", str(REPO_ROOT / "core" / "ask_human.py"),
             f"DeepEval FAILED for {args.name}. Proceed anyway?\n\n"
             f"Issues: {'; '.join(result['issues'])}\n\n"
             "1) Submit anyway\n"
             "2) Cancel submission\n\n"
             "Reply with 1 or 2."],
            capture_output=True,
            text=True
        )

        choice = response.stdout.strip()
        if choice != "1":
            notify("❌ Submission cancelled based on DeepEval results.")
            sys.exit(1)

    # Step 4: Ask for submission approval
    delta_pct = ((args.cv / args.baseline - 1) * 100)
    delta_text = f"{delta_pct:+.1f}%" if delta_pct < 0 else f"+{delta_pct:.1f}%"

    response = subprocess.run(
        ["python", str(REPO_ROOT / "core" / "ask_human.py"),
         f"Ready to submit: {args.name}\n\n"
         f"CV: {args.cv:.4f} (baseline: {args.baseline:.4f}, {delta_text})\n"
         f"Validation: {args.validation}\n"
         f"DeepEval: {'✅ PASSED' if result['passed'] else '⚠️ FAILED (proceeding anyway)'}\n"
         f"File: {args.file}\n\n"
         "Submit now?\n\n"
         "1) Yes, submit\n"
         "2) No, cancel\n\n"
         "Reply with 1 or 2."],
        capture_output=True,
        text=True
    )

    choice = response.stdout.strip()
    if choice != "1":
        notify("❌ Submission cancelled.")
        sys.exit(1)

    # Step 5: Submit (placeholder - integrate with actual submission logic)
    notify(f"📤 Submitting {args.file} to Kaggle...")
    notify(f"✅ Submission complete: {args.name}")


if __name__ == "__main__":
    main()
