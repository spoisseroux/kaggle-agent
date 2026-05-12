#!/usr/bin/env python3
"""Ensemble v52 - Hill Climbing Optimization (Kaggle Grandmaster Technique)

TECHNIQUE: Systematic weighted ensembling from NVIDIA Kaggle Grandmaster Playbook

Uses saved predictions from v51 to find optimal weights through hill climbing:
1. Start with best single model
2. Iteratively add models with varying weights
3. Keep combinations that improve validation score
4. Continue until no improvement

AVAILABLE MODELS (from v51):
- LightGBM: CV 0.4102
- XGBoost: CV 0.4516
- CatBoost: CV 0.4617

v51 stacking gave: CV 0.4063 (worse than v50)
Hill climbing should find better weight combination than Ridge regression
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import itertools

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from core.hypothesis_db import HypothesisDatabase
    HAS_DB = True
except ImportError:
    HAS_DB = False

PREDICTIONS_DIR = Path("competitions/active/store-sales-time-series-forecasting/predictions")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")


def evaluate_weights(predictions_dict, y_val, weights):
    """Evaluate weighted combination of predictions."""
    ensemble_preds = sum(
        predictions_dict[model] * weight
        for model, weight in weights.items()
    )
    ensemble_preds = np.maximum(ensemble_preds, 0)
    return np.sqrt(mean_squared_log_error(y_val, ensemble_preds))


def hill_climbing(predictions_dict, y_val, verbose=True):
    """
    Hill climbing to find optimal model weights.

    Returns:
        dict: Optimal weights for each model
    """
    model_names = list(predictions_dict.keys())

    # Start with best single model
    best_single = min(
        model_names,
        key=lambda m: np.sqrt(mean_squared_log_error(
            y_val,
            np.maximum(predictions_dict[m], 0)
        ))
    )

    current_weights = {best_single: 1.0}
    current_cv = evaluate_weights(predictions_dict, y_val, current_weights)

    if verbose:
        print(f"Starting with: {best_single} (CV: {current_cv:.4f})")
        print()

    remaining_models = [m for m in model_names if m != best_single]
    iteration = 1
    improved = True

    # Hill climbing iterations
    while improved and remaining_models:
        improved = False
        best_addition = None
        best_new_cv = current_cv
        best_new_weights = None

        if verbose:
            print(f"Iteration {iteration}:")

        # Try adding each remaining model
        for candidate in remaining_models:
            # Test different weight allocations for new model
            # Try weights: 0.05, 0.10, 0.15, ..., 0.50
            for new_weight in np.arange(0.05, 0.55, 0.05):
                # Scale existing weights down
                test_weights = {
                    m: w * (1 - new_weight)
                    for m, w in current_weights.items()
                }
                test_weights[candidate] = new_weight

                # Evaluate
                cv = evaluate_weights(predictions_dict, y_val, test_weights)

                if cv < best_new_cv:
                    best_new_cv = cv
                    best_addition = candidate
                    best_new_weights = test_weights

                    if verbose:
                        print(f"  {candidate} @ {new_weight:.2f}: CV {cv:.4f} ✓")

        if best_addition and best_new_cv < current_cv:
            improvement = (current_cv - best_new_cv) / current_cv * 100
            if verbose:
                print(f"  → Adding {best_addition}: {current_cv:.4f} → {best_new_cv:.4f} ({improvement:.2f}% better)")
                print(f"  → New weights: {best_new_weights}")
                print()

            current_weights = best_new_weights
            current_cv = best_new_cv
            remaining_models.remove(best_addition)
            improved = True
            iteration += 1
        else:
            if verbose:
                print(f"  No improvement found. Stopping.")
                print()

    return current_weights, current_cv


def main():
    print("="*70)
    print("Ensemble v52 - Hill Climbing Optimization")
    print("="*70)
    print("Finding optimal weights for v51 base models")
    print()

    # Load saved predictions
    print("Loading saved predictions from v51...")

    try:
        predictions = {
            'LightGBM': {
                'val': np.load(PREDICTIONS_DIR / "v51_lgbm_val.npy"),
                'test': np.load(PREDICTIONS_DIR / "v51_lgbm_test.npy"),
            },
            'XGBoost': {
                'val': np.load(PREDICTIONS_DIR / "v51_xgb_val.npy"),
                'test': np.load(PREDICTIONS_DIR / "v51_xgb_test.npy"),
            },
            'CatBoost': {
                'val': np.load(PREDICTIONS_DIR / "v51_cb_val.npy"),
                'test': np.load(PREDICTIONS_DIR / "v51_cb_test.npy"),
            }
        }
        y_val = np.load(PREDICTIONS_DIR / "v51_y_val.npy")

        print("✓ Loaded predictions for 3 models")
        print()

    except FileNotFoundError as e:
        print(f"✗ Error: Could not load predictions: {e}")
        print("Run v51 first to generate predictions")
        return 1

    # Individual model CVs
    print("Individual model performance:")
    val_preds_dict = {name: pred['val'] for name, pred in predictions.items()}

    for name, val_preds in val_preds_dict.items():
        cv = np.sqrt(mean_squared_log_error(y_val, np.maximum(val_preds, 0)))
        print(f"  {name}: CV {cv:.4f}")
    print()

    # Simple average baseline
    avg_preds = np.mean([p for p in val_preds_dict.values()], axis=0)
    avg_cv = np.sqrt(mean_squared_log_error(y_val, np.maximum(avg_preds, 0)))
    print(f"Simple average: CV {avg_cv:.4f}")
    print(f"v51 stacking (Ridge): CV 0.4063")
    print(f"v50 stacking (2-model): CV 0.3925")
    print()

    # Run hill climbing
    print("="*70)
    print("HILL CLIMBING OPTIMIZATION")
    print("="*70)
    print()

    optimal_weights, optimal_cv = hill_climbing(val_preds_dict, y_val, verbose=True)

    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print("Optimal weights found:")
    for model, weight in optimal_weights.items():
        print(f"  {model}: {weight:.4f} ({weight*100:.1f}%)")
    print()
    print(f"Hill climbing CV: {optimal_cv:.4f}")
    print(f"v51 stacking: 0.4063")
    print(f"v50 stacking: 0.3925")
    print()

    if optimal_cv < 0.3925:
        print(f"✓ BEATS V50: {(0.3925 - optimal_cv) / 0.3925 * 100:.1f}% better")
    else:
        print(f"✗ WORSE: {(optimal_cv - 0.3925) / 0.3925 * 100:.1f}% worse than v50")
    print()

    # Generate test predictions
    print("Generating test predictions with optimal weights...")
    test_preds_dict = {name: pred['test'] for name, pred in predictions.items()}

    test_preds = sum(
        test_preds_dict[model] * weight
        for model, weight in optimal_weights.items()
    )
    test_preds = np.maximum(test_preds, 0)

    # Load test IDs
    test_df = pd.read_csv(Path("data/store-sales-time-series-forecasting/test.csv"))

    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": test_preds
    })

    filename = f"ensemble_v52_hillclimb_{optimal_cv:.5f}.csv".replace(".", "")
    filepath = SUBMISSION_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"Submission saved: {filename}")
    print(f"Mean prediction: {test_preds.mean():.2f}")
    print()

    # Log to hypothesis database
    if HAS_DB:
        try:
            db = HypothesisDatabase()
            hyp_id = db.add_hypothesis(
                competition="store-sales-time-series-forecasting",
                experiment_id="v52",
                hypothesis="Hill climbing finds better weights than Ridge regression",
                rationale="Systematic weight optimization can outperform linear meta-model. Pattern: ensemble methods succeed.",
                category="ensemble_methods"
            )
            db.record_result(
                hypothesis_id=hyp_id,
                cv_score=optimal_cv,
                lb_score=None,
                baseline_cv=0.3925,  # v50
                succeeded=(optimal_cv < 0.3925),
                params={'weights': optimal_weights, 'method': 'hill_climbing'},
                metadata={'v51_cv': 0.4063, 'n_iterations': len(optimal_weights)}
            )
            print("✓ Logged to hypothesis database")
        except Exception as e:
            print(f"Note: Could not log to hypothesis DB: {e}")

    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
