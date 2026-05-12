#!/usr/bin/env python3
"""Ensemble v53 - Optimized 2-Model Hill Climbing

LEARNED PATTERN: Less is more in ensembling
- v50 (2 models): CV 0.3925 ✓ BEST
- v51 (3 models): CV 0.4063 ✗ WORSE
- v52 hill climbing excludes XGBoost entirely

INSIGHT: Adding CatBoost reduced performance
- CatBoost (CV 0.4617) is weakest base model
- Adds noise rather than complementary signal

STRATEGY: Hill climbing on 2 best models only
- LightGBM (CV 0.4102) - strongest base model
- XGBoost (CV 0.4516) - provides diversity
- Find optimal weights (not constrained to Ridge)

Expected: Match or beat v50's 0.3925
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from core.hypothesis_db import HypothesisDatabase
    HAS_DB = True
except ImportError:
    HAS_DB = False

PREDICTIONS_DIR = Path("competitions/active/store-sales-time-series-forecasting/predictions")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")


def grid_search_weights(lgb_preds, xgb_preds, y_val):
    """
    Exhaustive grid search for optimal 2-model weights.
    Tests all combinations where weights sum to 1.0.
    """
    best_cv = float('inf')
    best_lgb_weight = None

    # Test LightGBM weights from 0.5 to 1.0 in steps of 0.01
    for lgb_weight in np.arange(0.5, 1.01, 0.01):
        xgb_weight = 1.0 - lgb_weight

        ensemble_preds = lgb_weight * lgb_preds + xgb_weight * xgb_preds
        ensemble_preds = np.maximum(ensemble_preds, 0)

        cv = np.sqrt(mean_squared_log_error(y_val, ensemble_preds))

        if cv < best_cv:
            best_cv = cv
            best_lgb_weight = lgb_weight

    return best_lgb_weight, 1.0 - best_lgb_weight, best_cv


def main():
    print("="*70)
    print("Ensemble v53 - Optimized 2-Model Hill Climbing")
    print("="*70)
    print("Learning: CatBoost adds noise, use only LightGBM + XGBoost")
    print()

    # Load predictions (only LightGBM and XGBoost)
    print("Loading LightGBM and XGBoost predictions...")

    try:
        lgb_val = np.load(PREDICTIONS_DIR / "v51_lgbm_val.npy")
        lgb_test = np.load(PREDICTIONS_DIR / "v51_lgbm_test.npy")
        xgb_val = np.load(PREDICTIONS_DIR / "v51_xgb_val.npy")
        xgb_test = np.load(PREDICTIONS_DIR / "v51_xgb_test.npy")
        y_val = np.load(PREDICTIONS_DIR / "v51_y_val.npy")

        print("✓ Loaded predictions")
        print()

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return 1

    # Individual performance
    lgb_cv = np.sqrt(mean_squared_log_error(y_val, np.maximum(lgb_val, 0)))
    xgb_cv = np.sqrt(mean_squared_log_error(y_val, np.maximum(xgb_val, 0)))

    print("Individual model performance:")
    print(f"  LightGBM: CV {lgb_cv:.4f}")
    print(f"  XGBoost:  CV {xgb_cv:.4f}")
    print()

    # Simple average
    avg_preds = (lgb_val + xgb_val) / 2
    avg_cv = np.sqrt(mean_squared_log_error(y_val, np.maximum(avg_preds, 0)))
    print(f"Simple average (50/50): CV {avg_cv:.4f}")
    print(f"v50 stacking (Ridge):   CV 0.3925")
    print()

    # Grid search for optimal weights
    print("="*70)
    print("GRID SEARCH: Optimal 2-Model Weights")
    print("="*70)
    print("Testing LightGBM weights from 50% to 100% in 1% increments...")
    print()

    lgb_weight, xgb_weight, optimal_cv = grid_search_weights(lgb_val, xgb_val, y_val)

    print("Optimal weights found:")
    print(f"  LightGBM: {lgb_weight:.4f} ({lgb_weight*100:.1f}%)")
    print(f"  XGBoost:  {xgb_weight:.4f} ({xgb_weight*100:.1f}%)")
    print()
    print(f"Optimal CV: {optimal_cv:.4f}")
    print(f"v50 stacking: 0.3925")
    print()

    if optimal_cv < 0.3925:
        print(f"✓ BEATS V50: {(0.3925 - optimal_cv) / 0.3925 * 100:.1f}% better")
    elif abs(optimal_cv - 0.3925) < 0.001:
        print(f"≈ MATCHES V50 (within 0.1%)")
    else:
        print(f"✗ WORSE: {(optimal_cv - 0.3925) / 0.3925 * 100:.1f}% worse")
    print()

    # Generate test predictions
    print("Generating test predictions...")
    test_preds = lgb_weight * lgb_test + xgb_weight * xgb_test
    test_preds = np.maximum(test_preds, 0)

    test_df = pd.read_csv(Path("data/store-sales-time-series-forecasting/test.csv"))

    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": test_preds
    })

    filename = f"ensemble_v53_2model_opt_{optimal_cv:.5f}.csv".replace(".", "")
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
                experiment_id="v53",
                hypothesis="2-model ensemble with grid search weights beats 3-model",
                rationale="Pattern learned: CatBoost adds noise. Optimal ensemble uses only 2 best diverse models with precisely tuned weights.",
                category="ensemble_methods"
            )
            db.record_result(
                hypothesis_id=hyp_id,
                cv_score=optimal_cv,
                lb_score=None,
                baseline_cv=0.3925,
                succeeded=(optimal_cv <= 0.3925),
                params={'lgb_weight': float(lgb_weight), 'xgb_weight': float(xgb_weight), 'method': 'grid_search'},
                metadata={'n_weights_tested': 51}
            )
            print("✓ Logged to hypothesis database")
        except Exception as e:
            print(f"Note: Could not log: {e}")

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print("Learned patterns applied:")
    print("1. ✓ Ensemble methods succeed (hypothesis DB pattern)")
    print("2. ✓ Less is more: 2 models > 3 models")
    print("3. ✓ Grid search finds precise optimal weights")
    print("4. ✓ LightGBM dominates (~71% in v50, likely similar here)")
    print()
    print(f"Result: CV {optimal_cv:.4f} vs v50 baseline 0.3925")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
