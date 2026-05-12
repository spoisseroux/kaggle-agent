#!/usr/bin/env python3
"""Ensemble v67 - Ridge 90/10 FIXED (Alignment Bug Fix)

BUG in v63: Paired store-family sorted predictions with ID-sorted test IDs
FIX: Ensure predictions and IDs are in same order

v63 LB: 3.588 (catastrophic due to misalignment)
v67 Expected: ~0.451 (should match v50 since weights are similar)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

try:
    from core.hypothesis_db import HypothesisDatabase
    HAS_DB = True
except ImportError:
    HAS_DB = False

DATA_DIR = Path("data/store-sales-time-series-forecasting")
PREDICTIONS_DIR = Path("competitions/active/store-sales-time-series-forecasting/predictions")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("="*70)
    print("Ensemble v67 - Ridge 90/10 (ALIGNMENT FIXED)")
    print("="*70)
    print("Fixing v63's misalignment bug")
    print()

    # Load predictions (these are in STORE-FAMILY-DATE sorted order from v51)
    lgb_val = np.load(PREDICTIONS_DIR / "v51_lgbm_val.npy")
    lgb_test = np.load(PREDICTIONS_DIR / "v51_lgbm_test.npy")
    xgb_val = np.load(PREDICTIONS_DIR / "v51_xgb_val.npy")
    xgb_test = np.load(PREDICTIONS_DIR / "v51_xgb_test.npy")
    y_val = np.load(PREDICTIONS_DIR / "v51_y_val.npy")

    # Get intercept from Ridge
    meta_train = np.column_stack([lgb_val, xgb_val])
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(meta_train, y_val)
    intercept = ridge.intercept_

    print(f"Learned intercept: {intercept:.4f}")
    print()

    # Apply 90/10 weights
    lgb_weight = 0.90
    xgb_weight = 0.10

    val_preds = lgb_weight * lgb_val + xgb_weight * xgb_val + intercept
    val_preds = np.maximum(val_preds, 0)
    cv = np.sqrt(mean_squared_log_error(y_val, val_preds))

    print("Results:")
    print(f"  Weights: {lgb_weight:.2f} LGB + {xgb_weight:.2f} XGB + {intercept:.4f}")
    print(f"  CV: {cv:.4f}")
    print(f"  v50 baseline: 0.3925")
    print()

    # Test predictions
    test_preds = lgb_weight * lgb_test + xgb_weight * xgb_test + intercept
    test_preds = np.maximum(test_preds, 0)

    # FIX: Load test_df and sort it THE SAME WAY v51 sorted it
    test_df = pd.read_csv(DATA_DIR / "test.csv",
                         dtype={"store_nbr": "category", "family": "category"})

    # CRITICAL: Sort by store_nbr, family, date to match v51 prediction order
    test_df_sorted = test_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Now predictions[i] corresponds to test_df_sorted.iloc[i]
    submission = pd.DataFrame({
        "id": test_df_sorted["id"],
        "sales": test_preds
    })

    # Verify alignment
    print("Alignment verification:")
    print(f"  test_preds length: {len(test_preds)}")
    print(f"  test_df_sorted length: {len(test_df_sorted)}")
    print(f"  submission length: {len(submission)}")
    print(f"  First 5 IDs: {submission['id'].values[:5]}")
    print(f"  First 5 predictions: {submission['sales'].values[:5]}")
    print()

    cv_str = f"{cv:.5f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"ensemble_v67_ridge_90_10_fixed_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # Hypothesis DB
    if HAS_DB:
        try:
            db = HypothesisDatabase()
            hyp_id = db.add_hypothesis(
                competition="store-sales-time-series-forecasting",
                experiment_id="v67",
                hypothesis="Ridge 90/10 with FIXED alignment (v63 had misalignment bug)",
                rationale="v63 failed (LB 3.588) due to pairing store-family sorted predictions with ID-sorted IDs. v67 sorts test IDs to match prediction order.",
                category="ensemble_methods"
            )
            db.record_result(
                hypothesis_id=hyp_id,
                cv_score=cv,
                lb_score=None,
                baseline_cv=0.3925,
                succeeded=(cv < 0.40),
                params={'lgb_weight': lgb_weight, 'xgb_weight': xgb_weight, 'intercept': float(intercept)},
                metadata={'bug_fix': 'alignment', 'v63_lb': 3.588, 'expected_lb': '0.45-0.46'}
            )
        except Exception as e:
            print(f"Note: Could not log: {e}")

    print()
    print("="*70)
    print("ALIGNMENT BUG FIXED")
    print("="*70)
    print(f"v63 (broken): LB 3.588 (ID-sorted IDs + store-family sorted preds)")
    print(f"v67 (fixed):  Expected LB ~0.451 (properly aligned)")

    return {
        "cv": cv,
        "vs_v50": cv - 0.3925,
        "bug_fix": "alignment",
        "expected_lb": "0.45-0.46"
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
