#!/usr/bin/env python3
"""Investigate why v50's Ridge regression outperforms grid search optimization.

v50: Ridge(alpha=1.0) with 71% LGB + 27% XGB → CV 0.3925
v53: Grid search with 99% LGB + 1% XGB → CV 0.4099

Hypotheses to test:
1. Ridge's intercept term provides bias correction
2. Ridge regularization prevents overfitting to validation
3. Training on meta-features provides implicit regularization
4. Some implementation difference we haven't noticed
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge

PREDICTIONS_DIR = Path("competitions/active/store-sales-time-series-forecasting/predictions")


def test_hypothesis_1_intercept():
    """Test if Ridge's intercept term is the key."""
    print("="*70)
    print("HYPOTHESIS 1: Ridge Intercept Term")
    print("="*70)
    print()

    # Load predictions
    lgb_val = np.load(PREDICTIONS_DIR / "v51_lgbm_val.npy")
    xgb_val = np.load(PREDICTIONS_DIR / "v51_xgb_val.npy")
    y_val = np.load(PREDICTIONS_DIR / "v51_y_val.npy")

    # Create meta-features
    meta_train = np.column_stack([lgb_val, xgb_val])

    # Test 1: Ridge WITH intercept (v50 approach)
    ridge_with_intercept = Ridge(alpha=1.0, fit_intercept=True)
    ridge_with_intercept.fit(meta_train, y_val)

    preds_with = np.maximum(ridge_with_intercept.predict(meta_train), 0)
    cv_with = np.sqrt(mean_squared_log_error(y_val, preds_with))

    print("Ridge WITH intercept:")
    print(f"  Weights: LGB {ridge_with_intercept.coef_[0]:.4f}, XGB {ridge_with_intercept.coef_[1]:.4f}")
    print(f"  Intercept: {ridge_with_intercept.intercept_:.4f}")
    print(f"  CV: {cv_with:.4f}")
    print()

    # Test 2: Ridge WITHOUT intercept
    ridge_no_intercept = Ridge(alpha=1.0, fit_intercept=False)
    ridge_no_intercept.fit(meta_train, y_val)

    preds_without = np.maximum(ridge_no_intercept.predict(meta_train), 0)
    cv_without = np.sqrt(mean_squared_log_error(y_val, preds_without))

    print("Ridge WITHOUT intercept:")
    print(f"  Weights: LGB {ridge_no_intercept.coef_[0]:.4f}, XGB {ridge_no_intercept.coef_[1]:.4f}")
    print(f"  CV: {cv_without:.4f}")
    print()

    # Test 3: Manual weighted average using Ridge weights
    lgb_weight = ridge_with_intercept.coef_[0]
    xgb_weight = ridge_with_intercept.coef_[1]

    manual_preds = lgb_weight * lgb_val + xgb_weight * xgb_val + ridge_with_intercept.intercept_
    manual_preds = np.maximum(manual_preds, 0)
    cv_manual = np.sqrt(mean_squared_log_error(y_val, manual_preds))

    print("Manual application of Ridge weights + intercept:")
    print(f"  CV: {cv_manual:.4f}")
    print()

    print("FINDINGS:")
    print(f"  Intercept impact: {abs(cv_with - cv_without):.4f} CV difference")
    if abs(cv_with - cv_manual) < 0.001:
        print(f"  ✓ Manual replication matches Ridge exactly")
    print()

    return cv_with


def test_hypothesis_2_regularization():
    """Test different Ridge alpha values."""
    print("="*70)
    print("HYPOTHESIS 2: Ridge Regularization Strength")
    print("="*70)
    print()

    lgb_val = np.load(PREDICTIONS_DIR / "v51_lgbm_val.npy")
    xgb_val = np.load(PREDICTIONS_DIR / "v51_xgb_val.npy")
    y_val = np.load(PREDICTIONS_DIR / "v51_y_val.npy")

    meta_train = np.column_stack([lgb_val, xgb_val])

    print("Testing different alpha values:")
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    results = []

    for alpha in alphas:
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(meta_train, y_val)

        preds = np.maximum(ridge.predict(meta_train), 0)
        cv = np.sqrt(mean_squared_log_error(y_val, preds))

        results.append({
            'alpha': alpha,
            'cv': cv,
            'lgb_weight': ridge.coef_[0],
            'xgb_weight': ridge.coef_[1],
            'intercept': ridge.intercept_
        })

        print(f"  alpha={alpha:6.3f}: CV {cv:.4f}, LGB {ridge.coef_[0]:.3f}, XGB {ridge.coef_[1]:.3f}, intercept {ridge.intercept_:.2f}")

    best = min(results, key=lambda x: x['cv'])
    print()
    print(f"Best alpha: {best['alpha']} → CV {best['cv']:.4f}")
    print(f"v50 used alpha=1.0 → CV {[r for r in results if r['alpha']==1.0][0]['cv']:.4f}")
    print()

    return best['cv']


def test_hypothesis_3_normalization():
    """Test if prediction normalization matters."""
    print("="*70)
    print("HYPOTHESIS 3: Prediction Normalization")
    print("="*70)
    print()

    lgb_val = np.load(PREDICTIONS_DIR / "v51_lgbm_val.npy")
    xgb_val = np.load(PREDICTIONS_DIR / "v51_xgb_val.npy")
    y_val = np.load(PREDICTIONS_DIR / "v51_y_val.npy")

    # Test 1: Standard approach (what we've been doing)
    meta_train = np.column_stack([lgb_val, xgb_val])

    ridge = Ridge(alpha=1.0)
    ridge.fit(meta_train, y_val)

    preds = np.maximum(ridge.predict(meta_train), 0)
    cv_standard = np.sqrt(mean_squared_log_error(y_val, preds))

    print(f"Standard (no normalization): CV {cv_standard:.4f}")

    # Test 2: Normalize predictions to zero mean
    lgb_mean = lgb_val.mean()
    xgb_mean = xgb_val.mean()

    meta_train_normalized = np.column_stack([
        lgb_val - lgb_mean,
        xgb_val - xgb_mean
    ])

    ridge_norm = Ridge(alpha=1.0)
    ridge_norm.fit(meta_train_normalized, y_val)

    preds_norm = ridge_norm.predict(meta_train_normalized)
    preds_norm = np.maximum(preds_norm, 0)
    cv_normalized = np.sqrt(mean_squared_log_error(y_val, preds_norm))

    print(f"Normalized (zero mean): CV {cv_normalized:.4f}")
    print()

    return cv_standard


def compare_with_v50_reported():
    """Compare our results with v50's reported performance."""
    print("="*70)
    print("COMPARISON WITH V50 REPORTED RESULTS")
    print("="*70)
    print()

    print("v50 reported:")
    print("  Base LGB: CV 0.4102")
    print("  Base XGB: CV 0.4516")
    print("  Ridge stacking: CV 0.3925")
    print("  LGB weight: 71.45%")
    print("  XGB weight: 27.46%")
    print("  Intercept: -0.4937")
    print()

    # Our replication
    lgb_val = np.load(PREDICTIONS_DIR / "v51_lgbm_val.npy")
    xgb_val = np.load(PREDICTIONS_DIR / "v51_xgb_val.npy")
    y_val = np.load(PREDICTIONS_DIR / "v51_y_val.npy")

    lgb_cv = np.sqrt(mean_squared_log_error(y_val, np.maximum(lgb_val, 0)))
    xgb_cv = np.sqrt(mean_squared_log_error(y_val, np.maximum(xgb_val, 0)))

    print("Our replication:")
    print(f"  Base LGB: CV {lgb_cv:.4f}")
    print(f"  Base XGB: CV {xgb_cv:.4f}")

    if abs(lgb_cv - 0.4102) < 0.001 and abs(xgb_cv - 0.4516) < 0.001:
        print("  ✓ Base models match exactly!")
    else:
        print("  ✗ Base models differ!")
    print()

    # Ridge stacking
    meta_train = np.column_stack([lgb_val, xgb_val])
    ridge = Ridge(alpha=1.0)
    ridge.fit(meta_train, y_val)

    preds = np.maximum(ridge.predict(meta_train), 0)
    cv = np.sqrt(mean_squared_log_error(y_val, preds))

    total_weight = abs(ridge.coef_[0]) + abs(ridge.coef_[1])
    lgb_pct = abs(ridge.coef_[0]) / total_weight * 100
    xgb_pct = abs(ridge.coef_[1]) / total_weight * 100

    print(f"  Ridge stacking: CV {cv:.4f}")
    print(f"  LGB weight: {lgb_pct:.2f}%")
    print(f"  XGB weight: {xgb_pct:.2f}%")
    print(f"  Intercept: {ridge.intercept_:.4f}")
    print()

    if abs(cv - 0.3925) < 0.01:
        print("  ✓ CV matches v50 within 1%!")
    else:
        print(f"  ✗ CV differs by {abs(cv - 0.3925):.4f}")
        print()
        print("  POSSIBLE REASONS:")
        print("  1. Different validation split")
        print("  2. Different base model training")
        print("  3. Different preprocessing")

    return cv


def main():
    print("Investigating v50's Ridge Regression Advantage")
    print("="*70)
    print()

    # Run all tests
    cv_intercept = test_hypothesis_1_intercept()
    cv_regularization = test_hypothesis_2_regularization()
    cv_normalization = test_hypothesis_3_normalization()
    cv_v50_comparison = compare_with_v50_reported()

    # Summary
    print("="*70)
    print("INVESTIGATION SUMMARY")
    print("="*70)
    print()
    print(f"1. Ridge with intercept: CV {cv_intercept:.4f}")
    print(f"2. Best regularization: CV {cv_regularization:.4f}")
    print(f"3. No normalization: CV {cv_normalization:.4f}")
    print(f"4. V50 comparison: CV {cv_v50_comparison:.4f}")
    print()
    print(f"v50 target: CV 0.3925")
    print(f"Closest achieved: CV {min(cv_intercept, cv_regularization, cv_normalization, cv_v50_comparison):.4f}")
    print()

    if min(cv_intercept, cv_regularization, cv_normalization, cv_v50_comparison) < 0.3930:
        print("✓ SUCCESSFULLY REPLICATED V50!")
    else:
        print("✗ Could not replicate v50 - base models or split likely differ")


if __name__ == "__main__":
    main()
