#!/usr/bin/env python3
"""Feature Ablation Study

Remove each feature one at a time to measure true importance.
More insightful than feature importance scores - shows actual CV impact.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data/store-sales-time-series-forecasting")
RESULTS_FILE = Path("experiments/feature_ablation_results.json")


def test_feature_ablation():
    """Test removing each feature individually."""
    print("="*70)
    print("FEATURE ABLATION STUDY")
    print("="*70)
    print("Removing each feature one at a time to measure true impact")
    print()

    # Load data
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
    )
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])

    # Create features
    train = train.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    train["day_of_week"] = train["date"].dt.dayofweek
    train["is_weekend"] = (train["day_of_week"] >= 5).astype(int)
    train["Lag_3"] = train.groupby(["store_nbr", "family"], observed=True)["sales"].shift(3)
    train["Lag_7"] = train.groupby(["store_nbr", "family"], observed=True)["sales"].shift(7)

    for window in [7, 14, 30, 60, 90]:
        train[f"Roll_mean_{window}"] = train.groupby(["store_nbr", "family"], observed=True)["sales"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    train["Roll_std_7"] = train.groupby(["store_nbr", "family"], observed=True)["sales"].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )

    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    train["is_holiday"] = train["date"].isin(national_holidays).astype(int)
    train = train.dropna()

    # All 12 features
    all_features = ["onpromotion", "day_of_week", "is_weekend", "Lag_3", "Lag_7",
                    "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "Roll_mean_60",
                    "Roll_mean_90", "Roll_std_7", "is_holiday"]

    cutoff_date = train["date"].max() - pd.Timedelta(days=30)
    val_mask = train["date"] >= cutoff_date
    train_mask = ~val_mask

    y_train = train.loc[train_mask, "sales"]
    y_val = train.loc[val_mask, "sales"]

    # Baseline with all features
    print("Baseline (all 12 features)...")
    X_train = train.loc[train_mask, all_features]
    X_val = train.loc[val_mask, all_features]

    model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        num_leaves=64,
        max_depth=6,
        n_estimators=400,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    baseline_preds = model.predict(X_val)
    baseline_cv = np.sqrt(mean_squared_log_error(y_val, np.maximum(baseline_preds, 0)))

    print(f"Baseline CV: {baseline_cv:.4f}")
    print()

    # Test removing each feature
    results = []

    print("Testing ablation (removing one feature at a time):")
    print("-" * 70)

    for removed_feature in all_features:
        # Features without this one
        test_features = [f for f in all_features if f != removed_feature]

        X_train_test = train.loc[train_mask, test_features]
        X_val_test = train.loc[val_mask, test_features]

        model_test = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.05,
            num_leaves=64,
            max_depth=6,
            n_estimators=400,
            random_state=42,
            verbosity=-1
        )
        model_test.fit(X_train_test, y_train, eval_set=[(X_val_test, y_val)],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        preds = model_test.predict(X_val_test)
        cv = np.sqrt(mean_squared_log_error(y_val, np.maximum(preds, 0)))

        delta = cv - baseline_cv
        pct_change = (delta / baseline_cv) * 100

        result = {
            'removed_feature': removed_feature,
            'cv_with_removal': float(cv),
            'baseline_cv': float(baseline_cv),
            'delta': float(delta),
            'pct_change': float(pct_change),
            'degraded': bool(cv > baseline_cv)
        }

        results.append(result)

        symbol = "✗" if cv > baseline_cv else "≈"
        print(f"{symbol} Remove {removed_feature:<20} CV {cv:.4f}  (Δ {delta:+.4f}, {pct_change:+.2f}%)")

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("="*70)
    print("ABLATION SUMMARY")
    print("="*70)

    # Sort by impact
    sorted_results = sorted(results, key=lambda x: abs(x['delta']), reverse=True)

    print("\nMost critical features (largest degradation when removed):")
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"{i}. {r['removed_feature']:<20} Δ {r['delta']:+.4f} ({r['pct_change']:+.2f}%)")

    print("\nLeast critical features (smallest degradation when removed):")
    for i, r in enumerate(sorted_results[-5:], 1):
        print(f"{i}. {r['removed_feature']:<20} Δ {r['delta']:+.4f} ({r['pct_change']:+.2f}%)")

    all_degraded = all(r['degraded'] for r in results)
    if all_degraded:
        print("\n⚠️ FINDING: All features critical - cannot remove any without degradation")
    else:
        removable = [r for r in results if not r['degraded']]
        print(f"\n✓ FINDING: {len(removable)} features can be removed without degradation:")
        for r in removable:
            print(f"  - {r['removed_feature']}")

    print(f"\nResults saved: {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    results = test_feature_ablation()
