#!/usr/bin/env python3
"""Hyperparameter tuning for v67 Ridge 90/10 ensemble.

Baseline v67: CV 0.3908
- LightGBM: lr=0.05, depth=6, leaves=64, n_est=600
- XGBoost: lr=0.05, depth=6, n_est=600
- Ridge: 90% LGB + 10% XGB + intercept
- Validation: 30-day temporal holdout

Tests:
1. Learning rate variations (0.03, 0.07)
2. Max depth variations (5, 7, 8)
3. n_estimators variations (400, 800, 1000)
4. num_leaves variations (32, 96)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb

DATA_DIR = Path("data/store-sales-time-series-forecasting")


def load_data():
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
    )
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    return train, holidays


def create_features(df):
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["Lag_3"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(3)
    df["Lag_7"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(7)

    for window in [7, 14, 30, 60, 90]:
        df[f"Roll_mean_{window}"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )

    df["Roll_std_7"] = (
        df.groupby(["store_nbr", "family"], observed=True)["sales"]
        .transform(lambda x: x.rolling(window=7, min_periods=1).std())
    )

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def test_config(lgb_params, xgb_params, config_name):
    """Test a single hyperparameter configuration."""
    train_df, holidays = load_data()
    train_df = create_features(train_df)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    # 30-day temporal holdout (same as v67 baseline)
    cutoff_date = train_df["date"].max() - pd.Timedelta(days=30)
    val_mask = train_df["date"] >= cutoff_date
    train_mask = ~val_mask

    feature_cols = [
        "onpromotion", "day_of_week", "is_weekend", "is_holiday",
        "Lag_3", "Lag_7",
        "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "Roll_mean_60", "Roll_mean_90",
        "Roll_std_7"
    ]

    X = train_df[feature_cols]
    y = train_df["sales"]
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    # Train LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    lgb_val = np.maximum(lgb_model.predict(X_val), 0)
    lgb_cv = np.sqrt(mean_squared_log_error(y_val, lgb_val))

    # Train XGBoost (no early stopping to avoid API issues)
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)

    xgb_val = np.maximum(xgb_model.predict(X_val), 0)
    xgb_cv = np.sqrt(mean_squared_log_error(y_val, xgb_val))

    # Ridge 90/10 with intercept
    val_preds = 0.90 * lgb_val + 0.10 * xgb_val

    ridge = Ridge(alpha=1.0, fit_intercept=True)
    meta_train = np.column_stack([lgb_val, xgb_val])
    ridge.fit(meta_train, y_val)
    intercept = ridge.intercept_

    val_preds = val_preds + intercept
    val_preds = np.maximum(val_preds, 0)

    ensemble_cv = np.sqrt(mean_squared_log_error(y_val, val_preds))

    return {
        "config": config_name,
        "lgb_cv": lgb_cv,
        "xgb_cv": xgb_cv,
        "ensemble_cv": ensemble_cv,
        "intercept": intercept,
        "improvement_pct": ((0.3908 - ensemble_cv) / 0.3908) * 100
    }


def main():
    print("="*70)
    print("v67 Hyperparameter Tuning")
    print("="*70)
    print(f"Baseline v67: CV 0.3908")
    print()

    # Baseline configuration
    baseline_lgb = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 6,
        "n_estimators": 600,
        "random_state": 42,
        "verbosity": -1
    }
    baseline_xgb = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 600,
        "random_state": 42
    }

    results = []

    # Test 1: Lower learning rate
    print("Testing lower learning rate (0.03)...")
    lgb_params = baseline_lgb.copy()
    xgb_params = baseline_xgb.copy()
    lgb_params["learning_rate"] = 0.03
    xgb_params["learning_rate"] = 0.03
    results.append(test_config(lgb_params, xgb_params, "lr=0.03"))
    print(f"  CV: {results[-1]['ensemble_cv']:.4f} ({results[-1]['improvement_pct']:+.2f}%)")

    # Test 2: Higher learning rate
    print("Testing higher learning rate (0.07)...")
    lgb_params = baseline_lgb.copy()
    xgb_params = baseline_xgb.copy()
    lgb_params["learning_rate"] = 0.07
    xgb_params["learning_rate"] = 0.07
    results.append(test_config(lgb_params, xgb_params, "lr=0.07"))
    print(f"  CV: {results[-1]['ensemble_cv']:.4f} ({results[-1]['improvement_pct']:+.2f}%)")

    # Test 3: Shallower trees
    print("Testing shallower trees (depth=5)...")
    lgb_params = baseline_lgb.copy()
    xgb_params = baseline_xgb.copy()
    lgb_params["max_depth"] = 5
    xgb_params["max_depth"] = 5
    results.append(test_config(lgb_params, xgb_params, "depth=5"))
    print(f"  CV: {results[-1]['ensemble_cv']:.4f} ({results[-1]['improvement_pct']:+.2f}%)")

    # Test 4: Deeper trees (7)
    print("Testing deeper trees (depth=7)...")
    lgb_params = baseline_lgb.copy()
    xgb_params = baseline_xgb.copy()
    lgb_params["max_depth"] = 7
    xgb_params["max_depth"] = 7
    results.append(test_config(lgb_params, xgb_params, "depth=7"))
    print(f"  CV: {results[-1]['ensemble_cv']:.4f} ({results[-1]['improvement_pct']:+.2f}%)")

    # Test 5: Deeper trees (8)
    print("Testing deeper trees (depth=8)...")
    lgb_params = baseline_lgb.copy()
    xgb_params = baseline_xgb.copy()
    lgb_params["max_depth"] = 8
    xgb_params["max_depth"] = 8
    results.append(test_config(lgb_params, xgb_params, "depth=8"))
    print(f"  CV: {results[-1]['ensemble_cv']:.4f} ({results[-1]['improvement_pct']:+.2f}%)")

    # Test 6: Fewer estimators
    print("Testing fewer estimators (400)...")
    lgb_params = baseline_lgb.copy()
    xgb_params = baseline_xgb.copy()
    lgb_params["n_estimators"] = 400
    xgb_params["n_estimators"] = 400
    results.append(test_config(lgb_params, xgb_params, "n_est=400"))
    print(f"  CV: {results[-1]['ensemble_cv']:.4f} ({results[-1]['improvement_pct']:+.2f}%)")

    # Test 7: More estimators
    print("Testing more estimators (800)...")
    lgb_params = baseline_lgb.copy()
    xgb_params = baseline_xgb.copy()
    lgb_params["n_estimators"] = 800
    xgb_params["n_estimators"] = 800
    results.append(test_config(lgb_params, xgb_params, "n_est=800"))
    print(f"  CV: {results[-1]['ensemble_cv']:.4f} ({results[-1]['improvement_pct']:+.2f}%)")

    # Test 8: Even more estimators
    print("Testing even more estimators (1000)...")
    lgb_params = baseline_lgb.copy()
    xgb_params = baseline_xgb.copy()
    lgb_params["n_estimators"] = 1000
    xgb_params["n_estimators"] = 1000
    results.append(test_config(lgb_params, xgb_params, "n_est=1000"))
    print(f"  CV: {results[-1]['ensemble_cv']:.4f} ({results[-1]['improvement_pct']:+.2f}%)")

    # Test 9: Fewer leaves
    print("Testing fewer leaves (32)...")
    lgb_params = baseline_lgb.copy()
    xgb_params = baseline_xgb.copy()
    lgb_params["num_leaves"] = 32
    results.append(test_config(lgb_params, xgb_params, "leaves=32"))
    print(f"  CV: {results[-1]['ensemble_cv']:.4f} ({results[-1]['improvement_pct']:+.2f}%)")

    # Test 10: More leaves
    print("Testing more leaves (96)...")
    lgb_params = baseline_lgb.copy()
    xgb_params = baseline_xgb.copy()
    lgb_params["num_leaves"] = 96
    results.append(test_config(lgb_params, xgb_params, "leaves=96"))
    print(f"  CV: {results[-1]['ensemble_cv']:.4f} ({results[-1]['improvement_pct']:+.2f}%)")

    print()
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Baseline v67:  CV 0.3908 (0.00%)")
    print()

    # Sort by CV score
    results_sorted = sorted(results, key=lambda x: x["ensemble_cv"])

    for r in results_sorted:
        status = "✓" if r["ensemble_cv"] < 0.3908 else "✗"
        print(f"{status} {r['config']:15s}  CV: {r['ensemble_cv']:.4f}  ({r['improvement_pct']:+.2f}%)")

    print()
    best = results_sorted[0]
    if best["ensemble_cv"] < 0.3908:
        print(f"🎯 BEST: {best['config']} with CV {best['ensemble_cv']:.4f} ({best['improvement_pct']:+.2f}% improvement)")
    else:
        print(f"⚠️  No configuration improved over v67 baseline (CV 0.3908)")

    return results_sorted


if __name__ == "__main__":
    results = main()
