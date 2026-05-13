#!/usr/bin/env python3
"""Ensemble v73 - Tuned Hyperparameters

Based on v67 Ridge 90/10 with optimized hyperparameters from tuning experiments.

v67 baseline: CV 0.3908, LB 0.45416
v73 tuned: Expected CV 0.3757 (3.86% improvement)

Key changes from v67:
- max_depth: 6 → 7
- num_leaves: 64 → 96
- Other params unchanged (lr=0.05, n_est=600)

Tuning results:
- Tested 15 single-parameter variations
- Tested 5 combinations
- Best: depth=7 + leaves=96 (CV 0.3757)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb

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
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
    )
    test = pd.read_csv(
        DATA_DIR / "test.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
    )
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    return train, test, holidays


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


def main():
    print("="*70)
    print("Ensemble v73 - Tuned Hyperparameters (depth=7, leaves=96)")
    print("="*70)
    print("Based on v67 Ridge 90/10 with optimized hyperparameters")
    print("v67 baseline: CV 0.3908, LB 0.45416")
    print("v73 expected: CV 0.3757 (3.86% improvement)")
    print()

    train_df, test_df, holidays = load_data()
    train_df = create_features(train_df)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    # Test features
    test_df["sales"] = np.nan
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = create_features(combined)
    combined = add_holidays(combined, holidays)
    test_df = combined[combined["id"].isin(test_df["id"])].copy()

    lag_cols = ["Lag_3", "Lag_7"]
    for col in lag_cols:
        test_df[col] = test_df.groupby(["store_nbr", "family"], observed=True)[col].fillna(method="ffill")
        test_df[col] = test_df[col].fillna(0)

    feature_cols = [
        "onpromotion", "day_of_week", "is_weekend", "is_holiday",
        "Lag_3", "Lag_7",
        "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "Roll_mean_60", "Roll_mean_90",
        "Roll_std_7"
    ]

    # 30-day temporal holdout (same as v67)
    cutoff_date = train_df["date"].max() - pd.Timedelta(days=30)
    val_mask = train_df["date"] >= cutoff_date
    train_mask = ~val_mask

    X = train_df[feature_cols]
    y = train_df["sales"]
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    print(f"Train: {len(X_train):,} samples")
    print(f"Val: {len(X_val):,} samples (last 30 days)")
    print()

    # Train models with TUNED hyperparameters
    print("Training LightGBM (tuned: depth=7, leaves=96)...")
    lgb_params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 96,  # TUNED: was 64
        "max_depth": 7,     # TUNED: was 6
        "n_estimators": 600,
        "random_state": 42,
        "verbosity": -1
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    lgb_val = np.maximum(lgb_model.predict(X_val), 0)
    lgb_cv = np.sqrt(mean_squared_log_error(y_val, lgb_val))
    print(f"LightGBM CV: {lgb_cv:.4f}")

    print("Training XGBoost (tuned: depth=7)...")
    xgb_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 7,  # TUNED: was 6
        "n_estimators": 600,
        "random_state": 42
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)

    xgb_val = np.maximum(xgb_model.predict(X_val), 0)
    xgb_cv = np.sqrt(mean_squared_log_error(y_val, xgb_val))
    print(f"XGBoost CV: {xgb_cv:.4f}")
    print()

    # Ridge 90/10 with intercept (same as v67)
    print("Applying Ridge 90/10 ensemble...")
    val_preds = 0.90 * lgb_val + 0.10 * xgb_val

    ridge = Ridge(alpha=1.0, fit_intercept=True)
    meta_train = np.column_stack([lgb_val, xgb_val])
    ridge.fit(meta_train, y_val)
    intercept = ridge.intercept_

    val_preds = val_preds + intercept
    val_preds = np.maximum(val_preds, 0)

    ensemble_cv = np.sqrt(mean_squared_log_error(y_val, val_preds))

    print("Results:")
    print(f"  Weights: 0.90 LGB + 0.10 XGB + {intercept:.4f}")
    print(f"  CV: {ensemble_cv:.4f}")
    print(f"  v67 baseline: 0.3908")
    print(f"  Improvement: {((0.3908 - ensemble_cv) / 0.3908 * 100):+.2f}%")
    print()

    if ensemble_cv < 0.3908:
        print(f"✓ Tuning successful: {((0.3908 - ensemble_cv) / 0.3908 * 100):.2f}% better than v67")
    else:
        print(f"⚠️  Tuning did not improve CV (expected 0.3757, got {ensemble_cv:.4f})")

    print()

    # Train on full data
    print("Training on full dataset...")
    lgb_model_full = lgb.LGBMRegressor(**lgb_params)
    lgb_model_full.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    xgb_model_full = xgb.XGBRegressor(**xgb_params)
    xgb_model_full.fit(X, y, verbose=False)

    X_test = test_df[feature_cols]
    lgb_test = np.maximum(lgb_model_full.predict(X_test), 0)
    xgb_test = np.maximum(xgb_model_full.predict(X_test), 0)

    test_preds = 0.90 * lgb_test + 0.10 * xgb_test + intercept
    test_preds = np.maximum(test_preds, 0)

    # Sort test_df to match prediction order (fix v63 alignment bug)
    test_df_sorted = test_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    submission = pd.DataFrame({
        "id": test_df_sorted["id"],
        "sales": test_preds
    })

    cv_str = f"{ensemble_cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"ensemble_v73_tuned_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # Hypothesis DB
    if HAS_DB:
        try:
            db = HypothesisDatabase()
            hyp_id = db.add_hypothesis(
                competition="store-sales-time-series-forecasting",
                experiment_id="v73",
                hypothesis="Tuned hyperparameters (depth=7, leaves=96) improve v67 Ridge 90/10 ensemble",
                rationale="Hyperparameter tuning found depth=7+leaves=96 gives CV 0.3757 (3.86% better than v67's 0.3908). Tested 15 single variations + 5 combinations.",
                category="ensemble_methods"
            )
            db.record_result(
                hypothesis_id=hyp_id,
                cv_score=ensemble_cv,
                lb_score=None,
                baseline_cv=0.3908,
                succeeded=(ensemble_cv < 0.39),
                params={
                    'lgb_weight': 0.9,
                    'xgb_weight': 0.1,
                    'intercept': float(intercept),
                    'lgb_depth': 7,
                    'lgb_leaves': 96,
                    'xgb_depth': 7
                },
                metadata={
                    'tuning_method': 'systematic_grid',
                    'configs_tested': 20,
                    'best_single': 'depth=7 (CV 0.3792)',
                    'best_combo': 'depth=7+leaves=96 (CV 0.3757)',
                    'v67_lb': 0.45416
                }
            )
        except Exception as e:
            print(f"Note: Could not log: {e}")

    return {
        "cv": ensemble_cv,
        "vs_v67": ensemble_cv - 0.3908,
        "intercept": intercept,
        "params": {"depth": 7, "leaves": 96}
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
