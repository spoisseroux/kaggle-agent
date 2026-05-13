#!/usr/bin/env python3
"""Ensemble v72 - Ridge 85/15 Split

Testing if train-val split ratio affects performance.

v67 used 30-day holdout (~2% of data for validation)
v72 uses 15% validation (~90/10 becomes 85/15 for easier calculation)

Hypothesis: Larger validation set provides better weight estimation
Expected: Similar or slightly better than v67
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
    print("Ensemble v72 - Ridge 85/15 Split")
    print("="*70)
    print("Testing larger validation set (15% vs v67's 2%)")
    print()

    train_df, test_df, holidays = load_data()
    train_df = create_features(train_df)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

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

    # 85/15 split
    split_idx = int(len(train_df) * 0.85)
    train_mask = np.arange(len(train_df)) < split_idx
    val_mask = ~train_mask

    X = train_df[feature_cols]
    y = train_df["sales"]
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    print(f"Train: {len(X_train):,} (85%)")
    print(f"Val: {len(X_val):,} (15%)")
    print()

    # Train models
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        objective="regression", learning_rate=0.05, num_leaves=64,
        max_depth=6, n_estimators=600, random_state=42, verbosity=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    lgb_val = np.maximum(lgb_model.predict(X_val), 0)
    lgb_cv = np.sqrt(mean_squared_log_error(y_val, lgb_val))
    print(f"LightGBM CV: {lgb_cv:.4f}")

    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror", learning_rate=0.05,
        max_depth=6, n_estimators=600, random_state=42
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  verbose=False, early_stopping_rounds=50)

    xgb_val = np.maximum(xgb_model.predict(X_val), 0)
    xgb_cv = np.sqrt(mean_squared_log_error(y_val, xgb_val))
    print(f"XGBoost CV: {xgb_cv:.4f}")
    print()

    # Ridge with 90/10 manual weights
    print("Applying 90/10 weights...")
    val_preds = 0.90 * lgb_val + 0.10 * xgb_val

    # Learn intercept
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    meta_train = np.column_stack([lgb_val, xgb_val])
    ridge.fit(meta_train, y_val)
    intercept = ridge.intercept_

    val_preds = val_preds + intercept
    val_preds = np.maximum(val_preds, 0)

    ensemble_cv = np.sqrt(mean_squared_log_error(y_val, val_preds))

    print(f"Results:")
    print(f"  Weights: 0.90 LGB + 0.10 XGB + {intercept:.4f}")
    print(f"  CV: {ensemble_cv:.4f}")
    print(f"  v67 (30-day): 0.3908")
    print(f"  Difference: {(ensemble_cv - 0.3908):.4f}")
    print()

    # Train on full data
    lgb_model_full = lgb.LGBMRegressor(
        objective="regression", learning_rate=0.05, num_leaves=64,
        max_depth=6, n_estimators=600, random_state=42, verbosity=-1
    )
    lgb_model_full.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    xgb_model_full = xgb.XGBRegressor(
        objective="reg:squarederror", learning_rate=0.05,
        max_depth=6, n_estimators=600, random_state=42
    )
    xgb_model_full.fit(X, y, verbose=False)

    X_test = test_df[feature_cols]
    lgb_test = np.maximum(lgb_model_full.predict(X_test), 0)
    xgb_test = np.maximum(xgb_model_full.predict(X_test), 0)

    test_preds = 0.90 * lgb_test + 0.10 * xgb_test + intercept
    test_preds = np.maximum(test_preds, 0)

    submission = pd.DataFrame({"id": test_df["id"], "sales": test_preds})
    cv_str = f"{ensemble_cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"ensemble_v72_ridge_85_15_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    if HAS_DB:
        try:
            db = HypothesisDatabase()
            hyp_id = db.add_hypothesis(
                competition="store-sales-time-series-forecasting",
                experiment_id="v72",
                hypothesis="85/15 train split improves weight estimation",
                rationale="Larger validation set (15% vs 2%) may provide better Ridge weight learning.",
                category="ensemble_methods"
            )
            db.record_result(
                hypothesis_id=hyp_id,
                cv_score=ensemble_cv,
                lb_score=None,
                baseline_cv=0.3908,
                succeeded=(ensemble_cv < 0.39),
                params={'lgb_weight': 0.9, 'xgb_weight': 0.1, 'intercept': float(intercept)},
                metadata={'train_split': 0.85, 'val_split': 0.15}
            )
        except Exception as e:
            print(f"Note: Could not log: {e}")

    return {
        "cv": ensemble_cv,
        "vs_v67": ensemble_cv - 0.3908,
        "intercept": intercept,
        "split": "85/15"
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
