#!/usr/bin/env python3
"""Ensemble v76 - Train on ALL data (no validation holdout)

Hypothesis: Validation holdout (last 30 days) is a seasonal anomaly.
Training on full dataset avoids optimizing for this peak period.

Expected: More generalizable predictions, lower mean (closer to 356 than 472)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

DATA_DIR = Path("data/store-sales-time-series-forecasting")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")
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
    print("Ensemble v76 - Train on ALL data (no validation holdout)")
    print("="*70)
    print("Hypothesis: Avoid optimizing for validation period anomaly")
    print()

    train_df, test_df, holidays = load_data()
    train_df = create_features(train_df)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    # Test features - CRITICAL: sort first for alignment
    test_df = test_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    test_df["sales"] = np.nan
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = create_features(combined)
    combined = add_holidays(combined, holidays)
    test_df = combined[combined["id"].notna()].copy()

    # Fill NaN lags
    lag_cols = ["Lag_3", "Lag_7"]
    for col in lag_cols:
        test_df[col] = test_df.groupby(["store_nbr", "family"], observed=True)[col].ffill()
        test_df[col] = test_df[col].fillna(0)

    feature_cols = [
        "onpromotion", "day_of_week", "is_weekend", "is_holiday",
        "Lag_3", "Lag_7",
        "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "Roll_mean_60", "Roll_mean_90",
        "Roll_std_7"
    ]

    # NO VALIDATION HOLDOUT - use all data
    X = train_df[feature_cols]
    y = train_df["sales"]

    print(f"Training on ALL data: {len(X):,} samples")
    print(f"  Training mean: {y.mean():.2f}")
    print(f"  Training std: {y.std():.2f}")
    print()

    # Train LightGBM (same params as v67)
    print("Training LightGBM...")
    lgb_params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 6,
        "n_estimators": 600,
        "random_state": 42,
        "verbosity": -1
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    # Train XGBoost (same params as v67)
    print("Training XGBoost...")
    xgb_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 600,
        "random_state": 42
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X, y, verbose=False)

    # Test predictions
    X_test = test_df[feature_cols]
    lgb_test = np.maximum(lgb_model.predict(X_test), 0)
    xgb_test = np.maximum(xgb_model.predict(X_test), 0)

    # Ridge 90/10 ensemble (intercept from v67)
    intercept = -0.4937  # Use v67's intercept
    test_preds = 0.90 * lgb_test + 0.10 * xgb_test + intercept
    test_preds = np.maximum(test_preds, 0)

    print()
    print("Test predictions:")
    print(f"  Mean: {test_preds.mean():.2f}")
    print(f"  Std: {test_preds.std():.2f}")
    print(f"  Min: {test_preds.min():.2f}")
    print(f"  Max: {test_preds.max():.2f}")
    print()

    # Submission (test_df already sorted)
    submission = pd.DataFrame({
        "id": test_df["id"].astype(int),
        "sales": test_preds
    })

    submission_path = SUBMISSION_DIR / "ensemble_v76_full_data_noCV.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")
    print("✓ v76 complete")

    return test_preds.mean()


if __name__ == "__main__":
    mean_pred = main()
