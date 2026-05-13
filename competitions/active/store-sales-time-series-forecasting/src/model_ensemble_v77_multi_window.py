#!/usr/bin/env python3
"""Ensemble v77 - Multiple Validation Windows

Hypothesis: Average predictions from 3 models trained on different validation periods
to reduce dependence on any single temporal window.

Windows:
- May 16 - Jun 15 (30 days before Jul 16)
- Jun 16 - Jul 15 (30 days before Aug 15)
- Jul 16 - Aug 15 (current validation window, used by v67)

Expected: More robust to temporal distribution shift
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
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


def train_model_with_window(train_df, feature_cols, val_start, val_end, window_name):
    """Train a model with specific validation window"""
    print(f"\n{window_name}: {val_start} to {val_end}")

    val_mask = (train_df["date"] >= val_start) & (train_df["date"] <= val_end)
    train_mask = ~val_mask

    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, "sales"]
    X_val = train_df.loc[val_mask, feature_cols]
    y_val = train_df.loc[val_mask, "sales"]

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"  Val mean: {y_val.mean():.2f}")

    # Train LightGBM
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
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    # Train XGBoost
    xgb_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 600,
        "random_state": 42
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)

    # Validation CV
    lgb_val = np.maximum(lgb_model.predict(X_val), 0)
    xgb_val = np.maximum(xgb_model.predict(X_val), 0)

    # Get intercept from Ridge
    meta_train = np.column_stack([lgb_val, xgb_val])
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(meta_train, y_val)
    intercept = ridge.intercept_

    val_preds = 0.90 * lgb_val + 0.10 * xgb_val + intercept
    val_preds = np.maximum(val_preds, 0)
    cv = np.sqrt(mean_squared_log_error(y_val, val_preds))

    print(f"  CV: {cv:.4f}, Intercept: {intercept:.4f}")

    return lgb_model, xgb_model, intercept


def main():
    print("="*70)
    print("Ensemble v77 - Multiple Validation Windows")
    print("="*70)

    train_df, test_df, holidays = load_data()
    train_df = create_features(train_df)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    # Test features - CRITICAL: sort first
    test_df = test_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    test_df["sales"] = np.nan
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = create_features(combined)
    combined = add_holidays(combined, holidays)
    test_df = combined[combined["id"].notna()].copy()

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

    # Train 3 models with different validation windows
    max_date = train_df["date"].max()

    windows = [
        ("2017-07-16", max_date, "Window 3 (Jul 16-Aug 15)"),
        ("2017-06-16", "2017-07-15", "Window 2 (Jun 16-Jul 15)"),
        ("2017-05-16", "2017-06-15", "Window 1 (May 16-Jun 15)"),
    ]

    test_predictions = []
    X_test = test_df[feature_cols]

    for val_start, val_end, window_name in windows:
        lgb_model, xgb_model, intercept = train_model_with_window(
            train_df, feature_cols, val_start, val_end, window_name
        )

        # Test predictions
        lgb_test = np.maximum(lgb_model.predict(X_test), 0)
        xgb_test = np.maximum(xgb_model.predict(X_test), 0)
        test_preds = 0.90 * lgb_test + 0.10 * xgb_test + intercept
        test_preds = np.maximum(test_preds, 0)
        test_predictions.append(test_preds)

        print(f"  Test mean: {test_preds.mean():.2f}")

    # Average all 3 models
    ensemble_preds = np.mean(test_predictions, axis=0)

    print("\nEnsemble (average of 3 windows):")
    print(f"  Mean: {ensemble_preds.mean():.2f}")
    print(f"  Std: {ensemble_preds.std():.2f}")

    # Submission
    submission = pd.DataFrame({
        "id": test_df["id"].astype(int),
        "sales": ensemble_preds
    })

    submission_path = SUBMISSION_DIR / "ensemble_v77_multi_window_noCV.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSaved: {submission_path}")
    print("✓ v77 complete")


if __name__ == "__main__":
    main()
