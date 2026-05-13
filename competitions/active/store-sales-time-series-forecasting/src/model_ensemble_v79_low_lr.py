#!/usr/bin/env python3
"""Ensemble v79 - Lower Learning Rate

Hypothesis: Lower learning rate (0.01 vs 0.05) with more estimators (1200 vs 600)
produces smoother predictions that generalize better.

Expected: Less validation-specific overfitting, more conservative predictions
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


def main():
    print("="*70)
    print("Ensemble v79 - Lower Learning Rate (0.01)")
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

    # 30-day holdout (same as v67)
    cutoff_date = train_df["date"].max() - pd.Timedelta(days=30)
    val_mask = train_df["date"] >= cutoff_date
    train_mask = ~val_mask

    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, "sales"]
    X_val = train_df.loc[val_mask, feature_cols]
    y_val = train_df.loc[val_mask, "sales"]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    print()

    # Train LightGBM with LOWER learning rate, MORE estimators
    print("Training LightGBM (lr=0.01, n_est=1200)...")
    lgb_params = {
        "objective": "regression",
        "learning_rate": 0.01,  # 5x lower
        "num_leaves": 64,
        "max_depth": 6,
        "n_estimators": 1200,  # 2x more
        "random_state": 42,
        "verbosity": -1
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

    # Train XGBoost with LOWER learning rate
    print("Training XGBoost (lr=0.01, n_est=1200)...")
    xgb_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.01,  # 5x lower
        "max_depth": 6,
        "n_estimators": 1200,  # 2x more
        "random_state": 42
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)

    # Validation
    lgb_val = np.maximum(lgb_model.predict(X_val), 0)
    xgb_val = np.maximum(xgb_model.predict(X_val), 0)

    # Ridge for intercept
    meta_train = np.column_stack([lgb_val, xgb_val])
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(meta_train, y_val)
    intercept = ridge.intercept_

    val_preds = 0.90 * lgb_val + 0.10 * xgb_val + intercept
    val_preds = np.maximum(val_preds, 0)
    cv = np.sqrt(mean_squared_log_error(y_val, val_preds))

    print(f"\nCV: {cv:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print()

    # Test predictions
    X_test = test_df[feature_cols]
    lgb_test = np.maximum(lgb_model.predict(X_test), 0)
    xgb_test = np.maximum(xgb_model.predict(X_test), 0)
    test_preds = 0.90 * lgb_test + 0.10 * xgb_test + intercept
    test_preds = np.maximum(test_preds, 0)

    print("Test predictions:")
    print(f"  Mean: {test_preds.mean():.2f}")
    print(f"  Std: {test_preds.std():.2f}")

    # Submission
    submission = pd.DataFrame({
        "id": test_df["id"].astype(int),
        "sales": test_preds
    })

    cv_str = f"{cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"ensemble_v79_low_lr_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSaved: {submission_path}")
    print("✓ v79 complete")


if __name__ == "__main__":
    main()
