#!/usr/bin/env python3
"""Test feature selection with TimeSeriesSplit CV

Previous test with simple split showed feature reduction hurt performance.
Re-test with proper time series CV to see if results change.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

DATA_DIR = Path("data/store-sales-time-series-forecasting")

# Reduced feature set (19 features)
REDUCED_FEATURES = [
    'Time', 'day_of_week', 'month', 'is_weekend',
    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 'Lag_14', 'Lag_28',
    'Roll_mean_7', 'Roll_mean_14', 'Roll_mean_30',
    'Roll_std_7', 'Roll_std_14', 'Roll_std_30',
    'onpromotion', 'onpromotion_lag1', 'is_holiday'
]

# Full feature set (29 features)
FULL_FEATURES = [
    'Time', 'day_of_week', 'day_of_month', 'month', 'week_of_year',
    'is_month_end', 'is_month_start', 'is_weekend',
    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 'Lag_14', 'Lag_21', 'Lag_28',
    'Roll_mean_7', 'Roll_mean_14', 'Roll_mean_30', 'Roll_mean_60', 'Roll_mean_90',
    'Roll_std_7', 'Roll_std_14', 'Roll_std_30', 'Roll_std_60', 'Roll_std_90',
    'onpromotion', 'onpromotion_lag1', 'onpromotion_lag7', 'is_holiday'
]


def load_and_prepare_data():
    """Load and create features."""
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
    )
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])

    # Feature engineering
    train = train.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    train["Time"] = train.groupby(["store_nbr", "family"], observed=True).cumcount()
    train["day_of_week"] = train["date"].dt.dayofweek
    train["day_of_month"] = train["date"].dt.day
    train["month"] = train["date"].dt.month
    train["is_month_end"] = train["date"].dt.is_month_end.astype(int)
    train["is_month_start"] = train["date"].dt.is_month_start.astype(int)
    train["is_weekend"] = (train["day_of_week"] >= 5).astype(int)
    train["week_of_year"] = train["date"].dt.isocalendar().week.astype(int)

    for lag in [1, 2, 3, 7, 14, 21, 28]:
        train[f"Lag_{lag}"] = train.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

    for window in [7, 14, 30, 60, 90]:
        train[f"Roll_mean_{window}"] = (
            train.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        train[f"Roll_std_{window}"] = (
            train.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )

    train["onpromotion_lag1"] = train.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(1).fillna(0)
    train["onpromotion_lag7"] = train.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(7).fillna(0)

    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    train["is_holiday"] = train["date"].isin(national_holidays).astype(int)

    train = train.dropna()
    return train


def evaluate_with_tscv(X, y, name, n_splits=3):
    """Evaluate with TimeSeriesSplit CV."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        y_pred = model.predict(dval)
        y_pred = np.maximum(y_pred, 0)

        mask = y_val > 0
        fold_rmsle = np.sqrt(mean_squared_log_error(y_val[mask], y_pred[mask]))
        cv_scores.append(fold_rmsle)

    mean_cv = np.mean(cv_scores)
    return mean_cv, cv_scores


def main():
    print("=" * 70)
    print("Feature Selection with TimeSeriesSplit CV")
    print("=" * 70)
    print()

    # Load data
    print("Loading and preparing data...")
    train_df = load_and_prepare_data()
    print(f"Data shape: {train_df.shape}")
    print()

    # Test full features
    print("Testing FULL feature set (29 features)...")
    X_full = train_df[FULL_FEATURES]
    y = train_df['sales']
    full_cv, full_scores = evaluate_with_tscv(X_full, y, "full", n_splits=3)
    print(f"  Mean CV: {full_cv:.6f}")
    print(f"  Folds: {[f'{s:.4f}' for s in full_scores]}")
    print()

    # Test reduced features
    print("Testing REDUCED feature set (19 features)...")
    X_reduced = train_df[REDUCED_FEATURES]
    reduced_cv, reduced_scores = evaluate_with_tscv(X_reduced, y, "reduced", n_splits=3)
    print(f"  Mean CV: {reduced_cv:.6f}")
    print(f"  Folds: {[f'{s:.4f}' for s in reduced_scores]}")
    print()

    # Compare
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Full features (29):    {full_cv:.6f}")
    print(f"Reduced features (19): {reduced_cv:.6f}")

    diff = reduced_cv - full_cv
    if diff < 0:
        print(f"Improvement:           {-diff:.6f} ✅")
        print("Feature reduction helps with proper CV!")
    else:
        print(f"Degradation:           +{diff:.6f}")
        print("Full feature set still better")

    print()
    print(f"Previous test (simple split):")
    print(f"  Full: 0.321, Reduced: 0.350 (full better)")
    print(f"New test (TimeSeriesSplit):")
    print(f"  Full: {full_cv:.3f}, Reduced: {reduced_cv:.3f}")

    return full_cv, reduced_cv


if __name__ == "__main__":
    full, reduced = main()
