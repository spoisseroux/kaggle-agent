#!/usr/bin/env python3
"""XGBoost v5 - TimeSeriesSplit cross-validation

Implements proper time series CV instead of simple train/test split.
This should give more realistic CV scores that match LB better.

Research shows winners use TimeSeriesSplit, not simple splits.
Current gap: CV 0.321 vs LB 0.526 likely due to validation strategy.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import mlflow

DATA_DIR = Path("data/store-sales-time-series-forecasting")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")


def load_data():
    """Load datasets."""
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


def create_features(df, holidays):
    """Create all v1 features."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    df["Time"] = df.groupby(["store_nbr", "family"], observed=True).cumcount()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

    for window in [7, 14, 30, 60, 90]:
        df[f"Roll_mean_{window}"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        df[f"Roll_std_{window}"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )

    df["onpromotion_lag1"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(1).fillna(0)
    df["onpromotion_lag7"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(7).fillna(0)

    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)

    return df


def train_with_timeseries_cv(X, y, n_splits=5):
    """Train with TimeSeriesSplit cross-validation."""
    print(f"\nTraining with {n_splits}-fold TimeSeriesSplit CV...")

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
        print(f"\nFold {fold}/{n_splits}")

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        print(f"  Train: {len(train_idx):,} samples")
        print(f"  Val: {len(val_idx):,} samples")

        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        y_pred = model.predict(dval)
        y_pred = np.maximum(y_pred, 0)

        mask = y_val_fold > 0
        fold_rmsle = np.sqrt(mean_squared_log_error(y_val_fold[mask], y_pred[mask]))
        cv_scores.append(fold_rmsle)

        print(f"  Fold {fold} RMSLE: {fold_rmsle:.6f}")

    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    print(f"\n{'='*60}")
    print(f"TimeSeriesSplit CV Results ({n_splits} folds):")
    print(f"  Mean RMSLE: {mean_cv:.6f} (+/- {std_cv:.6f})")
    print(f"  All folds: {[f'{s:.6f}' for s in cv_scores]}")
    print(f"{'='*60}")

    return mean_cv, std_cv, cv_scores


def main():
    print("=" * 70)
    print("XGBoost v5 - TimeSeriesSplit Cross-Validation")
    print("=" * 70)
    print("Testing proper time series CV vs simple train/test split")
    print("Expected: More realistic CV score closer to LB")
    print()

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    # Create features
    print("Creating features...")
    train_df = create_features(train_df, holidays)
    train_df = train_df.dropna()

    print(f"Train shape: {train_df.shape}")

    # Prepare features
    feature_cols = [c for c in train_df.columns if c not in ['sales', 'date', 'store_nbr', 'family', 'id']]
    X = train_df[feature_cols]
    y = train_df['sales']

    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X):,}")
    print()

    # TimeSeriesSplit CV
    mlflow.set_experiment("store-sales-xgboost")
    with mlflow.start_run(run_name="xgb_v5_timeseries_cv"):
        mean_cv, std_cv, fold_scores = train_with_timeseries_cv(X, y, n_splits=5)

        # Log to MLflow
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("validation", "TimeSeriesSplit")
        mlflow.log_param("n_folds", 5)
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_metric("cv_rmsle_mean", mean_cv)
        mlflow.log_metric("cv_rmsle_std", std_cv)
        for i, score in enumerate(fold_scores, 1):
            mlflow.log_metric(f"cv_fold_{i}", score)

        # Compare to baseline
        baseline_cv = 0.321032
        print(f"\nComparison:")
        print(f"  Simple split CV (v1):     {baseline_cv:.6f}")
        print(f"  TimeSeriesSplit CV (v5):  {mean_cv:.6f} (+/- {std_cv:.6f})")
        print(f"  v1 LB score:              0.52575")
        print()

        if mean_cv > baseline_cv:
            diff = mean_cv - baseline_cv
            print(f"  TimeSeriesSplit CV is {diff:.6f} higher (more realistic)")
            print(f"  This suggests v1's simple split was overly optimistic")
        else:
            diff = baseline_cv - mean_cv
            print(f"  TimeSeriesSplit CV is {diff:.6f} lower")

        print()
        print("✅ TimeSeriesSplit CV complete")
        print(f"   Mean CV: {mean_cv:.6f}")
        print(f"   This should better predict actual LB performance")

        return mean_cv


if __name__ == "__main__":
    cv = main()
