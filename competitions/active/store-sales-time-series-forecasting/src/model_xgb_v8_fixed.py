#!/usr/bin/env python3
"""XGBoost v8 - v1 Features + TimeSeriesSplit Validation

Strategy: Take v1's PROVEN feature set (got LB 0.526) but validate properly
with TimeSeriesSplit instead of simple train/test split.

This gives us:
- Realistic CV scores (not overly optimistic like v1's 0.321)
- Proven features that work on LB
- Better prediction of actual LB performance
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


def create_features(df, is_train=True, train_df=None):
    """Create v1's full feature set."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    df["Time"] = df.groupby(["store_nbr", "family"], observed=True).cumcount()

    # Full seasonality (v1's features)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    if is_train:
        # All lags from v1
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

        # All rolling windows from v1
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
    else:
        # Test set: use v1's simple mean-fill approach (proven to work)
        if train_df is not None:
            last_30_stats = (
                train_df.groupby(["store_nbr", "family"], observed=True)
                .tail(30)
                .groupby(["store_nbr", "family"], observed=True)["sales"]
                .agg(["mean", "std"])
                .reset_index()
            )
            last_30_stats.columns = ["store_nbr", "family", "lag_mean", "lag_std"]

            df = df.merge(last_30_stats, on=["store_nbr", "family"], how="left")

            for lag in [1, 2, 3, 7, 14, 21, 28]:
                df[f"Lag_{lag}"] = df["lag_mean"].fillna(0)

            for window in [7, 14, 30, 60, 90]:
                df[f"Roll_mean_{window}"] = df["lag_mean"].fillna(0)
                df[f"Roll_std_{window}"] = df["lag_std"].fillna(0)

            df["onpromotion_lag1"] = df["onpromotion"]
            df["onpromotion_lag7"] = df["onpromotion"]

    return df


def add_holidays(df, holidays):
    """Add holiday indicator."""
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    """Train with v1 features but TimeSeriesSplit validation."""
    print("="*70)
    print("XGBoost v8 - v1 Features + Proper Validation")
    print("="*70)
    print("Using proven v1 feature set with realistic TimeSeriesSplit CV")
    print()

    # Load
    train_df, test_df, holidays = load_data()

    # Create features
    print("Creating v1's full feature set...")
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)

    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)

    # Drop nulls
    train_df = train_df.dropna()

    print(f"\nAfter feature engineering:")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Features (same as v1)
    feature_cols = [
        "Time", "day_of_week", "day_of_month", "month", "week_of_year",
        "is_month_end", "is_month_start", "is_weekend",
        "Lag_1", "Lag_2", "Lag_3", "Lag_7", "Lag_14", "Lag_21", "Lag_28",
        "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "Roll_mean_60", "Roll_mean_90",
        "Roll_std_7", "Roll_std_14", "Roll_std_30", "Roll_std_60", "Roll_std_90",
        "onpromotion", "onpromotion_lag1", "onpromotion_lag7", "is_holiday"
    ]

    X = train_df[feature_cols]
    y = train_df["sales"]
    X_test = test_df[feature_cols]

    print(f"\nFeatures ({len(feature_cols)}): Same as v1 (proven LB 0.526)")
    print(f"Samples: {len(X):,}")

    # TimeSeriesSplit validation
    print("\n" + "="*70)
    print("Training with 5-fold TimeSeriesSplit CV...")
    print("="*70)

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"\nFold {fold}/5")
        print(f"  Train: {len(train_idx):,} samples")
        print(f"  Val: {len(val_idx):,} samples")

        # Train (same params as v1)
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            early_stopping_rounds=50,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Predict
        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0, None)

        # Score
        rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
        cv_scores.append(rmsle)
        print(f"  Fold {fold} RMSLE: {rmsle:.6f}")

    # CV summary
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    print("\n" + "="*70)
    print(f"TimeSeriesSplit CV Results:")
    print(f"  Mean RMSLE: {mean_cv:.6f} (+/- {std_cv:.6f})")
    print(f"  All folds: {[f'{s:.6f}' for s in cv_scores]}")
    print("="*70)

    # Compare
    print("\nComparison:")
    print(f"  v1 (simple split): CV 0.321, LB 0.526")
    print(f"  v5 (TimeSeriesSplit, 29 features): CV 0.373")
    print(f"  v8 (TimeSeriesSplit, v1 features): CV {mean_cv:.3f}")

    if mean_cv < 0.373:
        improvement = ((0.373 - mean_cv) / 0.373) * 100
        print(f"\n✅ IMPROVED! {improvement:.1f}% better than v5")
        print("This is our new best - ready to submit!")
    else:
        print(f"\n⚠️ Not better than v5 ({mean_cv:.3f} vs 0.373)")

    # Train final model
    print("\nTraining final model on full dataset...")
    final_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
    )

    final_model.fit(X, y, verbose=False)

    # Generate submission
    print("Generating predictions...")
    test_preds = final_model.predict(X_test)
    test_preds = np.clip(test_preds, 0, None)

    # Save
    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": test_preds
    })

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / f"xgb_v8_v1features_{mean_cv:.4f}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\n✅ Submission saved: {submission_path.name}")
    print(f"📊 CV: {mean_cv:.6f} (TimeSeriesSplit)")
    print(f"🎯 Expected LB: ~{mean_cv + 0.15:.3f} (based on v5's CV→LB gap)")

    # MLflow
    mlflow.set_experiment("store-sales-forecasting")
    with mlflow.start_run(run_name="xgb_v8_v1_features_tscv"):
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("features", "v1_full_set")
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("validation", "TimeSeriesSplit_5fold")
        mlflow.log_metric("cv_rmsle_mean", mean_cv)
        mlflow.log_metric("cv_rmsle_std", std_cv)
        for i, score in enumerate(cv_scores, 1):
            mlflow.log_metric(f"cv_fold_{i}", score)

    return mean_cv


if __name__ == "__main__":
    cv_score = main()
