#!/usr/bin/env python3
"""XGBoost v6 - Simple Features + TimeSeriesSplit

Learning from past experiments:
1. Simple features beat complex (v1 > v2)
2. TimeSeriesSplit gives realistic CV (0.373 vs 0.321)
3. External data breaks LB

Strategy: Minimal feature set + proper validation
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
    print("Loading data...")
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
    return train, test


def create_simple_features(df, is_train=True, train_df=None):
    """Create SIMPLE feature set only - test if less is more."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Time index
    df["time_idx"] = df.groupby(["store_nbr", "family"], observed=True).cumcount()

    # Core seasonality (just the essentials)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if is_train:
        # Simple lags - weekly patterns only
        for lag in [7, 14, 28]:
            df[f"lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

        # Simple rolling - just 2 windows
        for window in [7, 28]:
            df[f"roll_mean_{window}"] = (
                df.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )

        # Promotion features
        df["promo_lag7"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(7).fillna(0)

    else:
        # Test set: use last 30 days mean from training
        if train_df is not None:
            last_30_stats = (
                train_df.groupby(["store_nbr", "family"], observed=True)
                .tail(30)
                .groupby(["store_nbr", "family"], observed=True)["sales"]
                .agg(["mean"])
                .reset_index()
            )
            last_30_stats.columns = ["store_nbr", "family", "lag_mean"]

            df = df.merge(last_30_stats, on=["store_nbr", "family"], how="left")

            # Fill lags with mean
            for lag in [7, 14, 28]:
                df[f"lag_{lag}"] = df["lag_mean"].fillna(0)

            # Fill rolling with mean
            for window in [7, 28]:
                df[f"roll_mean_{window}"] = df["lag_mean"].fillna(0)

            df["promo_lag7"] = df["onpromotion"].fillna(0)

    return df


def main():
    """Train with TimeSeriesSplit and simple features."""
    print("="*70)
    print("XGBoost v6 - Simple Features + TimeSeriesSplit")
    print("="*70)
    print("Testing: Fewer features + proper validation")
    print()

    # Load
    train_df, test_df = load_data()

    # Create features
    print("Creating simple features...")
    train_df = create_simple_features(train_df, is_train=True)
    test_df = create_simple_features(test_df, is_train=False, train_df=train_df)

    # Drop nulls
    train_df = train_df.dropna()

    print(f"\nAfter feature engineering:")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Features
    feature_cols = [
        "time_idx", "day_of_week", "month", "is_weekend",
        "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_mean_28",
        "onpromotion", "promo_lag7"
    ]

    X = train_df[feature_cols]
    y = train_df["sales"]
    X_test = test_df[feature_cols]

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
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

        # Train
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
        y_pred = np.clip(y_pred, 0, None)  # No negative sales

        # Score
        rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
        cv_scores.append(rmsle)
        print(f"  Fold {fold} RMSLE: {rmsle:.6f}")

    # CV summary
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    print("\n" + "="*70)
    print(f"TimeSeriesSplit CV Results (5 folds):")
    print(f"  Mean RMSLE: {mean_cv:.6f} (+/- {std_cv:.6f})")
    print(f"  All folds: {[f'{s:.6f}' for s in cv_scores]}")
    print("="*70)

    # Train final model on ALL data
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
    submission_path = SUBMISSION_DIR / f"xgb_v6_simple_{mean_cv:.4f}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\n✅ Submission saved to: {submission_path}")
    print(f"📊 TimeSeriesSplit CV: {mean_cv:.6f}")
    print(f"🎯 Expected LB: ~0.52-0.55 (based on CV/LB gap from v5)")

    # MLflow logging
    mlflow.set_experiment("store-sales-forecasting")
    with mlflow.start_run(run_name="xgb_v6_simple_features"):
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("features", "simple_only")
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("validation", "TimeSeriesSplit_5fold")
        mlflow.log_metric("cv_rmsle_mean", mean_cv)
        mlflow.log_metric("cv_rmsle_std", std_cv)
        for i, score in enumerate(cv_scores, 1):
            mlflow.log_metric(f"cv_fold_{i}", score)

    return mean_cv


if __name__ == "__main__":
    cv_score = main()
