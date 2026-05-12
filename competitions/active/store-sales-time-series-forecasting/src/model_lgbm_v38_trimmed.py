#!/usr/bin/env python3
"""LightGBM v38 - Trimmed Features (v19 minus weak correlations)

STRATEGY: Remove weak features to reduce overfitting
- v19 baseline: CV 0.3572 → LB 0.49833 (BEST)
- Remove 3 weak features: is_weekend, day_of_week, is_holiday
- All 3 have |correlation| < 0.06 with sales
- Hypothesis: Temporal noise hurts generalization

Features (9 total - v19's strong features only):
- Lags: 3, 7
- Rolling means: 7, 14, 30, 60, 90
- Rolling std: 7
- Promotion: onpromotion

Removed (weak |corr| < 0.1):
- day_of_week (corr: 0.037)
- is_weekend (corr: 0.052)
- is_holiday (corr: 0.016)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import mlflow

# Add parent directory to path for core imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from core.langfuse_logger import log_kaggle_model
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False

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


def create_features(df, is_train=True, train_df=None):
    """v19 minus weak temporal features (day_of_week, is_weekend, is_holiday)."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    if is_train:
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
    else:
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

            df["Lag_3"] = df["lag_mean"].fillna(0)
            df["Lag_7"] = df["lag_mean"].fillna(0)
            for window in [7, 14, 30, 60, 90]:
                df[f"Roll_mean_{window}"] = df["lag_mean"].fillna(0)
            df["Roll_std_7"] = df["lag_std"].fillna(0)

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    print("="*70)
    print("LightGBM v38 - Trimmed Features (v19 minus weak correlations)")
    print("="*70)
    print("Removing temporal noise to improve generalization")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-lightgbm-v1")

    # Load data
    print("Loading data...")
    train_df, test_df, _ = load_data()

    # Create features (v19 minus weak features)
    print("Creating trimmed features...")
    train_df = create_features(train_df, is_train=True)
    train_df = train_df.dropna()

    test_df = create_features(test_df, is_train=False, train_df=train_df)

    # Feature columns
    exclude_cols = ["id", "date", "sales", "store_nbr", "family"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"Features: {len(feature_cols)} (v19 strong features only)")
    print()

    # 30-day holdout
    cutoff_date = train_df['date'].max() - pd.Timedelta(days=30)
    val_mask = train_df['date'] >= cutoff_date
    train_mask = ~val_mask

    X = train_df[feature_cols]
    y = train_df["sales"]
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    print()

    # LightGBM parameters (ANTI-OVERFITTING)
    # Goal: Reduce 39.5% CV-LB gap by adding regularization
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,  # Lower (was 0.05)
        "num_leaves": 31,  # 2^5 - 1 (was 64)
        "max_depth": 5,  # Shallower (was 6)
        "min_child_samples": 50,  # Higher (was 20)
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,  # L1 regularization (was 0.0)
        "reg_lambda": 0.1,  # L2 regularization (was 0.0)
        "n_estimators": 1000,  # More trees with lower LR (was 600)
        "random_state": 42,
        "verbosity": -1,
    }

    print("Training LightGBM...")
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # Validation score
    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)

    mask = y_val > 0
    if mask.sum() > 0:
        holdout_cv = np.sqrt(mean_squared_log_error(y_val[mask], preds[mask]))
    else:
        holdout_cv = float('inf')

    print(f"\nHoldout CV: {holdout_cv:.4f}")
    print(f"XGBoost v1: 0.4842")
    print(f"XGBoost v18 (Optuna): 0.4872")

    print(f"v19 (12 features): 0.3572")
    print(f"v38 (9 features): {holdout_cv:.4f}")

    if holdout_cv < 0.3572:
        improvement = 0.3572 - holdout_cv
        print(f"✓ IMPROVEMENT: -{improvement:.4f} ({improvement/0.3572*100:.1f}%)")
    else:
        decline = holdout_cv - 0.3572
        print(f"✗ Declined: +{decline:.4f}")
    print()

    # Train on full data
    print("Training on full dataset...")
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    # Generate predictions
    print("Generating test predictions...")
    X_test = test_df[feature_cols]
    predictions = final_model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    # Create submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": predictions
    })

    cv_str = f"{holdout_cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"lgbm_v38_trimmed_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # Log to MLflow
    with mlflow.start_run(run_name="lgbm_v38_trimmed"):
        mlflow.log_params(params)
        mlflow.log_metric("holdout_cv", holdout_cv)
        mlflow.log_metric("vs_v19", holdout_cv - 0.3572)
        mlflow.log_metric("num_features", len(feature_cols))
        mlflow.log_artifact(str(submission_path))

    # Log to Langfuse
    if HAS_LANGFUSE:
        log_kaggle_model(
            name="v38_trimmed",
            competition="store-sales-time-series-forecasting",
            model_type="LightGBM",
            cv_score=holdout_cv,
            lb_score=None,
            features=len(feature_cols),
            hyperparameters=params,
            status="testing"
        )

    print()
    print("="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Holdout CV: {holdout_cv:.4f}")
    print(f"vs v19: {holdout_cv - 0.3572:.4f}")
    print(f"Features: {len(feature_cols)} (trimmed from 12)")
    print(f"Submission: {submission_path.name}")

    return {
        "holdout_cv": holdout_cv,
        "vs_v19": holdout_cv - 0.3572,
        "num_features": len(feature_cols),
        "success": True
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
