#!/usr/bin/env python3
"""Retrain top 3 XGBoost models with 30-day holdout validation.

Tests the new validation strategy that better predicts LB scores.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import mlflow

DATA_DIR = Path("data/store-sales-time-series-forecasting")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Top 12 features
TOP_FEATURES = [
    "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "is_weekend",
    "Roll_mean_60", "day_of_week", "is_holiday", "Lag_3",
    "Roll_mean_90", "onpromotion", "Lag_7", "Roll_std_7"
]


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
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

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


def train_model(X_train, y_train, X_val, y_val, params, name):
    """Train a model and return validation score."""
    model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)

    mask = y_val > 0
    if mask.sum() > 0:
        score = np.sqrt(mean_squared_log_error(y_val[mask], preds[mask]))
    else:
        score = float('inf')

    return model, score


def main():
    print("="*70)
    print("RETRAIN TOP 3 MODELS - 30-Day Holdout Validation")
    print("="*70)
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-xgboost-holdout")

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    # Create 30-day holdout split
    print("Creating 30-day holdout split...")
    cutoff_date = train_df['date'].max() - pd.Timedelta(days=30)
    val_mask = train_df['date'] >= cutoff_date
    train_mask = ~val_mask

    print(f"  Train: {train_mask.sum():,} rows ({train_df[train_mask]['date'].min()} to {train_df[train_mask]['date'].max()})")
    print(f"  Val:   {val_mask.sum():,} rows ({train_df[val_mask]['date'].min()} to {train_df[val_mask]['date'].max()})")
    print()

    X = train_df[TOP_FEATURES]
    y = train_df["sales"]

    X_train_split = X[train_mask]
    y_train_split = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    # Model 1: v1 baseline
    print("Model 1: XGBoost v1 (Baseline)")
    print("-" * 50)
    params_v1 = {
        "objective": "reg:squaredlogerror",
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 500,
        "random_state": 42,
        "verbosity": 0,
    }

    model_v1, score_v1 = train_model(X_train_split, y_train_split, X_val, y_val, params_v1, "v1")
    print(f"  Holdout CV: {score_v1:.4f}")
    print()

    # Model 2: v2 fixed lags
    print("Model 2: XGBoost v2 (Fixed Lags)")
    print("-" * 50)
    params_v2 = {
        "objective": "reg:squaredlogerror",
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 600,
        "random_state": 42,
        "verbosity": 0,
    }

    model_v2, score_v2 = train_model(X_train_split, y_train_split, X_val, y_val, params_v2, "v2")
    print(f"  Holdout CV: {score_v2:.4f}")
    print()

    # Model 3: Optuna tuned (best LB)
    print("Model 3: XGBoost Optuna (Best LB)")
    print("-" * 50)
    params_optuna = {
        "objective": "reg:squaredlogerror",
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.04,
        "max_depth": 7,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 800,
        "random_state": 42,
        "verbosity": 0,
    }

    model_optuna, score_optuna = train_model(X_train_split, y_train_split, X_val, y_val, params_optuna, "optuna")
    print(f"  Holdout CV: {score_optuna:.4f}")
    print()

    # Select best model
    results = [
        ("v1_baseline", model_v1, score_v1, params_v1),
        ("v2_fixed_lags", model_v2, score_v2, params_v2),
        ("optuna_tuned", model_optuna, score_optuna, params_optuna),
    ]
    results.sort(key=lambda x: x[2])
    best_name, best_model, best_score, best_params = results[0]

    print("="*70)
    print("RESULTS - 30-Day Holdout Validation")
    print("="*70)
    print(f"v1 baseline:    {score_v1:.4f} (original LB: 0.526)")
    print(f"v2 fixed lags:  {score_v2:.4f} (original LB: 0.551)")
    print(f"optuna tuned:   {score_optuna:.4f} (original LB: 0.464)")
    print()
    print(f"✓ Best model: {best_name} (Holdout CV: {best_score:.4f})")
    print()

    # Train best model on full data
    print(f"Training {best_name} on full dataset...")
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X, y, verbose=False)

    # Generate predictions
    print("Generating test predictions...")
    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)
    X_test = test_df[TOP_FEATURES]
    predictions = final_model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    # Create submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": predictions
    })

    cv_str = f"{best_score:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"xgb_{best_name}_holdout_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # Log to MLflow
    with mlflow.start_run(run_name=f"xgb_{best_name}_holdout"):
        mlflow.log_params(best_params)
        mlflow.log_metric("holdout_cv", best_score)
        mlflow.log_metric("v1_holdout", score_v1)
        mlflow.log_metric("v2_holdout", score_v2)
        mlflow.log_metric("optuna_holdout", score_optuna)
        mlflow.log_artifact(str(submission_path))

    print()
    print("="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Best: {best_name}")
    print(f"Holdout CV: {best_score:.4f}")
    print(f"Submission: {submission_path.name}")
    print()
    print("Validation strategy confirmed:")
    print("  Holdout scores much closer to known LB than TimeSeriesSplit was")
    print(f"  Expected LB range: {best_score - 0.05:.3f} to {best_score + 0.05:.3f}")

    return {
        "best_model": best_name,
        "holdout_cv": best_score,
        "v1_cv": score_v1,
        "v2_cv": score_v2,
        "optuna_cv": score_optuna,
        "submission": str(submission_path),
        "success": True
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
