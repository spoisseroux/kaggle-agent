#!/usr/bin/env python3
"""LightGBM v48 - Seed Ensemble (Research-Backed)

INSIGHT FROM KAGGLE GRANDMASTER PLAYBOOK:
"Extra Training: Train multiple instances with different random seeds, average predictions"

STRATEGY: Reduce prediction variance through ensemble of same architecture
- Train 5 LightGBM models with seeds: 42, 123, 456, 789, 1337
- Same v1 features and hyperparameters (like v19)
- Average predictions across all 5 models
- More robust than single model

Expected:
- Individual CV scores ~0.357 (same as v19)
- Ensemble should be slightly better due to variance reduction
- LB should improve from averaging out random fluctuations
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import mlflow

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
    """v1 proven feature set"""
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


def main():
    print("="*70)
    print("LightGBM v48 - Seed Ensemble (5 models)")
    print("="*70)
    print("Strategy: Train 5 models with different seeds, average predictions")
    print("Expected: Reduced variance, more robust predictions")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-lightgbm-v1")

    print("Loading data...")
    train_df, test_df, holidays = load_data()

    print("Creating v1 features...")
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)

    exclude_cols = ["id", "date", "sales", "store_nbr", "family", "lag_mean", "lag_std"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Features: {len(feature_cols)} (v1 proven set)")
    print()

    # 30-day holdout validation
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

    # Base parameters (same as v19)
    base_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "verbosity": -1,
    }

    # Seeds for ensemble
    seeds = [42, 123, 456, 789, 1337]

    models = []
    val_predictions = []
    test_predictions = []
    individual_cv_scores = []

    print(f"Training {len(seeds)} models with different seeds...")
    print()

    for i, seed in enumerate(seeds):
        print(f"Model {i+1}/{len(seeds)} (seed={seed}):")

        params = base_params.copy()
        params["random_state"] = seed

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )

        # Validation predictions
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        val_preds = np.maximum(val_preds, 0)
        val_predictions.append(val_preds)

        individual_cv = np.sqrt(mean_squared_log_error(y_val, val_preds))
        individual_cv_scores.append(individual_cv)

        # Test predictions
        X_test = test_df[feature_cols]
        test_preds = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds = np.maximum(test_preds, 0)
        test_predictions.append(test_preds)

        models.append(model)

        print(f"  CV: {individual_cv:.4f}, Iterations: {model.best_iteration}")

    print()
    print("Individual model CV scores:")
    for i, cv in enumerate(individual_cv_scores):
        print(f"  Model {i+1}: {cv:.4f}")
    print(f"  Mean: {np.mean(individual_cv_scores):.4f}")
    print(f"  Std:  {np.std(individual_cv_scores):.4f}")
    print()

    # Ensemble predictions (average)
    ensemble_val_preds = np.mean(val_predictions, axis=0)
    ensemble_test_preds = np.mean(test_predictions, axis=0)

    ensemble_cv = np.sqrt(mean_squared_log_error(y_val, ensemble_val_preds))

    print("="*70)
    print("ENSEMBLE RESULTS:")
    print("="*70)
    print(f"Ensemble CV: {ensemble_cv:.4f}")
    print(f"v19 (single model): 0.3572")
    print(f"Improvement over mean individual: {(np.mean(individual_cv_scores) - ensemble_cv) / np.mean(individual_cv_scores) * 100:.2f}%")

    if ensemble_cv < 0.3572:
        print(f"✓ BEATS V19: {(0.3572 - ensemble_cv) / 0.3572 * 100:.1f}% better")
    else:
        print(f"✗ WORSE THAN V19: {(ensemble_cv - 0.3572) / 0.3572 * 100:.1f}% worse")
    print()

    # Log to MLflow
    with mlflow.start_run(run_name="lgbm_v48_seed_ensemble"):
        mlflow.log_params(base_params)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("ensemble_size", len(seeds))
        mlflow.log_param("seeds", str(seeds))

        mlflow.log_metric("ensemble_cv", ensemble_cv)
        mlflow.log_metric("mean_individual_cv", np.mean(individual_cv_scores))
        mlflow.log_metric("std_individual_cv", np.std(individual_cv_scores))
        mlflow.log_metric("vs_v19", ensemble_cv - 0.3572)

    # Save submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": ensemble_test_preds
    })

    filename = f"lgbm_v48_ensemble_{ensemble_cv:.5f}.csv".replace(".", "")
    filepath = SUBMISSION_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"Submission saved: {filename}")
    print(f"Mean prediction: {ensemble_test_preds.mean():.2f}")
    print()

    if HAS_LANGFUSE:
        try:
            log_kaggle_model(
                competition="store-sales-time-series-forecasting",
                model_name="LightGBM v48 Seed Ensemble",
                cv_score=ensemble_cv,
                model_type="lightgbm",
                params=base_params,
                features=feature_cols
            )
        except Exception:
            pass

    print("="*70)
    print("ANALYSIS:")
    print("Seed ensemble reduces variance through model averaging")
    print("Even small CV improvement translates to better LB robustness")
    print("Standard technique used by top Kaggle competitors")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
