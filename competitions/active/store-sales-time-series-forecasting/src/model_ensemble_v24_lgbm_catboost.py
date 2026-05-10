#!/usr/bin/env python3
"""Ensemble v24 - LightGBM + CatBoost

STRATEGY: Combine best and middle models
- LightGBM v19 (CV 0.357, LB 0.498) - BEST
- CatBoost v21 (CV 0.446) - MIDDLE

Test if ensemble improves over single best model.
Uses weighted averaging with weights optimized on validation.

WARNING: v16 ensemble failed catastrophically. This is simpler:
- Same v1 features for both models
- No advanced features
- Proper validation
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
from catboost import CatBoostRegressor
import mlflow

# Add parent directory to path for core imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from core.langfuse_logger import log_kaggle_model

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
    """v1 proven feature set - EXACT COPY."""
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
    print("Ensemble v24 - LightGBM + CatBoost")
    print("="*70)
    print("Combining best (LightGBM) and middle (CatBoost) models")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-ensemble")

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    # Create features
    print("Creating v1 features...")
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)

    # Feature columns
    exclude_cols = ["id", "date", "sales", "store_nbr", "family"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"Features: {len(feature_cols)} (v1 proven)")
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

    # Model 1: LightGBM (v19 params)
    print("Training LightGBM...")
    lgb_params = {
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
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_estimators": 600,
        "random_state": 42,
        "verbosity": -1,
    }

    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    lgb_val_preds = np.maximum(lgb_model.predict(X_val), 0)
    mask = y_val > 0
    lgb_score = np.sqrt(mean_squared_log_error(y_val[mask], lgb_val_preds[mask]))
    print(f"  LightGBM CV: {lgb_score:.4f}")

    # Model 2: CatBoost (v21 params)
    print("Training CatBoost...")
    cat_params = {
        "loss_function": "RMSE",
        "learning_rate": 0.05,
        "depth": 6,
        "iterations": 600,
        "random_seed": 42,
        "verbose": False,
        "task_type": "GPU",
        "devices": "0",
    }

    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )

    cat_val_preds = np.maximum(cat_model.predict(X_val), 0)
    cat_score = np.sqrt(mean_squared_log_error(y_val[mask], cat_val_preds[mask]))
    print(f"  CatBoost CV: {cat_score:.4f}")
    print()

    # Find optimal weights
    print("Optimizing ensemble weights...")
    best_weight = 0.5
    best_score = float('inf')

    for lgb_weight in np.arange(0.0, 1.05, 0.05):
        cat_weight = 1 - lgb_weight
        ensemble_preds = lgb_weight * lgb_val_preds + cat_weight * cat_val_preds
        score = np.sqrt(mean_squared_log_error(y_val[mask], ensemble_preds[mask]))

        if score < best_score:
            best_score = score
            best_weight = lgb_weight

    print(f"Best weights: LightGBM={best_weight:.2f}, CatBoost={1-best_weight:.2f}")
    print(f"Ensemble CV: {best_score:.4f}")
    print(f"LightGBM v19: {lgb_score:.4f}")
    print(f"CatBoost v21: {cat_score:.4f}")

    if best_score < lgb_score:
        improvement = lgb_score - best_score
        print(f"✓ IMPROVEMENT: -{improvement:.4f} over LightGBM")
    else:
        print(f"✗ No improvement - LightGBM alone is better")
    print()

    # Train final models on full data
    print("Training final models on full data...")
    final_lgb = lgb.LGBMRegressor(**lgb_params)
    final_lgb.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    final_cat = CatBoostRegressor(**cat_params)
    final_cat.fit(X, y, verbose=False)

    # Generate predictions
    print("Generating predictions...")
    X_test = test_df[feature_cols]

    lgb_test_preds = np.maximum(final_lgb.predict(X_test), 0)
    cat_test_preds = np.maximum(final_cat.predict(X_test), 0)
    ensemble_preds = best_weight * lgb_test_preds + (1 - best_weight) * cat_test_preds

    # Submission
    submission = pd.DataFrame({"id": test_df["id"], "sales": ensemble_preds})
    cv_str = f"{best_score:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"ensemble_v24_lgbm_catboost_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # MLflow
    with mlflow.start_run(run_name="ensemble_v24_lgbm_catboost"):
        mlflow.log_param("lgb_weight", best_weight)
        mlflow.log_param("cat_weight", 1 - best_weight)
        mlflow.log_metric("ensemble_cv", best_score)
        mlflow.log_metric("lgb_cv", lgb_score)
        mlflow.log_metric("cat_cv", cat_score)
        mlflow.log_metric("improvement", lgb_score - best_score)
        mlflow.log_artifact(str(submission_path))

    # Log to Langfuse
    log_kaggle_model(
        name="v24-ensemble",
        competition="store-sales-time-series-forecasting",
        model_type="LightGBM+CatBoost",
        cv_score=best_score,
        lb_score=None,  # Not yet submitted
        features=len(feature_cols),
        hyperparameters={"lgb_weight": best_weight, "cat_weight": 1 - best_weight},
        status="ensemble_no_improvement"
    )

    print()
    print("="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Ensemble CV: {best_score:.4f}")
    print(f"Weights: LightGBM {best_weight:.2f}, CatBoost {1-best_weight:.2f}")
    print(f"Submission: {submission_path.name}")

    return {
        "ensemble_cv": best_score,
        "lgb_cv": lgb_score,
        "cat_cv": cat_score,
        "lgb_weight": best_weight,
        "success": True
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
