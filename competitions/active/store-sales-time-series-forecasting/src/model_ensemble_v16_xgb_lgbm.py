#!/usr/bin/env python3
"""Ensemble v16 - XGBoost + LightGBM Stacking

Combines v15 XGBoost (CV 0.365) with tuned LightGBM using weighted averaging.
Tests if ensemble can beat single best model.

Uses 30-day holdout validation.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import mlflow

DATA_DIR = Path("data/store-sales-time-series-forecasting")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "int16", "family": "category"},
        parse_dates=["date"],
    )
    test = pd.read_csv(
        DATA_DIR / "test.csv",
        dtype={"store_nbr": "int16", "family": "category"},
        parse_dates=["date"],
    )
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    return train, test, holidays, stores


def create_features(df, is_train=True, train_stats=None):
    """Create all v14/v15 advanced features."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Base temporal
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(np.int16)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year - 2013
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(np.int8)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(np.int8)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(np.int8)

    if is_train:
        # Lags
        for lag in [3, 7, 14]:
            df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

        # Rolling
        for window in [7, 14, 30, 60]:
            df[f"Roll_mean_{window}"] = (
                df.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )

        df["Roll_std_7"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=7, min_periods=1).std())
        )

        # EWMA
        for span in [7, 14, 30]:
            df[f"EWMA_{span}"] = (
                df.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            )

        # WoW diff
        df["WoW_diff"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].diff(7)
    else:
        # Use train stats for test
        for lag in [3, 7, 14]:
            df[f"Lag_{lag}"] = train_stats["mean"].fillna(0)
        for window in [7, 14, 30, 60]:
            df[f"Roll_mean_{window}"] = train_stats["mean"].fillna(0)
        df["Roll_std_7"] = train_stats["std"].fillna(0)
        for span in [7, 14, 30]:
            df[f"EWMA_{span}"] = train_stats["mean"].fillna(0)
        df["WoW_diff"] = 0

    # Interactions
    df["promo_holiday"] = (df["onpromotion"] * df["is_holiday"]).astype(np.int8)
    df["weekend_promo"] = (df["is_weekend"] * df["onpromotion"]).astype(np.int8)

    # Store-family encoding
    df["store_family"] = df["store_nbr"].astype(str) + "_" + df["family"].astype(str)
    le = LabelEncoder()
    df["store_family_enc"] = le.fit_transform(df["store_family"])
    df = df.drop(columns=["store_family"])

    return df


def main():
    print("="*70)
    print("Ensemble v16 - XGBoost + LightGBM Stacking")
    print("="*70)
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-ensemble")

    # Load and prepare data
    print("Loading data...")
    train_df, test_df, holidays, stores = load_data()

    # Add holidays and stores
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    train_df["is_holiday"] = train_df["date"].isin(national_holidays).astype(np.int8)
    test_df["is_holiday"] = test_df["date"].isin(national_holidays).astype(np.int8)

    train_df = train_df.merge(stores[["store_nbr", "type", "cluster"]], on="store_nbr", how="left")
    test_df = test_df.merge(stores[["store_nbr", "type", "cluster"]], on="store_nbr", how="left")

    le_type = LabelEncoder()
    le_cluster = LabelEncoder()
    train_df["store_type_enc"] = le_type.fit_transform(train_df["type"].astype(str))
    train_df["store_cluster_enc"] = le_cluster.fit_transform(train_df["cluster"].astype(str))
    test_df["store_type_enc"] = le_type.transform(test_df["type"].astype(str))
    test_df["store_cluster_enc"] = le_cluster.transform(test_df["cluster"].astype(str))
    train_df = train_df.drop(columns=["type", "cluster"])
    test_df = test_df.drop(columns=["type", "cluster"])

    # Features
    print("Creating features...")
    train_df = create_features(train_df, is_train=True)

    # Get train stats for test
    train_stats = (
        train_df.groupby(["store_nbr", "family"], observed=True)
        .tail(30)
        .groupby(["store_nbr", "family"], observed=True)["sales"]
        .agg(["mean", "std"])
    )
    test_df = test_df.merge(train_stats, on=["store_nbr", "family"], how="left")
    test_df = create_features(test_df, is_train=False, train_stats=test_df)

    train_df = train_df.dropna()

    # Get feature columns
    exclude_cols = ["id", "date", "sales", "store_nbr", "family"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # 30-day holdout split
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

    # Model 1: XGBoost (v15 best params)
    print("Training XGBoost (v15 params)...")
    xgb_params = {
        "objective": "reg:squaredlogerror",
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 0,
        "random_state": 42,
        "learning_rate": 0.058,
        "max_depth": 7,
        "min_child_weight": 1,
        "subsample": 0.91,
        "colsample_bytree": 0.84,
        "reg_alpha": 0.038,
        "reg_lambda": 1.67,
        "n_estimators": 1000,
    }

    xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    xgb_val_preds = np.maximum(xgb_model.predict(X_val), 0)
    mask = y_val > 0
    xgb_score = np.sqrt(mean_squared_log_error(y_val[mask], xgb_val_preds[mask]))
    print(f"  XGBoost CV: {xgb_score:.4f}")

    # Model 2: LightGBM (tuned for ensemble)
    print("Training LightGBM...")
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 40,
        "max_depth": 7,
        "min_child_samples": 15,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 1.5,
        "n_estimators": 1000,
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
    lgb_score = np.sqrt(mean_squared_log_error(y_val[mask], lgb_val_preds[mask]))
    print(f"  LightGBM CV: {lgb_score:.4f}")
    print()

    # Test different ensemble weights
    print("Testing ensemble weights...")
    best_weight = 0.5
    best_score = float('inf')

    for xgb_weight in np.arange(0.3, 0.8, 0.05):
        lgb_weight = 1 - xgb_weight
        ensemble_preds = xgb_weight * xgb_val_preds + lgb_weight * lgb_val_preds
        score = np.sqrt(mean_squared_log_error(y_val[mask], ensemble_preds[mask]))

        if score < best_score:
            best_score = score
            best_weight = xgb_weight

    print(f"Best weights: XGBoost={best_weight:.2f}, LightGBM={1-best_weight:.2f}")
    print(f"Ensemble CV: {best_score:.4f}")
    print(f"v15 baseline: {xgb_score:.4f}")

    if best_score < xgb_score:
        improvement = xgb_score - best_score
        print(f"✓ IMPROVEMENT: -{improvement:.4f} ({improvement/xgb_score*100:.1f}%)")
    else:
        print(f"✗ No improvement - using XGBoost alone")
        best_weight = 1.0
        best_score = xgb_score
    print()

    # Train final models on full data
    print("Training final models on full data...")
    final_xgb = xgb.XGBRegressor(**xgb_params)
    final_xgb.fit(X, y, verbose=False)

    final_lgb = lgb.LGBMRegressor(**lgb_params)
    final_lgb.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    # Generate predictions
    print("Generating predictions...")
    X_test = test_df[feature_cols]

    xgb_test_preds = np.maximum(final_xgb.predict(X_test), 0)
    lgb_test_preds = np.maximum(final_lgb.predict(X_test), 0)
    ensemble_preds = best_weight * xgb_test_preds + (1 - best_weight) * lgb_test_preds

    # Submission
    submission = pd.DataFrame({"id": test_df["id"], "sales": ensemble_preds})
    cv_str = f"{best_score:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"ensemble_v16_xgb_lgbm_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # MLflow
    with mlflow.start_run(run_name="ensemble_v16"):
        mlflow.log_param("xgb_weight", best_weight)
        mlflow.log_param("lgbm_weight", 1 - best_weight)
        mlflow.log_metric("ensemble_cv", best_score)
        mlflow.log_metric("xgb_cv", xgb_score)
        mlflow.log_metric("lgbm_cv", lgb_score)
        mlflow.log_metric("improvement_vs_v15", xgb_score - best_score)
        mlflow.log_artifact(str(submission_path))

    print()
    print("="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Ensemble CV: {best_score:.4f}")
    print(f"Weights: XGBoost {best_weight:.2f}, LightGBM {1-best_weight:.2f}")
    print(f"Submission: {submission_path.name}")

    return {
        "ensemble_cv": best_score,
        "xgb_cv": xgb_score,
        "lgbm_cv": lgb_score,
        "xgb_weight": best_weight,
        "success": True
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
