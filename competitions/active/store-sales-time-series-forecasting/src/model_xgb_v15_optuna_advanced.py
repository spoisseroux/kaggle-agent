#!/usr/bin/env python3
"""XGBoost v15 - Optuna Tuning on Advanced Features

Takes v14's advanced features and applies Optuna hyperparameter optimization.
Uses 30-day holdout validation for accurate LB prediction.

Target: <0.46 holdout CV
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import optuna
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
    """Create all v14 advanced features."""
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


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function."""
    params = {
        "objective": "reg:squaredlogerror",
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 0,
        "random_state": 42,

        # Tunable params
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "n_estimators": 1000,
    }

    model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)

    mask = y_val > 0
    if mask.sum() > 0:
        score = np.sqrt(mean_squared_log_error(y_val[mask], preds[mask]))
    else:
        score = float('inf')

    return score


def main():
    print("="*70)
    print("XGBoost v15 - Optuna Tuning on Advanced Features")
    print("="*70)
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-xgboost-holdout")

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
    print(f"Features: {len(feature_cols)}")

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

    # Optuna optimization
    print("Running Optuna optimization (30 trials)...")
    print("This will take ~20-25 minutes...")
    print()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=30, show_progress_bar=False)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best holdout CV: {study.best_value:.4f}")
    print(f"v14 baseline: 0.4640")

    if study.best_value < 0.4640:
        improvement = 0.4640 - study.best_value
        print(f"✓ IMPROVEMENT: -{improvement:.4f} ({improvement/0.4640*100:.1f}%)")
    else:
        print(f"✗ No improvement")

    print(f"\nBest params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print()

    # Train final model with best params
    print("Training final model...")
    best_params = {
        "objective": "reg:squaredlogerror",
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 0,
        "random_state": 42,
        "n_estimators": 1000,
        **study.best_params
    }

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X, y, verbose=False)

    # Generate predictions
    print("Generating predictions...")
    X_test = test_df[feature_cols]
    predictions = final_model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    # Submission
    submission = pd.DataFrame({"id": test_df["id"], "sales": predictions})
    cv_str = f"{study.best_value:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"xgb_v15_optuna_adv_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # MLflow
    with mlflow.start_run(run_name="xgb_v15_optuna_advanced"):
        mlflow.log_params(best_params)
        mlflow.log_metric("holdout_cv", study.best_value)
        mlflow.log_metric("improvement_vs_v14", 0.4640 - study.best_value)
        mlflow.log_metric("optuna_trials", 30)
        mlflow.log_artifact(str(submission_path))

    print()
    print("="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Holdout CV: {study.best_value:.4f}")
    print(f"Submission: {submission_path.name}")

    return {"holdout_cv": study.best_value, "success": True}


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
