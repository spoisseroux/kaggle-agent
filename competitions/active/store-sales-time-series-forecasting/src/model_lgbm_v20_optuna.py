#!/usr/bin/env python3
"""LightGBM v20 - Optuna Tuning on v1 Features

STRATEGY: Optimize LightGBM hyperparameters
- Uses v1's exact feature set (validated)
- LightGBM model (better than XGBoost: LB 0.498 vs 0.529)
- Optuna hyperparameter optimization
- Expected: Better than v19's LB 0.498

Features (12 total - PROVEN):
- Lags: 3, 7
- Rolling means: 7, 14, 30, 60, 90
- Rolling std: 7
- Temporal: day_of_week, is_weekend, is_holiday
- Promotion: onpromotion
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import optuna
import mlflow

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


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for LightGBM."""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": 42,

        # Tunable params
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "n_estimators": 1000,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

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
    print("LightGBM v20 - Optuna on v1 Features")
    print("="*70)
    print("Optimizing LightGBM hyperparameters")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-lightgbm-optuna")

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    # Create features (v1 exact)
    print("Creating v1 features...")
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)

    # Feature columns
    exclude_cols = ["id", "date", "sales", "store_nbr", "family"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"Features: {len(feature_cols)} (v1 proven set)")
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

    # Optuna
    print("Running Optuna (30 trials, ~25 min)...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=30, show_progress_bar=False)

    print(f"\nBest CV: {study.best_value:.4f}")
    print(f"LightGBM v19 baseline: 0.3572")

    if study.best_value < 0.3572:
        imp = 0.3572 - study.best_value
        print(f"✓ IMPROVEMENT: -{imp:.4f} ({imp/0.3572*100:.1f}%)")
    else:
        print(f"✗ No improvement")

    print(f"\nBest params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print()

    # Train final
    print("Training final model...")
    best_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": 42,
        "n_estimators": 1000,
        "subsample_freq": 1,
        **study.best_params
    }

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    # Predict
    print("Generating predictions...")
    X_test = test_df[feature_cols]
    predictions = final_model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    # Submission
    submission = pd.DataFrame({"id": test_df["id"], "sales": predictions})
    cv_str = f"{study.best_value:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"lgbm_v20_optuna_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # MLflow
    with mlflow.start_run(run_name="lgbm_v20_optuna"):
        mlflow.log_params(best_params)
        mlflow.log_metric("holdout_cv", study.best_value)
        mlflow.log_metric("vs_v19", study.best_value - 0.3572)
        mlflow.log_artifact(str(submission_path))

    print()
    print("="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Holdout CV: {study.best_value:.4f}")
    print(f"Expected LB: ~{study.best_value * 1.395:.2f} (based on v19's 39.5% gap)")
    print(f"Submission: {submission_path.name}")

    return {"holdout_cv": study.best_value, "success": True}


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
