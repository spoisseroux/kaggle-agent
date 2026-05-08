"""Optuna hyperparameter tuning for LightGBM

Searches for optimal hyperparameters to improve beyond 0.401 RMSLE baseline.
Target: 0.38-0.39 RMSLE
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import optuna
import mlflow
from datetime import datetime

# Paths
DATA_DIR = Path("data/store-sales-time-series-forecasting")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")


def load_data():
    """Load all datasets."""
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
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    return train, test, holidays


def create_features(df, is_train=True, train_df=None):
    """Create comprehensive time series features."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    df["Time"] = df.groupby(["store_nbr", "family"], observed=True).cumcount()

    # Seasonality
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    if is_train:
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

        df["onpromotion_lag1"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(1)
        df["onpromotion_lag7"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(7)
        df["onpromotion_lag1"] = df["onpromotion_lag1"].fillna(0)
        df["onpromotion_lag7"] = df["onpromotion_lag7"].fillna(0)
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

            for lag in [1, 2, 3, 7, 14, 21, 28]:
                df[f"Lag_{lag}"] = df["lag_mean"].fillna(0)

            for window in [7, 14, 30, 60, 90]:
                df[f"Roll_mean_{window}"] = df["lag_mean"].fillna(0)
                df[f"Roll_std_{window}"] = df["lag_std"].fillna(0)

            df["onpromotion_lag1"] = df["onpromotion"]
            df["onpromotion_lag7"] = df["onpromotion"]

            df = df.drop(columns=["lag_mean", "lag_std"])

    return df


def add_holidays(df, holidays):
    """Add holiday indicator."""
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function."""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
        ],
    )

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = np.clip(y_pred, 0, None)

    rmsle = np.sqrt(mean_squared_log_error(y_val + 1, y_pred + 1))
    return rmsle


def main():
    """Run Optuna hyperparameter tuning."""
    print("=== Optuna Hyperparameter Tuning for LightGBM ===\n")

    # Load and prepare data
    train_df, test_df, holidays = load_data()

    print("Creating features...")
    train_df = create_features(train_df, is_train=True)
    test_df = create_features(test_df, is_train=False, train_df=train_df)
    train_df = add_holidays(train_df, holidays)
    test_df = add_holidays(test_df, holidays)
    train_df = train_df.dropna()

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Time-based split
    split_date = train_df["date"].max() - pd.Timedelta(days=15)
    train_mask = train_df["date"] <= split_date
    val_mask = train_df["date"] > split_date

    feature_cols = [
        "Time",
        "day_of_week", "day_of_month", "month", "week_of_year",
        "is_month_end", "is_month_start", "is_weekend",
        "Lag_1", "Lag_2", "Lag_3", "Lag_7", "Lag_14", "Lag_21", "Lag_28",
        "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "Roll_mean_60", "Roll_mean_90",
        "Roll_std_7", "Roll_std_14", "Roll_std_30", "Roll_std_60", "Roll_std_90",
        "onpromotion", "onpromotion_lag1", "onpromotion_lag7",
        "is_holiday",
    ]

    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, "sales"]
    X_val = train_df.loc[val_mask, feature_cols]
    y_val = train_df.loc[val_mask, "sales"]

    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")

    # Run Optuna
    print("\n=== Starting Optuna Optimization ===")
    print("Baseline RMSLE: 0.401283")
    print("Target: <0.39 RMSLE\n")

    study = optuna.create_study(
        direction="minimize",
        study_name="lgbm_store_sales",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=50,
        show_progress_bar=True,
        callbacks=[
            lambda study, trial: print(
                f"Trial {trial.number}: RMSLE = {trial.value:.6f} (best: {study.best_value:.6f})"
            )
        ],
    )

    print("\n=== Optimization Complete ===")
    print(f"Best RMSLE: {study.best_value:.6f}")
    print(f"Improvement: {0.401283 - study.best_value:.6f}")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Train final model with best parameters
    print("\n=== Training final model with best parameters ===")
    mlflow.set_experiment("store-sales-optuna")

    with mlflow.start_run(run_name=f"lgbm_optuna_best_{datetime.now().strftime('%H%M')}"):
        best_params = study.best_params.copy()
        best_params.update({
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
        })

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred_val = np.clip(y_pred_val, 0, None)
        final_rmsle = np.sqrt(mean_squared_log_error(y_val + 1, y_pred_val + 1))

        print(f"\nFinal validation RMSLE: {final_rmsle:.6f}")

        # Log to MLflow
        for key, value in best_params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_metric("cv_rmsle", final_rmsle)
        mlflow.log_metric("best_iteration", model.best_iteration)
        mlflow.log_metric("improvement_vs_baseline", 0.401283 - final_rmsle)

        # Generate submission
        print("\nGenerating submission...")
        X_test = test_df[feature_cols]
        y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred_test = np.clip(y_pred_test, 0, None)

        submission = test_df[["id"]].copy()
        submission["sales"] = y_pred_test

        submission_path = SUBMISSION_DIR / f"lgbm_optuna_{final_rmsle:.4f}.csv"
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved: {submission_path}")

        mlflow.log_artifact(str(submission_path))

        print(f"\n✅ Optuna tuning complete!")
        print(f"Best RMSLE: {final_rmsle:.6f}")
        print(f"Improvement: {(0.401283 - final_rmsle) / 0.401283 * 100:.1f}%")

        return final_rmsle


if __name__ == "__main__":
    main()
