#!/usr/bin/env python3
"""XGBoost Optuna v1 - Hyperparameter Optimization

Using Optuna to find better hyperparameters than v10's manual tuning.
Target: Beat CV 0.370
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import optuna
import mlflow
import json

DATA_DIR = Path("data/store-sales-time-series-forecasting")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")


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
    df["Time"] = df.groupby(["store_nbr", "family"], observed=True).cumcount()
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
        df["onpromotion_lag1"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(1).fillna(0)
        df["onpromotion_lag7"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(7).fillna(0)
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
    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def objective(trial, X, y):
    """Optuna objective function."""
    params = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'seed': 42
    }

    # TimeSeriesSplit CV
    tscv = TimeSeriesSplit(n_splits=3)  # Use 3 folds for speed
    cv_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=[(dval, 'val')],
            early_stopping_rounds=30,
            verbose_eval=False
        )

        y_pred = model.predict(dval)
        y_pred = np.maximum(y_pred, 0)

        # Masked RMSLE
        mask = y_val > 0
        rmsle = np.sqrt(mean_squared_log_error(y_val[mask], y_pred[mask]))
        cv_scores.append(rmsle)

    return np.mean(cv_scores)


def main():
    print("="*70)
    print("XGBoost Optuna v1 - Hyperparameter Optimization")
    print("="*70)
    print("Finding better hyperparameters than v10's manual tuning")
    print("Baseline to beat: CV 0.370")
    print()

    # Load and prepare data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    print("Creating features...")
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)
    train_df = train_df.dropna()

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

    print(f"Features: {len(feature_cols)}, Samples: {len(X):,}")

    # Run Optuna optimization
    print("\n" + "="*70)
    print("Running Optuna hyperparameter search (50 trials)...")
    print("="*70)

    study = optuna.create_study(direction='minimize', study_name='xgboost_optuna_v1')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50, show_progress_bar=True)

    print(f"\n{'='*70}")
    print(f"Best trial:")
    print(f"  Value (CV RMSLE): {study.best_trial.value:.6f}")
    print(f"  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print(f"{'='*70}")

    print(f"\nComparison:")
    print(f"  v10 manual params: CV 0.370")
    print(f"  Optuna best params: CV {study.best_trial.value:.3f}")

    if study.best_trial.value < 0.370:
        improvement = ((0.370 - study.best_trial.value) / 0.370) * 100
        print(f"\n✅ IMPROVEMENT! {improvement:.1f}% better")
    else:
        print(f"\n⚠️ v10 still better")

    # Save best params
    best_params_file = Path("competitions/active/store-sales-time-series-forecasting/optuna_best_params.json")
    with open(best_params_file, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nBest params saved to: {best_params_file}")

    # Train final model with best params
    print("\nTraining final model with best params...")
    final_params = {**study.best_params, 'objective': 'reg:squarederror', 'seed': 42}

    dtrain_full = xgb.DMatrix(X, label=y)
    final_model = xgb.train(final_params, dtrain_full, num_boost_round=500)

    # Predict
    dtest = xgb.DMatrix(X_test)
    test_preds = final_model.predict(dtest)
    test_preds = np.maximum(test_preds, 0)

    # Save submission
    submission = pd.DataFrame({"id": test_df["id"], "sales": test_preds})
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / f"xgb_optuna_v1_{study.best_trial.value:.4f}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\n✅ Saved: {submission_path.name}")
    print(f"📊 CV: {study.best_trial.value:.6f}")

    # MLflow
    mlflow.set_experiment("store-sales-forecasting")
    with mlflow.start_run(run_name="xgb_optuna_v1"):
        mlflow.log_param("model", "xgboost_optuna")
        mlflow.log_param("n_trials", 50)
        for key, value in study.best_params.items():
            mlflow.log_param(f"best_{key}", value)
        mlflow.log_metric("cv_rmsle", study.best_trial.value)

    return study.best_trial.value


if __name__ == "__main__":
    cv_score = main()
