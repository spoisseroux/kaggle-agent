#!/usr/bin/env python3
"""Optuna hyperparameter tuning with TimeSeriesSplit CV

Previous Optuna used wrong validation (simple split).
Re-run with proper CV and reduced feature set.

Target: Find hyperparameters that optimize for TimeSeriesSplit CV,
which should better predict actual LB performance.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import optuna

DATA_DIR = Path("data/store-sales-time-series-forecasting")

# Use reduced feature set (shown to be better with proper CV)
FEATURES = [
    'Time', 'day_of_week', 'month', 'is_weekend',
    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 'Lag_14', 'Lag_28',
    'Roll_mean_7', 'Roll_mean_14', 'Roll_mean_30',
    'Roll_std_7', 'Roll_std_14', 'Roll_std_30',
    'onpromotion', 'onpromotion_lag1', 'is_holiday'
]


def load_and_prepare_data():
    """Load and create features."""
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
    )
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])

    # Feature engineering
    train = train.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    train["Time"] = train.groupby(["store_nbr", "family"], observed=True).cumcount()
    train["day_of_week"] = train["date"].dt.dayofweek
    train["day_of_month"] = train["date"].dt.day
    train["month"] = train["date"].dt.month
    train["is_month_end"] = train["date"].dt.is_month_end.astype(int)
    train["is_month_start"] = train["date"].dt.is_month_start.astype(int)
    train["is_weekend"] = (train["day_of_week"] >= 5).astype(int)
    train["week_of_year"] = train["date"].dt.isocalendar().week.astype(int)

    for lag in [1, 2, 3, 7, 14, 21, 28]:
        train[f"Lag_{lag}"] = train.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

    for window in [7, 14, 30, 60, 90]:
        train[f"Roll_mean_{window}"] = (
            train.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        train[f"Roll_std_{window}"] = (
            train.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )

    train["onpromotion_lag1"] = train.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(1).fillna(0)
    train["onpromotion_lag7"] = train.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(7).fillna(0)

    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    train["is_holiday"] = train["date"].isin(national_holidays).astype(int)

    train = train.dropna()
    return train


def objective(trial, X, y):
    """Optuna objective with TimeSeriesSplit CV."""
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'seed': 42
    }

    # Use 3-fold TimeSeriesSplit for faster tuning
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

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

        mask = y_val > 0
        rmsle = np.sqrt(mean_squared_log_error(y_val[mask], y_pred[mask]))
        cv_scores.append(rmsle)

    return np.mean(cv_scores)


def main():
    print("=" * 70)
    print("Optuna Tuning with TimeSeriesSplit CV + Reduced Features")
    print("=" * 70)
    print("Features: 19 (reduced set)")
    print("CV: 3-fold TimeSeriesSplit")
    print("Trials: 30 (reduced from 50 due to time)")
    print("Baseline: 0.367 (reduced features with default params)")
    print()

    # Load data
    print("Loading data...")
    train_df = load_and_prepare_data()
    X = train_df[FEATURES]
    y = train_df['sales']
    print(f"Data: {len(X):,} samples, {len(FEATURES)} features")
    print()

    # Run Optuna
    print("Starting optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=30,
        show_progress_bar=True
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Best CV (TimeSeriesSplit): {study.best_value:.6f}")
    print(f"Baseline (reduced, default): 0.367232")

    improvement = 0.367232 - study.best_value
    if improvement > 0:
        print(f"Improvement: -{improvement:.6f} ✅")
    else:
        print(f"Change: +{abs(improvement):.6f}")

    print()
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key:20s} {value}")

    print()
    print(f"Total trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")

    return study.best_value, study.best_params


if __name__ == "__main__":
    best_cv, best_params = main()
    print(f"\n✅ Optuna complete - Best CV: {best_cv:.6f}")
