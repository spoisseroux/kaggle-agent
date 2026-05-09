#!/usr/bin/env python3
"""Optuna hyperparameter optimization for XGBoost v1

Systematically search for better hyperparameters to improve CV score.
Target: Beat 0.321 CV baseline
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import optuna

DATA_DIR = Path("data/store-sales-time-series-forecasting")


def load_data():
    """Load datasets."""
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
    )
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    return train, holidays


def create_features(df, holidays):
    """Create all v1 features."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    df["Time"] = df.groupby(["store_nbr", "family"], observed=True).cumcount()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

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

    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)

    return df


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function."""

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'seed': 42
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    preds = model.predict(dval)
    preds = np.maximum(preds, 0)

    mask = y_val > 0
    rmsle = np.sqrt(mean_squared_log_error(y_val[mask], preds[mask]))

    return rmsle


def main():
    print("=" * 70)
    print("Optuna Hyperparameter Optimization - XGBoost")
    print("=" * 70)
    print("Target: Beat baseline CV 0.321032")
    print("Trials: 50")
    print()

    # Load and prepare data
    print("Loading data...")
    train_df, holidays = load_data()
    train_df = create_features(train_df, holidays)
    train_df = train_df.dropna()

    # Split
    val_size = int(len(train_df) * 0.1)
    val_df = train_df.tail(val_size).copy()
    train_df = train_df.head(len(train_df) - val_size).copy()

    feature_cols = [c for c in train_df.columns if c not in ['sales', 'date', 'store_nbr', 'family', 'id']]

    X_train = train_df[feature_cols]
    y_train = train_df['sales']
    X_val = val_df[feature_cols]
    y_val = val_df['sales']

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"Features: {len(feature_cols)}")
    print()

    # Create study
    print("Starting Optuna optimization...")
    print()

    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=50,
        show_progress_bar=True
    )

    print()
    print("=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Best CV RMSLE:    {study.best_value:.6f}")
    print(f"Baseline CV:      0.321032")

    improvement = 0.321032 - study.best_value
    if improvement > 0:
        print(f"Improvement:      -{improvement:.6f} ✅")
    else:
        print(f"Change:           +{abs(improvement):.6f}")

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
