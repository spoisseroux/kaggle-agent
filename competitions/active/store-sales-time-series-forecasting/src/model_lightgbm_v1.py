#!/usr/bin/env python3
"""LightGBM v1 - v1 Features

LightGBM is often faster than XGBoost and can be equally performant.
Using v1's proven feature set.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import mlflow

DATA_DIR = Path("data/store-sales-time-series-forecasting")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")


def load_data():
    """Load datasets."""
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
    """Create v1's full feature set."""
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
    """Add holiday indicator."""
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    """Train with LightGBM."""
    print("="*70)
    print("LightGBM v1 - Fast Gradient Boosting")
    print("="*70)
    print("Using v1 features with LightGBM (GPU)")
    print()

    train_df, test_df, holidays = load_data()

    print("Creating features...")
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)

    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)

    train_df = train_df.dropna()

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

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

    print("\nTraining with 5-fold TimeSeriesSplit CV...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'device': 'cpu',
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"\nFold {fold}/5 - Train: {len(train_idx):,}, Val: {len(val_idx):,}")

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        )

        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = np.clip(y_pred, 0, None)

        mask = y_val > 0
        rmsle = np.sqrt(mean_squared_log_error(y_val[mask], y_pred[mask]))
        cv_scores.append(rmsle)
        print(f"  Fold {fold} RMSLE: {rmsle:.6f}")

    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    print(f"\nCV Results: {mean_cv:.6f} (+/- {std_cv:.6f})")
    print(f"All folds: {[f'{s:.6f}' for s in cv_scores]}")

    print(f"\nComparison:")
    print(f"  XGBoost v10 (best): CV 0.370")
    print(f"  LightGBM v1: CV {mean_cv:.3f}")

    if mean_cv < 0.370:
        print(f"✅ BEATS XGBOOST! {((0.370 - mean_cv) / 0.370 * 100):.1f}% better")
    else:
        print(f"⚠️ XGBoost still better (by {mean_cv - 0.370:.3f})")

    print("\nTraining final model...")
    final_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(lgb_params, final_data, num_boost_round=500)

    test_preds = final_model.predict(X_test)
    test_preds = np.clip(test_preds, 0, None)

    submission = pd.DataFrame({"id": test_df["id"], "sales": test_preds})
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / f"lightgbm_v1_{mean_cv:.4f}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\n✅ Saved: {submission_path.name}")

    mlflow.set_experiment("store-sales-forecasting")
    with mlflow.start_run(run_name="lightgbm_v1"):
        mlflow.log_param("model", "lightgbm")
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_metric("cv_rmsle_mean", mean_cv)
        mlflow.log_metric("cv_rmsle_std", std_cv)

    return mean_cv


if __name__ == "__main__":
    cv_score = main()
