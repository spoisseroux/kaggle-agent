#!/usr/bin/env python3
"""Ensemble v1 - XGBoost v10 + LightGBM v1

Simple weighted average ensemble of top 2 models.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import mlflow

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


def main():
    print("="*70)
    print("Ensemble v1 - XGBoost v10 + LightGBM v1")
    print("="*70)
    print("Weighted average of top 2 models")
    print()

    train_df, test_df, holidays = load_data()

    print("Creating features...")
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)
    train_df = train_df.dropna()

    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

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

    # Find best ensemble weights via CV
    print("\nFinding optimal ensemble weights...")
    print("="*70)

    tscv = TimeSeriesSplit(n_splits=5)
    best_weights = None
    best_cv = float('inf')

    # Try different weight combinations
    weight_combos = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.4, 0.6), (0.3, 0.7)]

    for xgb_weight, lgb_weight in weight_combos:
        print(f"\nTrying XGB:{xgb_weight:.1f} LGB:{lgb_weight:.1f}")
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # XGBoost (v10 params)
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            xgb_model = xgb.train(
                {'objective': 'reg:squarederror', 'max_depth': 8, 'learning_rate': 0.05,
                 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
                 'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'seed': 42},
                dtrain, num_boost_round=500, evals=[(dval, 'val')],
                early_stopping_rounds=50, verbose_eval=False
            )

            # LightGBM
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            lgb_model = lgb.train(
                {'objective': 'regression', 'num_leaves': 64, 'learning_rate': 0.05,
                 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'max_depth': 8,
                 'min_child_weight': 3, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'seed': 42},
                train_data, num_boost_round=500, valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            # Ensemble predictions
            xgb_pred = xgb_model.predict(dval)
            lgb_pred = lgb_model.predict(X_val)
            ensemble_pred = xgb_weight * xgb_pred + lgb_weight * lgb_pred
            ensemble_pred = np.clip(ensemble_pred, 0, None)

            # Score (masked)
            mask = y_val > 0
            rmsle = np.sqrt(mean_squared_log_error(y_val[mask], ensemble_pred[mask]))
            cv_scores.append(rmsle)

        mean_cv = np.mean(cv_scores)
        print(f"  Mean CV: {mean_cv:.6f}")

        if mean_cv < best_cv:
            best_cv = mean_cv
            best_weights = (xgb_weight, lgb_weight)

    print(f"\n{'='*70}")
    print(f"Best weights: XGB {best_weights[0]:.1f}, LGB {best_weights[1]:.1f}")
    print(f"Best CV: {best_cv:.6f}")
    print(f"\nComparison:")
    print(f"  XGBoost v10 alone: CV 0.370")
    print(f"  LightGBM v1 alone: CV 0.394")
    print(f"  Ensemble: CV {best_cv:.3f}")

    if best_cv < 0.370:
        print(f"\n✅ ENSEMBLE WINS! {((0.370 - best_cv) / 0.370 * 100):.1f}% better")
    else:
        print(f"\n⚠️ XGBoost v10 alone still better")

    # Train final models
    print("\nTraining final ensemble...")
    dtrain_full = xgb.DMatrix(X, label=y)
    final_xgb = xgb.train(
        {'objective': 'reg:squarederror', 'max_depth': 8, 'learning_rate': 0.05,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
         'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'seed': 42},
        dtrain_full, num_boost_round=500
    )

    final_lgb = lgb.train(
        {'objective': 'regression', 'num_leaves': 64, 'learning_rate': 0.05,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'max_depth': 8,
         'min_child_weight': 3, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'seed': 42},
        lgb.Dataset(X, label=y), num_boost_round=500
    )

    # Final predictions
    dtest = xgb.DMatrix(X_test)
    xgb_test_pred = final_xgb.predict(dtest)
    lgb_test_pred = final_lgb.predict(X_test)
    ensemble_test_pred = best_weights[0] * xgb_test_pred + best_weights[1] * lgb_test_pred
    ensemble_test_pred = np.clip(ensemble_test_pred, 0, None)

    # Save
    submission = pd.DataFrame({"id": test_df["id"], "sales": ensemble_test_pred})
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / f"ensemble_xgb_lgb_{best_cv:.4f}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\n✅ Saved: {submission_path.name}")
    print(f"📊 CV: {best_cv:.6f}")

    mlflow.set_experiment("store-sales-forecasting")
    with mlflow.start_run(run_name="ensemble_v1_xgb_lgb"):
        mlflow.log_param("model", "ensemble_xgb_lgb")
        mlflow.log_param("xgb_weight", best_weights[0])
        mlflow.log_param("lgb_weight", best_weights[1])
        mlflow.log_metric("cv_rmsle", best_cv)

    return best_cv


if __name__ == "__main__":
    cv_score = main()
