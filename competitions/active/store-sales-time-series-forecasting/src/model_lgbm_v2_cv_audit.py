#!/usr/bin/env python3
"""LightGBM v2 - CV Audit

Investigating LightGBM's potential (LB: 0.572) with proper TimeSeriesSplit CV.
The Analyst noted that LightGBM is underexplored despite promising LB scores.
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
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Top 12 features from v11 analysis
TOP_FEATURES = [
    "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "is_weekend",
    "Roll_mean_60", "day_of_week", "is_holiday", "Lag_3",
    "Roll_mean_90", "onpromotion", "Lag_7", "Roll_std_7"
]


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
    print("LightGBM v2 - CV Audit")
    print("="*70)
    print("Testing LightGBM with proper TimeSeriesSplit CV")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-lightgbm")

    train_df, test_df, holidays = load_data()

    print("Creating features...")
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)

    X = train_df[TOP_FEATURES]
    y = train_df["sales"]

    # LightGBM parameters (balanced for regularization)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "n_estimators": 1000,
        "random_state": 42,
        "verbosity": -1,
    }

    print("\nLightGBM settings:")
    print(f"  Learning rate: {params['learning_rate']}")
    print(f"  Num leaves: {params['num_leaves']}")
    print(f"  Max depth: {params['max_depth']}")
    print(f"  L1 (alpha): {params['reg_alpha']}")
    print(f"  L2 (lambda): {params['reg_lambda']}")

    # TimeSeriesSplit cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    print(f"\nRunning {tscv.n_splits}-fold TimeSeriesSplit CV...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        preds = model.predict(X_val)
        preds = np.maximum(preds, 0)

        # Mask zero true values for RMSLE
        mask = y_val > 0
        if mask.sum() > 0:
            score = np.sqrt(mean_squared_log_error(y_val[mask], preds[mask]))
        else:
            score = float('inf')

        cv_scores.append(score)
        print(f"  Fold {fold}: {score:.4f}")

    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    print(f"\nCV Score: {mean_cv:.4f} ± {std_cv:.4f}")

    # Train on full data
    print("\nTraining on full dataset...")
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    # Generate predictions
    print("Generating test predictions...")
    X_test = test_df[TOP_FEATURES]
    predictions = final_model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    # Create submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": predictions
    })

    cv_str = f"{mean_cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"lgbm_v2_cv_audit_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # Log to MLflow
    with mlflow.start_run(run_name="lgbm_v2_cv_audit"):
        mlflow.log_params(params)
        mlflow.log_metric("cv_score", mean_cv)
        mlflow.log_metric("cv_std", std_cv)
        mlflow.log_artifact(str(submission_path))

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"CV: {mean_cv:.4f}")
    print(f"Submission: {submission_path.name}")
    print("\nComparison to best XGBoost (v3): 0.3150")
    if mean_cv < 0.315:
        print("✓ LightGBM outperforms XGBoost!")
    else:
        print("✗ XGBoost still better")

    return {"cv_score": mean_cv, "lb_score": None, "success": True}


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
