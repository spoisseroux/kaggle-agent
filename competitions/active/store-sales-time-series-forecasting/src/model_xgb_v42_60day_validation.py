#!/usr/bin/env python3
"""XGBoost v42 - 60-Day Validation Window

STRATEGY: Reduce validation bias by using longer validation period
- v19/v41: 30-day validation (Aug 2017 only - peak sales, 33% higher than mean)
- v42: 60-day validation (Jul-Aug 2017 - more representative period)
- More balanced train/val ratio (3.5% vs 1.8%)
- Reduces optimization bias toward peak sales period

Expected: Lower CV but better generalization to test set
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from core.langfuse_logger import log_kaggle_model
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False

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


def create_features(df, is_train=True, train_df=None, holidays=None):
    """v1 features"""
    df = df.copy()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if holidays is not None:
        holiday_dates = set(holidays["date"].dt.date)
        df["is_holiday"] = df["date"].dt.date.isin(holiday_dates).astype(int)
    else:
        df["is_holiday"] = 0

    if is_train:
        for lag in [3, 7]:
            df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

        for window in [7, 14, 30, 60, 90]:
            df[f"Roll_mean_{window}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

        df["Roll_std_7"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        )
    else:
        if train_df is not None:
            lag_mean = train_df.groupby(["store_nbr", "family"], observed=True)["sales"].mean()
            df = df.merge(
                lag_mean.rename("lag_mean").reset_index(),
                on=["store_nbr", "family"],
                how="left"
            )

            for lag in [3, 7]:
                df[f"Lag_{lag}"] = df["lag_mean"]

            for window in [7, 14, 30, 60, 90]:
                df[f"Roll_mean_{window}"] = df["lag_mean"]

            df["Roll_std_7"] = df["lag_mean"]

            df = df.drop(columns=["lag_mean"])

    return df


def main():
    print("="*70)
    print("XGBoost v42 - 60-Day Validation Window")
    print("="*70)
    print("Strategy: Longer validation period to reduce bias")
    print("30-day window (v41) vs 60-day window (v42)")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-xgboost-v1")

    print("Loading data...")
    train_df, test_df, holidays = load_data()

    print("Creating v1 features...")
    train_df = create_features(train_df, is_train=True, holidays=holidays)
    train_df = train_df.dropna()

    test_df = create_features(test_df, is_train=False, train_df=train_df, holidays=holidays)

    # 60-day validation window (vs 30-day in v41)
    cutoff_date = train_df['date'].max() - pd.Timedelta(days=60)
    val_mask = train_df['date'] >= cutoff_date
    train_mask = ~val_mask

    exclude_cols = ["id", "date", "sales", "store_nbr", "family"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X = train_df[feature_cols]
    y = train_df["sales"]
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    print(f"Train: {len(X_train):,} ({len(X_train)/(len(X_train)+len(X_val))*100:.1f}%)")
    print(f"Val:   {len(X_val):,} ({len(X_val)/(len(X_train)+len(X_val))*100:.1f}%)")
    print(f"Features: {len(feature_cols)}")
    print()

    # Check validation period sales distribution
    val_dates = train_df[val_mask]['date']
    print(f"Validation period: {val_dates.min().date()} to {val_dates.max().date()}")
    print(f"Validation mean sales: {y_val.mean():.2f}")
    print(f"Training mean sales: {y_train.mean():.2f}")
    print(f"Ratio: {y_val.mean() / y_train.mean():.2%}")
    print()

    # Historical parameters (from v1/v3)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    print("XGBoost parameters:")
    for k, v in params.items():
        if k not in ["n_jobs", "eval_metric", "tree_method"]:
            print(f"  {k}: {v}")
    print()

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

    evals = [(dval, 'validation')]

    with mlflow.start_run(run_name="xgb_v42_60day_val"):
        mlflow.log_params(params)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("validation_days", 60)

        print("Training XGBoost...")
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=100
        )

        val_preds = model.predict(dval)
        val_preds = np.maximum(val_preds, 0)

        cv_score = np.sqrt(mean_squared_log_error(y_val, val_preds))

        print()
        print(f"Holdout CV (60-day): {cv_score:.4f}")
        print(f"v41 (30-day): 0.4554")
        print(f"v19 (LightGBM 30-day): 0.3572")
        print()

        mlflow.log_metric("holdout_cv", cv_score)
        mlflow.log_metric("vs_v41", cv_score - 0.4554)
        mlflow.log_metric("vs_v19", cv_score - 0.3572)

    print("Generating test predictions...")
    X_test = test_df[feature_cols]
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    test_preds = model.predict(dtest)
    test_preds = np.maximum(test_preds, 0)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": test_preds
    })

    filename = f"xgb_v42_60day_{cv_score:.5f}.csv".replace(".", "")
    filepath = SUBMISSION_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"Submission saved: {filename}")
    print(f"Mean prediction: {test_preds.mean():.2f}")
    print()

    if HAS_LANGFUSE:
        try:
            log_kaggle_model(
                competition="store-sales-time-series-forecasting",
                model_name="XGBoost v42 60-day",
                cv_score=cv_score,
                model_type="xgboost",
                params=params,
                features=feature_cols
            )
        except Exception:
            pass

    print("="*70)
    print("ANALYSIS:")
    print(f"Longer validation window (60 vs 30 days)")
    print(f"Should reduce bias toward peak sales period")
    print(f"CV may be higher but LB should improve")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
