#!/usr/bin/env python3
"""XGBoost v40 - Simple Parameters + v1 Features

STRATEGY: Return to XGBoost with proven approach
- Competition history: XGBoost has 9.3% CV-LB gap vs LightGBM's 39.5%
- Use v1 features (proven best)
- SIMPLE parameters to avoid overfitting
- Let early stopping find optimal iterations

Features (v1 - same as v19):
- Lags: 3, 7
- Rolling means: 7, 14, 30, 60, 90
- Rolling std: 7
- Temporal: day_of_week, is_weekend, is_holiday
- Promotion: onpromotion

Parameters:
- max_depth: 4 (shallow, conservative)
- learning_rate: 0.05 (moderate)
- subsample: 0.8 (prevent overfitting)
- colsample_bytree: 0.8 (feature sampling)
- No explicit regularization (let tree structure handle it)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import mlflow

# Add parent directory to path
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
    """v1 features - proven best feature set"""
    df = df.copy()

    # Temporal features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Holidays
    if holidays is not None:
        holiday_dates = set(holidays["date"].dt.date)
        df["is_holiday"] = df["date"].dt.date.isin(holiday_dates).astype(int)
    else:
        df["is_holiday"] = 0

    # Lags and rolling features (only for training data with sales column)
    if is_train:
        # Lags (3, 7 days)
        for lag in [3, 7]:
            df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

        # Rolling means (7, 14, 30, 60, 90 days)
        for window in [7, 14, 30, 60, 90]:
            df[f"Roll_mean_{window}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

        # Rolling std (7 days)
        df["Roll_std_7"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        )
    else:
        # For test set, fill with store-family mean from training
        if train_df is not None:
            lag_mean = train_df.groupby(["store_nbr", "family"], observed=True)["sales"].mean()
            df = df.merge(
                lag_mean.rename("lag_mean").reset_index(),
                on=["store_nbr", "family"],
                how="left"
            )

            # Create lag and rolling columns filled with mean
            for lag in [3, 7]:
                df[f"Lag_{lag}"] = df["lag_mean"]

            for window in [7, 14, 30, 60, 90]:
                df[f"Roll_mean_{window}"] = df["lag_mean"]

            df["Roll_std_7"] = df["lag_mean"]

            df = df.drop(columns=["lag_mean"])

    return df


def main():
    print("="*70)
    print("XGBoost v40 - Simple Parameters + v1 Features")
    print("="*70)
    print("Strategy: Conservative XGBoost to avoid overfitting")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-xgboost-v1")

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    # Create features
    print("Creating v1 features...")
    train_df = create_features(train_df, is_train=True, holidays=holidays)
    train_df = train_df.dropna()

    test_df = create_features(test_df, is_train=False, train_df=train_df, holidays=holidays)

    # 30-day holdout validation
    cutoff_date = train_df['date'].max() - pd.Timedelta(days=30)
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

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"Features: {len(feature_cols)}")
    print()

    # XGBoost parameters - SIMPLE and CONSERVATIVE
    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,              # Shallow trees
        "learning_rate": 0.05,       # Moderate learning rate
        "subsample": 0.8,            # Row sampling (prevent overfitting)
        "colsample_bytree": 0.8,     # Column sampling (prevent overfitting)
        "min_child_weight": 3,       # Minimum samples per leaf
        "random_state": 42,
        "n_jobs": -1,
    }

    print("XGBoost parameters:")
    for k, v in params.items():
        if k != "n_jobs":
            print(f"  {k}: {v}")
    print()

    # Train
    print("Training XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

    evals = [(dval, 'validation')]

    with mlflow.start_run(run_name="xgb_v40_simple"):
        mlflow.log_params(params)
        mlflow.log_param("num_features", len(feature_cols))

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=100
        )

        # Validation predictions
        val_preds = model.predict(dval)
        val_preds = np.maximum(val_preds, 0)  # Clip negatives

        cv_score = np.sqrt(mean_squared_log_error(y_val, val_preds))

        print()
        print(f"Holdout CV: {cv_score:.4f}")
        print(f"v19 (LightGBM) CV: 0.3572")
        print(f"Improvement: {(0.3572 - cv_score) / 0.3572 * 100:.2f}%")
        print()

        mlflow.log_metric("holdout_cv", cv_score)
        mlflow.log_metric("vs_v19", cv_score - 0.3572)

    # Test predictions
    print("Generating test predictions...")
    X_test = test_df[feature_cols]
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    test_preds = model.predict(dtest)
    test_preds = np.maximum(test_preds, 0)

    # Create submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": test_preds
    })

    filename = f"xgb_v40_simple_{cv_score:.5f}.csv".replace(".", "")
    filepath = SUBMISSION_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"Submission saved: {filename}")
    print(f"Mean prediction: {test_preds.mean():.2f}")
    print(f"Median prediction: {np.median(test_preds):.2f}")
    print()

    # Log to Langfuse
    if HAS_LANGFUSE:
        try:
            log_kaggle_model(
                competition="store-sales-time-series-forecasting",
                model_name="XGBoost v40 Simple",
                cv_score=cv_score,
                model_type="xgboost",
                params=params,
                features=feature_cols
            )
        except Exception as e:
            print(f"Langfuse logging failed: {e}")

    print("="*70)
    print("NEXT STEPS:")
    print("1. Review CV score vs v19")
    print("2. If CV is good, submit to Kaggle for LB score")
    print("3. Compare XGBoost CV-LB gap vs LightGBM")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
