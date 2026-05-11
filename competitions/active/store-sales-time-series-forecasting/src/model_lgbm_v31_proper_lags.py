#!/usr/bin/env python3
"""LightGBM v31 - Proper Lag Features for Test Set

CRITICAL FIX: Test set is 16 days AFTER training ends
- v1-v19 use lag 3/7 which are NaN in test set
- Top leaderboard uses lag 16+ (always available)

BREAKTHROUGH DISCOVERY:
My 39.5% CV-LB gap is from invalid features, not overfitting!
- CV 0.357: validation has lag 3/7 available
- LB 0.498: test set lags are NaN, model is blind

NEW FEATURES (all valid in test set):
- Lag 16, 30, 60, 365 days (safe, always available)
- Rolling means 20/30/45/60/90/120 shifted by 16
- Date features (day_of_week, is_weekend, is_holiday)
- NO lag 3/7 (those break in test)

Expected: CV may worsen slightly (less recent data), but LB should
dramatically improve because features actually work in test period.

Based on comprehensive guide by ekrembayar (2895 votes):
https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import mlflow

# Add parent directory to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

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


def create_features(df, is_train=True, train_df=None):
    """Create features valid in test set (16+ days future).

    Key insight: Test starts 16 days after training ends.
    Any lag < 16 will be NaN in test set!
    """
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Date features (always valid)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = (df["day_of_month"] <= 7).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 23).astype(int)

    if is_train:
        # Lag features >= 16 (safe for test set)
        df["Lag_16"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(16)
        df["Lag_30"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(30)
        df["Lag_60"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(60)
        df["Lag_365"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(365)

        # Rolling means shifted by 16 (available in test)
        for window in [20, 30, 45, 60, 90, 120]:
            df[f"SMA{window}_lag16"] = (
                df.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(16))
            )

        # Rolling std shifted by 16
        df["Roll_std_30_lag16"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=30, min_periods=1).std().shift(16))
        )

    else:
        # Test set: Use last known values from training
        if train_df is not None:
            # Get last 365 days for each store-family
            last_stats = (
                train_df.groupby(["store_nbr", "family"], observed=True)
                .tail(365)
                .groupby(["store_nbr", "family"], observed=True)
            )

            # Calculate lag features from end of training
            lag_features = []
            for (store, family), group in train_df.groupby(["store_nbr", "family"], observed=True):
                group = group.sort_values("date")
                sales = group["sales"].values

                # Lags from end of training
                lag_16 = sales[-16] if len(sales) >= 16 else sales[-1]
                lag_30 = sales[-30] if len(sales) >= 30 else sales[-1]
                lag_60 = sales[-60] if len(sales) >= 60 else sales[-1]
                lag_365 = sales[-365] if len(sales) >= 365 else np.mean(sales)

                # Rolling means from last known data
                sma_features = {}
                for window in [20, 30, 45, 60, 90, 120]:
                    if len(sales) >= window + 16:
                        # Mean of last window before lag 16
                        sma_features[f"SMA{window}_lag16"] = np.mean(sales[-(window+16):-16])
                    else:
                        sma_features[f"SMA{window}_lag16"] = np.mean(sales)

                # Rolling std
                if len(sales) >= 30 + 16:
                    std_val = np.std(sales[-(30+16):-16])
                else:
                    std_val = np.std(sales)

                lag_features.append({
                    "store_nbr": store,
                    "family": family,
                    "Lag_16": lag_16,
                    "Lag_30": lag_30,
                    "Lag_60": lag_60,
                    "Lag_365": lag_365,
                    **sma_features,
                    "Roll_std_30_lag16": std_val
                })

            lag_df = pd.DataFrame(lag_features)
            df = df.merge(lag_df, on=["store_nbr", "family"], how="left")

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    print("="*70)
    print("LightGBM v31 - Proper Lag Features (16+ days)")
    print("="*70)
    print("CRITICAL FIX: Using only lags valid in test set")
    print("Test starts 16 days after training - lag 3/7 are NaN!")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-lightgbm-proper-lags")

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    # Check test set timing
    train_end = train_df['date'].max()
    test_start = test_df['date'].min()
    gap = (test_start - train_end).days
    print(f"Training ends: {train_end}")
    print(f"Test starts: {test_start}")
    print(f"Gap: {gap} days ⚠️ MUST use lag >= {gap}")
    print()

    # Create features
    print("Creating proper lag features (16+)...")
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)

    # Feature columns
    exclude_cols = ["id", "date", "sales", "store_nbr", "family"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Features: {len(feature_cols)} total")
    print(f"  - Lags (16/30/60/365): 4")
    print(f"  - Rolling SMA shifted 16: 6")
    print(f"  - Rolling std: 1")
    print(f"  - Date features: {len(feature_cols) - 11}")
    print(f"\nFeature list: {feature_cols}")
    print()

    # 30-day holdout (same as v19 for comparison)
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

    # v19 params (proven architecture)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_estimators": 600,
        "random_state": 42,
        "verbosity": -1,
    }

    print("Training LightGBM with proper lags...")
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # Validation
    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)

    mask = y_val > 0
    if mask.sum() > 0:
        holdout_cv = np.sqrt(mean_squared_log_error(y_val[mask], preds[mask]))
    else:
        holdout_cv = float('inf')

    print(f"\nHoldout CV: {holdout_cv:.4f}")
    print(f"LightGBM v19 (broken lags): CV 0.3572, LB 0.498")
    print(f"Expected: CV may be worse, but LB should be much better")
    print()

    # Feature importance
    print("Top 10 features:")
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    print(feat_imp.head(10).to_string(index=False))
    print()

    # Train on full data
    print("Training on full dataset...")
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    # Generate predictions
    print("Generating test predictions...")
    X_test = test_df[feature_cols]
    predictions = final_model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    # Submission
    submission = pd.DataFrame({"id": test_df["id"], "sales": predictions})
    cv_str = f"{holdout_cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"lgbm_v31_proper_lags_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # MLflow
    with mlflow.start_run(run_name="lgbm_v31_proper_lags"):
        mlflow.log_params(params)
        mlflow.log_metric("holdout_cv", holdout_cv)
        mlflow.log_metric("vs_v19_cv", holdout_cv - 0.3572)
        mlflow.log_metric("num_features", len(feature_cols))
        mlflow.log_metric("min_lag", 16)
        mlflow.log_artifact(str(submission_path))

    # Langfuse
    if HAS_LANGFUSE:
        log_kaggle_model(
            name="v31",
            competition="store-sales-time-series-forecasting",
            model_type="LightGBM",
            cv_score=holdout_cv,
            lb_score=None,
            features=len(feature_cols),
            hyperparameters=params,
            status="proper_lags_16plus"
        )

    print()
    print("="*70)
    print("COMPLETE - Ready for Leaderboard Test")
    print("="*70)
    print(f"Holdout CV: {holdout_cv:.4f}")
    print(f"Submission: {submission_path.name}")
    print(f"\nThis should DRAMATICALLY improve LB score!")
    print(f"All features are valid in test set (16+ days future)")

    return {
        "holdout_cv": holdout_cv,
        "vs_v19_cv": holdout_cv - 0.3572,
        "success": True
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
