#!/usr/bin/env python3
"""LightGBM v30 - External Data Integration

STRATEGY: Add safe external data sources
- Oil prices: Ecuador's economy depends on oil
- Store metadata: city, state, type, cluster
- Proper encoding (avoid v14-v17 label encoder bug)
- Expected: 2-5% improvement

External data sources:
- oil.csv: Daily oil prices (WTI)
- stores.csv: Store location and characteristics

vs v19: Same features + external data
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

    # External data
    oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA_DIR / "stores.csv")

    return train, test, holidays, oil, stores


def add_oil_features(df, oil):
    """Add oil price features (lagged and rolling)."""
    # Merge oil prices
    df = df.merge(oil, on="date", how="left")

    # Fill missing oil prices with forward fill
    df["dcoilwtico"] = df["dcoilwtico"].fillna(method="ffill").fillna(method="bfill")

    # Sort for lag features
    df = df.sort_values("date")

    # Lag features
    df["oil_lag_1"] = df["dcoilwtico"].shift(1)
    df["oil_lag_7"] = df["dcoilwtico"].shift(7)
    df["oil_lag_30"] = df["dcoilwtico"].shift(30)

    # Rolling statistics
    df["oil_roll_mean_7"] = df["dcoilwtico"].rolling(window=7, min_periods=1).mean()
    df["oil_roll_mean_30"] = df["dcoilwtico"].rolling(window=30, min_periods=1).mean()
    df["oil_roll_std_7"] = df["dcoilwtico"].rolling(window=7, min_periods=1).std()

    # Oil price change
    df["oil_pct_change"] = df["dcoilwtico"].pct_change().fillna(0)

    return df


def add_store_features(df, stores):
    """Add store metadata features (properly encoded)."""
    # Merge store data
    df = df.merge(stores, on="store_nbr", how="left")

    # One-hot encode categorical store features (SAFE - no leakage)
    # Don't use label encoder - caused v14-v17 bugs
    df = pd.get_dummies(df, columns=["city", "state", "type"], prefix=["city", "state", "type"])

    # Cluster is numeric, keep as is
    df["cluster"] = df["cluster"].fillna(0).astype(int)

    return df


def create_features(df, is_train=True, train_df=None, oil=None, stores=None):
    """Create v1 features + external data."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Add external data first
    if oil is not None:
        df = add_oil_features(df, oil)
    if stores is not None:
        df = add_store_features(df, stores)

    # Original v1 features
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
    print("LightGBM v30 - External Data Integration")
    print("="*70)
    print("Adding oil prices + store metadata to v1 features")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-lightgbm-external")

    # Load data
    print("Loading data + external sources...")
    train_df, test_df, holidays, oil, stores = load_data()

    # Create features
    print("Creating v1 + external features...")
    train_df = create_features(train_df, is_train=True, oil=oil, stores=stores)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    test_df = create_features(test_df, is_train=False, train_df=train_df, oil=oil, stores=stores)
    test_df = add_holidays(test_df, holidays)

    # Feature columns (exclude identifiers and targets)
    exclude_cols = ["id", "date", "sales", "store_nbr", "family", "lag_mean", "lag_std"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols and c != "dcoilwtico"]

    print(f"Features: {len(feature_cols)} total")
    print(f"  - v1 base: 12")
    print(f"  - Oil: 7")
    print(f"  - Store: ~{len(feature_cols) - 19}")
    print()

    # 30-day holdout
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

    # v19 params (proven)
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

    print("Training LightGBM with external data...")
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
    print(f"LightGBM v19 (baseline): 0.3572")

    if holdout_cv < 0.3572:
        improvement = 0.3572 - holdout_cv
        print(f"✓ IMPROVEMENT: -{improvement:.4f} ({improvement/0.3572*100:.1f}%)")
    else:
        decline = holdout_cv - 0.3572
        print(f"✗ Declined: +{decline:.4f}")
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
    submission_path = SUBMISSION_DIR / f"lgbm_v30_external_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # MLflow
    with mlflow.start_run(run_name="lgbm_v30_external"):
        mlflow.log_params(params)
        mlflow.log_metric("holdout_cv", holdout_cv)
        mlflow.log_metric("vs_v19", holdout_cv - 0.3572)
        mlflow.log_metric("num_features", len(feature_cols))
        mlflow.log_artifact(str(submission_path))

    # Langfuse
    if HAS_LANGFUSE:
        log_kaggle_model(
            name="v30",
            competition="store-sales-time-series-forecasting",
            model_type="LightGBM",
            cv_score=holdout_cv,
            lb_score=None,
            features=len(feature_cols),
            hyperparameters=params,
            status="external_data"
        )

    print()
    print("="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Holdout CV: {holdout_cv:.4f}")
    print(f"Submission: {submission_path.name}")

    return {
        "holdout_cv": holdout_cv,
        "vs_v19": holdout_cv - 0.3572,
        "success": True
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
