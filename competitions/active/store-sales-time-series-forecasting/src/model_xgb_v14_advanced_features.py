#!/usr/bin/env python3
"""XGBoost v14 - Advanced Feature Engineering

Based on Kaggle best practices research:
1. Exponential weighted moving averages (EWMA)
2. More temporal features (week, month, year)
3. Interaction features (promotion × holiday, store × day_of_week)
4. Store/family clustering via encoding
5. Difference features (week-over-week change)

Uses 30-day holdout validation for accurate LB prediction.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import mlflow

DATA_DIR = Path("data/store-sales-time-series-forecasting")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "int16", "family": "category"},
        parse_dates=["date"],
    )
    test = pd.read_csv(
        DATA_DIR / "test.csv",
        dtype={"store_nbr": "int16", "family": "category"},
        parse_dates=["date"],
    )
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    return train, test, holidays, stores


def create_base_features(df):
    """Basic time features."""
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(np.int16)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year - 2013  # Normalize from 0
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(np.int8)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(np.int8)
    return df


def create_lag_features(df, is_train=True):
    """Lag and rolling features."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    if is_train:
        # Lags
        for lag in [3, 7, 14]:
            df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

        # Rolling means
        for window in [7, 14, 30, 60]:
            df[f"Roll_mean_{window}"] = (
                df.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )

        # Rolling std
        df["Roll_std_7"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=7, min_periods=1).std())
        )

        # EWMA - exponential weighted moving average
        for span in [7, 14, 30]:
            df[f"EWMA_{span}"] = (
                df.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            )

        # Week-over-week difference
        df["WoW_diff"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].diff(7)

    return df


def create_interaction_features(df):
    """Interaction features."""
    # Promotion × holiday
    df["promo_holiday"] = (df["onpromotion"] * df["is_holiday"]).astype(np.int8)

    # Weekend × promotion
    df["weekend_promo"] = (df["is_weekend"] * df["onpromotion"]).astype(np.int8)

    # Store-family encoding (captures store×family patterns)
    df["store_family"] = df["store_nbr"].astype(str) + "_" + df["family"].astype(str)
    le = LabelEncoder()
    df["store_family_enc"] = le.fit_transform(df["store_family"])
    df = df.drop(columns=["store_family"])

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(np.int8)
    return df


def add_store_metadata(df, stores):
    """Add store type and cluster."""
    df = df.merge(stores[["store_nbr", "type", "cluster"]], on="store_nbr", how="left")

    # Encode categorical store features
    le_type = LabelEncoder()
    le_cluster = LabelEncoder()
    df["store_type_enc"] = le_type.fit_transform(df["type"].astype(str))
    df["store_cluster_enc"] = le_cluster.fit_transform(df["cluster"].astype(str))
    df = df.drop(columns=["type", "cluster"])

    return df


def prepare_test_features(test_df, train_df):
    """Create test features using train statistics."""
    test_df = test_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Use last 30 days of train for lag approximation
    last_stats = (
        train_df.groupby(["store_nbr", "family"], observed=True)
        .tail(30)
        .groupby(["store_nbr", "family"], observed=True)["sales"]
        .agg(["mean", "std"])
        .reset_index()
    )
    last_stats.columns = ["store_nbr", "family", "lag_mean", "lag_std"]
    test_df = test_df.merge(last_stats, on=["store_nbr", "family"], how="left")

    # Fill lag features with stats
    for lag in [3, 7, 14]:
        test_df[f"Lag_{lag}"] = test_df["lag_mean"].fillna(0)

    for window in [7, 14, 30, 60]:
        test_df[f"Roll_mean_{window}"] = test_df["lag_mean"].fillna(0)

    test_df["Roll_std_7"] = test_df["lag_std"].fillna(0)

    for span in [7, 14, 30]:
        test_df[f"EWMA_{span}"] = test_df["lag_mean"].fillna(0)

    test_df["WoW_diff"] = 0  # Unknown for test

    test_df = test_df.drop(columns=["lag_mean", "lag_std"], errors="ignore")

    return test_df


def main():
    print("="*70)
    print("XGBoost v14 - Advanced Feature Engineering")
    print("="*70)
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-xgboost-holdout")

    # Load data
    print("Loading data...")
    train_df, test_df, holidays, stores = load_data()

    # Feature engineering - train
    print("Creating advanced features for train...")
    train_df = create_base_features(train_df)
    train_df = add_holidays(train_df, holidays)
    train_df = add_store_metadata(train_df, stores)
    train_df = create_lag_features(train_df, is_train=True)
    train_df = create_interaction_features(train_df)
    train_df = train_df.dropna()

    # Feature engineering - test
    print("Creating advanced features for test...")
    test_df = create_base_features(test_df)
    test_df = add_holidays(test_df, holidays)
    test_df = add_store_metadata(test_df, stores)
    test_df = prepare_test_features(test_df, train_df)
    test_df = create_interaction_features(test_df)

    # Get feature columns
    exclude_cols = ["id", "date", "sales", "store_nbr", "family"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Feature groups:")
    print(f"  - Temporal: {len([c for c in feature_cols if any(t in c for t in ['day', 'week', 'month', 'year'])])}")
    print(f"  - Lag: {len([c for c in feature_cols if 'Lag' in c])}")
    print(f"  - Rolling: {len([c for c in feature_cols if 'Roll' in c or 'EWMA' in c])}")
    print(f"  - Interaction: {len([c for c in feature_cols if any(i in c for i in ['promo_', 'weekend_'])])}")
    print(f"  - Store/Family: {len([c for c in feature_cols if 'store' in c.lower() or 'family' in c.lower()])}")
    print()

    # 30-day holdout split
    print("Creating 30-day holdout split...")
    cutoff_date = train_df['date'].max() - pd.Timedelta(days=30)
    val_mask = train_df['date'] >= cutoff_date
    train_mask = ~val_mask

    X = train_df[feature_cols]
    y = train_df["sales"]

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    print(f"Train: {len(X_train):,} rows")
    print(f"Val:   {len(X_val):,} rows")
    print()

    # Model parameters (similar to v1 baseline but with more capacity for extra features)
    params = {
        "objective": "reg:squaredlogerror",
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.04,
        "max_depth": 7,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 800,
        "random_state": 42,
        "verbosity": 0,
    }

    print("Training model...")
    model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Validation score
    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)

    mask = y_val > 0
    if mask.sum() > 0:
        holdout_cv = np.sqrt(mean_squared_log_error(y_val[mask], preds[mask]))
    else:
        holdout_cv = float('inf')

    print(f"\nHoldout CV: {holdout_cv:.4f}")
    print(f"Baseline (v1): 0.4842")

    if holdout_cv < 0.4842:
        improvement = 0.4842 - holdout_cv
        print(f"✓ IMPROVEMENT: -{improvement:.4f} ({improvement/0.4842*100:.1f}%)")
    else:
        decline = holdout_cv - 0.4842
        print(f"✗ Declined: +{decline:.4f}")
    print()

    # Train on full data
    print("Training on full dataset...")
    final_model = xgb.XGBRegressor(**params)
    final_model.fit(X, y, verbose=False)

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 15 features:")
    for i, row in importance.head(15).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")
    print()

    # Generate predictions
    print("Generating test predictions...")
    X_test = test_df[feature_cols]
    predictions = final_model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    # Create submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": predictions
    })

    cv_str = f"{holdout_cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"xgb_v14_advanced_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # Log to MLflow
    with mlflow.start_run(run_name="xgb_v14_advanced"):
        mlflow.log_params(params)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_metric("holdout_cv", holdout_cv)
        mlflow.log_metric("improvement_vs_v1", 0.4842 - holdout_cv)
        mlflow.log_artifact(str(submission_path))

        # Log top features
        for i, row in importance.head(10).iterrows():
            mlflow.log_metric(f"feat_{row['feature']}", row['importance'])

    print()
    print("="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Holdout CV: {holdout_cv:.4f}")
    print(f"Submission: {submission_path.name}")

    return {
        "holdout_cv": holdout_cv,
        "num_features": len(feature_cols),
        "top_features": importance.head(10)["feature"].tolist(),
        "success": True
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
