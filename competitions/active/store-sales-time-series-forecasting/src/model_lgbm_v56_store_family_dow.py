#!/usr/bin/env python3
"""LightGBM v56 - Store-Family-DayOfWeek Features

HYPOTHESIS: Different stores/families have different weekly patterns
- AUTOMOTIVE sells more on weekends
- GROCERY has weekday peaks
- Store-specific patterns (urban vs rural)

NEW FEATURES:
- store_family_dow_mean: Historical average for this (store, family, dayofweek)
- store_family_dow_std: Variance for this combination

EXPECTED: CV < 0.35 (3-5% improvement over v19's 0.357)

Research: "Recent Winning Strategies: Average for specific day highly predictive"
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import mlflow

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

try:
    from core.hypothesis_db import HypothesisDatabase
    HAS_DB = True
except ImportError:
    HAS_DB = False

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


def create_features_v1(df, is_train=True, train_df=None):
    """v1 baseline features."""
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
        # Test set: use training statistics
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


def add_store_family_dow_features(df, is_train=True, train_df=None):
    """NEW: Add store-family-dayofweek specific features."""
    df["day_of_week"] = df["date"].dt.dayofweek

    if is_train:
        # Calculate historical mean/std for each (store, family, dow) combination
        # Use expanding window to avoid lookahead bias
        df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

        # Shift sales by 1 to avoid using current day
        df["sales_shifted"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(1)

        # Calculate expanding mean/std for each dow
        df["store_family_dow_mean"] = (
            df.groupby(["store_nbr", "family", "day_of_week"], observed=True)["sales_shifted"]
            .transform(lambda x: x.expanding().mean())
        )
        df["store_family_dow_std"] = (
            df.groupby(["store_nbr", "family", "day_of_week"], observed=True)["sales_shifted"]
            .transform(lambda x: x.expanding().std())
        )

        # Drop temporary column
        df = df.drop(columns=["sales_shifted"])

    else:
        # Test set: use full training history for each (store, family, dow)
        if train_df is not None:
            dow_stats = (
                train_df.groupby(["store_nbr", "family", "day_of_week"], observed=True)["sales"]
                .agg(["mean", "std"])
                .reset_index()
            )
            dow_stats.columns = ["store_nbr", "family", "day_of_week", "store_family_dow_mean", "store_family_dow_std"]

            df = df.merge(
                dow_stats,
                on=["store_nbr", "family", "day_of_week"],
                how="left"
            )

    # Fill NaN with 0 (for rare combinations)
    df["store_family_dow_mean"] = df["store_family_dow_mean"].fillna(0)
    df["store_family_dow_std"] = df["store_family_dow_std"].fillna(0)

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    print("="*70)
    print("LightGBM v56 - Store-Family-DayOfWeek Features")
    print("="*70)
    print("v1 features + store-family-dow patterns")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-lightgbm-v56")

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()
    print()

    # Create v1 features
    print("Creating v1 features...")
    train_df = create_features_v1(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)

    # Add NEW store-family-dow features
    print("Adding store-family-dayofweek features...")
    train_df = add_store_family_dow_features(train_df, is_train=True)
    train_df = train_df.dropna()

    # Create test features
    test_df = create_features_v1(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)
    test_df = add_store_family_dow_features(test_df, is_train=False, train_df=train_df)

    # Feature columns
    exclude_cols = ["id", "date", "sales", "store_nbr", "family"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"Features: {len(feature_cols)}")
    print(f"  v1 features: 12")
    print(f"  NEW features: {len(feature_cols) - 12}")
    print(f"Feature list: {feature_cols}")
    print()

    # 30-day holdout
    cutoff_date = train_df["date"].max() - pd.Timedelta(days=30)
    val_mask = train_df["date"] >= cutoff_date
    train_mask = ~val_mask

    X = train_df[feature_cols]
    y = train_df["sales"]
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    print()

    # v19 params
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

    print("Training LightGBM...")
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    # Validation
    print("\nValidating...")
    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)
    holdout_cv = np.sqrt(mean_squared_log_error(y_val, preds))
    print(f"Holdout CV: {holdout_cv:.4f}")
    print(f"v19 baseline: 0.3572")
    print(f"Improvement: {(0.3572 - holdout_cv) / 0.3572 * 100:+.2f}%")
    print()

    # Feature importance
    print("Top 10 features:")
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    for i, row in importances.head(10).iterrows():
        marker = "NEW" if "dow" in row["feature"] else ""
        print(f"  {row['feature']:<30} {row['importance']:>10.0f}  {marker}")
    print()

    # Train on full data
    print("Training on full dataset...")
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    # Test predictions
    print("Generating test predictions...")
    X_test = test_df[feature_cols]
    test_preds = final_model.predict(X_test)
    test_preds = np.maximum(test_preds, 0)

    # Save submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": test_preds
    })

    cv_str = f"{holdout_cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"lgbm_v56_store_family_dow_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSaved: {submission_path}")

    # MLflow
    with mlflow.start_run(run_name="lgbm_v56_store_family_dow"):
        mlflow.log_params(params)
        mlflow.log_metric("holdout_cv", holdout_cv)
        mlflow.log_metric("vs_v19", holdout_cv - 0.3572)
        mlflow.log_metric("num_features", len(feature_cols))
        mlflow.log_param("new_features", "store_family_dow_mean,store_family_dow_std")
        mlflow.log_artifact(str(submission_path))

    # Hypothesis DB
    if HAS_DB:
        try:
            db = HypothesisDatabase()
            hyp_id = db.add_hypothesis(
                competition="store-sales-time-series-forecasting",
                experiment_id="v56",
                hypothesis="Store-family-dayofweek features capture weekly patterns (3-5% improvement)",
                rationale="Different stores/families have different weekly patterns. AUTOMOTIVE peaks weekends, GROCERY weekdays. Research shows 'average for specific day' is highly predictive.",
                category="feature_engineering"
            )
            db.record_result(
                hypothesis_id=hyp_id,
                cv_score=holdout_cv,
                lb_score=None,
                baseline_cv=0.3572,
                succeeded=(holdout_cv < 0.3572),
                params=params,
                metadata={
                    "new_features": ["store_family_dow_mean", "store_family_dow_std"],
                    "total_features": len(feature_cols)
                }
            )
            print("✓ Logged to hypothesis database")
        except Exception as e:
            print(f"Note: Could not log: {e}")

    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Holdout CV: {holdout_cv:.4f}")
    print(f"v19 baseline: 0.3572")
    if holdout_cv < 0.3572:
        print(f"✓ IMPROVEMENT: {(0.3572 - holdout_cv) / 0.3572 * 100:.2f}% better")
    else:
        print(f"✗ WORSE: {(holdout_cv - 0.3572) / 0.3572 * 100:.2f}% worse")
    print(f"\nSubmission: {submission_path.name}")

    return {
        "holdout_cv": holdout_cv,
        "vs_v19": holdout_cv - 0.3572,
        "num_features": len(feature_cols),
        "success": holdout_cv < 0.3572
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
