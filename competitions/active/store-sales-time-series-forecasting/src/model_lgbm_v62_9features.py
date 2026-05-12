#!/usr/bin/env python3
"""LightGBM v62 - 9 Features (Remove 3 Lowest Importance)

A/B TEST 1: Remove least important features
- Removed: is_weekend (12), Roll_mean_60 (191), Roll_mean_90 (298)
- Kept: 9 highest importance features
- Hypothesis: Less features = less noise = better generalization

Expected: CV ≤ 0.3572 (same or better than v19's 12 features)
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


def create_features_9(df, is_train=True, train_df=None):
    """9 features - removed is_weekend, Roll_mean_60, Roll_mean_90."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    df["day_of_week"] = df["date"].dt.dayofweek

    if is_train:
        df["Lag_3"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(3)
        df["Lag_7"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(7)

        # Only 7, 14, 30 rolling means (removed 60, 90)
        for window in [7, 14, 30]:
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
            for window in [7, 14, 30]:
                df[f"Roll_mean_{window}"] = df["lag_mean"].fillna(0)
            df["Roll_std_7"] = df["lag_std"].fillna(0)

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    print("="*70)
    print("LightGBM v62 - A/B Test 1: 9 Features")
    print("="*70)
    print("Testing: Remove 3 lowest importance features")
    print("Removed: is_weekend, Roll_mean_60, Roll_mean_90")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-ab-testing")

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    # Create 9 features
    print("Creating 9-feature set...")
    train_df = create_features_9(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    test_df = create_features_9(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)

    # Feature columns (9 total)
    feature_cols = [
        "onpromotion", "day_of_week", "Lag_3", "Lag_7",
        "Roll_mean_7", "Roll_mean_14", "Roll_mean_30",
        "Roll_std_7", "is_holiday"
    ]

    print(f"Features: {len(feature_cols)}")
    print(f"  v19 baseline: 12 features")
    print(f"  v62: {len(feature_cols)} features (-3)")
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
    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)
    holdout_cv = np.sqrt(mean_squared_log_error(y_val, preds))

    print(f"\n{'='*70}")
    print("A/B TEST 1 RESULTS")
    print(f"{'='*70}")
    print(f"v19 (12 features): CV 0.3572")
    print(f"v62 (9 features):  CV {holdout_cv:.4f}")

    if holdout_cv < 0.3572:
        improvement = (0.3572 - holdout_cv) / 0.3572 * 100
        print(f"✓ IMPROVED: {improvement:.2f}% better")
        decision = "KEEP - Continue to v63 (remove 2 more)"
    elif abs(holdout_cv - 0.3572) < 0.002:
        print(f"≈ SAME: Within 0.2% margin")
        decision = "KEEP - No degradation"
    else:
        decline = (holdout_cv - 0.3572) / 0.3572 * 100
        print(f"✗ WORSE: {decline:.2f}% degradation")
        decision = "REVERT - Keep v19's 12 features"

    print(f"\nDecision: {decision}")
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
    submission_path = SUBMISSION_DIR / f"lgbm_v62_9features_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # MLflow
    with mlflow.start_run(run_name="lgbm_v62_9features"):
        mlflow.log_params(params)
        mlflow.log_metric("holdout_cv", holdout_cv)
        mlflow.log_metric("vs_v19", holdout_cv - 0.3572)
        mlflow.log_metric("num_features", len(feature_cols))
        mlflow.log_param("removed_features", "is_weekend,Roll_mean_60,Roll_mean_90")
        mlflow.log_param("ab_test", "remove_3_lowest")
        mlflow.log_artifact(str(submission_path))

    # Hypothesis DB
    if HAS_DB:
        try:
            db = HypothesisDatabase()
            hyp_id = db.add_hypothesis(
                competition="store-sales-time-series-forecasting",
                experiment_id="v62",
                hypothesis="Removing 3 lowest importance features improves or maintains performance",
                rationale="A/B test: Less features may reduce noise. Removed is_weekend (12), Roll_mean_60 (191), Roll_mean_90 (298).",
                category="feature_engineering"
            )
            db.record_result(
                hypothesis_id=hyp_id,
                cv_score=holdout_cv,
                lb_score=None,
                baseline_cv=0.3572,
                succeeded=(holdout_cv <= 0.3572),
                params=params,
                metadata={
                    "num_features": len(feature_cols),
                    "removed": ["is_weekend", "Roll_mean_60", "Roll_mean_90"],
                    "ab_test": "remove_3_lowest"
                }
            )
        except Exception as e:
            print(f"Note: Could not log: {e}")

    return {
        "holdout_cv": holdout_cv,
        "vs_v19": holdout_cv - 0.3572,
        "decision": decision,
        "success": holdout_cv <= 0.3572
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
