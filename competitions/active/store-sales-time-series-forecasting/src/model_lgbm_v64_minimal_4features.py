#!/usr/bin/env python3
"""LightGBM v64 - Minimal 4 Features (Ablation Study Winner)

DISCOVERY: Ablation study shows only 4 features actually help!
Removing other 8 features IMPROVES performance.

CRITICAL FEATURES:
1. Roll_std_7     (45.7% degradation if removed)
2. Roll_mean_7    (4.99% degradation if removed)
3. is_holiday     (0.62% degradation if removed)
4. Roll_mean_60   (0.42% degradation if removed)

NOISE FEATURES (improve when removed):
- Lag_3 (-4.21% when removed)
- Lag_7 (-2.79%)
- onpromotion (-2.71%)
- day_of_week (-2.29%)
- Roll_mean_14, 30, 90, is_weekend

Expected: Best CV yet if ablation study is correct
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


def create_features_4(df, is_train=True, train_df=None):
    """Only 4 critical features."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    if is_train:
        # Only Roll_mean_7, Roll_mean_60, Roll_std_7
        for window in [7, 60]:
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

            for window in [7, 60]:
                df[f"Roll_mean_{window}"] = df["lag_mean"].fillna(0)
            df["Roll_std_7"] = df["lag_std"].fillna(0)

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    print("="*70)
    print("LightGBM v64 - Minimal 4 Features")
    print("="*70)
    print("Based on ablation study - only features that help")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-minimal")

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    # Create minimal 4 features
    print("Creating 4-feature set...")
    train_df = create_features_4(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    test_df = create_features_4(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)

    # Only 4 features
    feature_cols = ["Roll_mean_7", "Roll_mean_60", "Roll_std_7", "is_holiday"]

    print(f"Features: {len(feature_cols)}")
    print(f"  v19 baseline: 12 features")
    print(f"  v64: {len(feature_cols)} features")
    for feat in feature_cols:
        print(f"    - {feat}")
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
    print("ABLATION TEST RESULTS")
    print(f"{'='*70}")
    print(f"v19 (12 features): CV 0.3572")
    print(f"v64 (4 features):  CV {holdout_cv:.4f}")

    if holdout_cv < 0.3572:
        improvement = (0.3572 - holdout_cv) / 0.3572 * 100
        print(f"✓ IMPROVED: {improvement:.2f}% better")
        decision = "BREAKTHROUGH - Simpler is MUCH better"
    else:
        decline = (holdout_cv - 0.3572) / 0.3572 * 100
        print(f"✗ WORSE: {decline:.2f}% degradation")
        decision = "Ablation study misleading - keep 12 features"

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
    submission_path = SUBMISSION_DIR / f"lgbm_v64_minimal_4features_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # MLflow
    with mlflow.start_run(run_name="lgbm_v64_minimal_4features"):
        mlflow.log_params(params)
        mlflow.log_metric("holdout_cv", holdout_cv)
        mlflow.log_metric("vs_v19", holdout_cv - 0.3572)
        mlflow.log_metric("num_features", len(feature_cols))
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_param("ablation_study", True)
        mlflow.log_artifact(str(submission_path))

    # Hypothesis DB
    if HAS_DB:
        try:
            db = HypothesisDatabase()
            hyp_id = db.add_hypothesis(
                competition="store-sales-time-series-forecasting",
                experiment_id="v64",
                hypothesis="Ablation study shows only 4 features needed (Roll_std_7, Roll_mean_7, Roll_mean_60, is_holiday)",
                rationale="Feature ablation revealed 8/12 features add noise. Testing minimal 4-feature set.",
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
                    "num_features": len(feature_cols),
                    "features": feature_cols,
                    "discovery": "ablation_study"
                }
            )
        except Exception as e:
            print(f"Note: Could not log: {e}")

    return {
        "holdout_cv": holdout_cv,
        "vs_v19": holdout_cv - 0.3572,
        "decision": decision,
        "success": holdout_cv < 0.3572
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
