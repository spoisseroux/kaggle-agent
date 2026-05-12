#!/usr/bin/env python3
"""LightGBM v47 - Day-of-Week Specific Features (Research-Backed)

INSIGHT FROM TOP KAGGLE SOLUTIONS:
"Average for that specific day for a product in a store is a super valuable feature"

STRATEGY: Add day-of-week specific historical averages
- New features: store_family_dow_mean (mean sales for this store/family/dayofweek combo)
- Example: Average Monday sales for GROCERY at store 1
- Rationale: Day-of-week patterns are strong (weekends vs weekdays)

Features added:
- store_family_dow_mean: Historical average for this specific combination
- store_family_dow_std: Std dev for stability

Expected: Better capture of weekly seasonality, CV improvement over v19
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
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


def create_features(df, is_train=True, train_df=None):
    """v1 features + day-of-week specific averages"""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if is_train:
        # Standard v1 features
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

        # NEW: Day-of-week specific features
        # Compute historical average for each store/family/dayofweek combination
        df["store_family_dow_mean"] = df.groupby(
            ["store_nbr", "family", "day_of_week"], observed=True
        )["sales"].transform("mean")

        df["store_family_dow_std"] = df.groupby(
            ["store_nbr", "family", "day_of_week"], observed=True
        )["sales"].transform("std")
    else:
        if train_df is not None:
            # Standard v1 test features
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

            # NEW: Day-of-week specific features from training data
            dow_stats = (
                train_df.groupby(["store_nbr", "family", "day_of_week"], observed=True)["sales"]
                .agg(["mean", "std"])
                .reset_index()
            )
            dow_stats.columns = ["store_nbr", "family", "day_of_week", "dow_mean", "dow_std"]

            df = df.merge(
                dow_stats,
                on=["store_nbr", "family", "day_of_week"],
                how="left"
            )

            df["store_family_dow_mean"] = df["dow_mean"].fillna(df["lag_mean"])
            df["store_family_dow_std"] = df["dow_std"].fillna(df["lag_std"])

            df = df.drop(columns=["dow_mean", "dow_std"])

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    print("="*70)
    print("LightGBM v47 - Day-of-Week Specific Features")
    print("="*70)
    print("Strategy: Add store/family/dayofweek historical averages")
    print("Insight: Day-specific patterns are highly predictive")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-lightgbm-v1")

    print("Loading data...")
    train_df, test_df, holidays = load_data()

    print("Creating v1 + day-of-week features...")
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)

    exclude_cols = ["id", "date", "sales", "store_nbr", "family", "lag_mean", "lag_std"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Features: {len(feature_cols)} (v1: 12, new: 2)")
    print(f"New features: store_family_dow_mean, store_family_dow_std")
    print()

    # 30-day holdout validation
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

    # v19 parameters
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
        "random_state": 42,
        "verbosity": -1,
    }

    print("LightGBM parameters (v19 proven):")
    for k, v in params.items():
        if k not in ["verbosity"]:
            print(f"  {k}: {v}")
    print()

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    with mlflow.start_run(run_name="lgbm_v47_dow_features"):
        mlflow.log_params(params)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("new_features", "store_family_dow_mean,store_family_dow_std")

        print("Training LightGBM...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )

        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        val_preds = np.maximum(val_preds, 0)

        cv_score = np.sqrt(mean_squared_log_error(y_val, val_preds))

        print()
        print(f"Holdout CV: {cv_score:.4f}")
        print(f"v19 (12 features): 0.3572")
        if cv_score < 0.3572:
            print(f"✓ IMPROVEMENT: {(0.3572 - cv_score) / 0.3572 * 100:.1f}% better")
        else:
            print(f"✗ WORSE: {(cv_score - 0.3572) / 0.3572 * 100:.1f}% worse")
        print()

        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print("Top 10 features by importance:")
        print(feature_importance.head(10).to_string(index=False))
        print()

        mlflow.log_metric("holdout_cv", cv_score)
        mlflow.log_metric("vs_v19", cv_score - 0.3572)

    print("Generating test predictions...")
    X_test = test_df[feature_cols]
    test_preds = model.predict(X_test, num_iteration=model.best_iteration)
    test_preds = np.maximum(test_preds, 0)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": test_preds
    })

    filename = f"lgbm_v47_dow_{cv_score:.5f}.csv".replace(".", "")
    filepath = SUBMISSION_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"Submission saved: {filename}")
    print(f"Mean prediction: {test_preds.mean():.2f}")
    print()

    if HAS_LANGFUSE:
        try:
            log_kaggle_model(
                competition="store-sales-time-series-forecasting",
                model_name="LightGBM v47 DOW Features",
                cv_score=cv_score,
                model_type="lightgbm",
                params=params,
                features=feature_cols
            )
        except Exception:
            pass

    print("="*70)
    print("ANALYSIS:")
    print("Day-of-week specific features capture weekly patterns")
    print("Check feature importance to see if dow_mean ranks highly")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())