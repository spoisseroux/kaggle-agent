#!/usr/bin/env python3
"""LightGBM v65 - Long Lags (16+) for Test Set Coverage

BREAKTHROUGH: Top scorers use lag 16+ to ensure ALL test days reference training data!

Test period: Aug 16-31 (16 days)
Training ends: Aug 15

lag 3 on Aug 19 = Aug 16 (in test set, can't access!) ✗
lag 16 on Aug 31 = Aug 15 (in training set, accessible!) ✓

Features from top public notebook (2895 votes):
- Rolling means (20, 30, 60, 90 day windows) with lag 16, 30, 60
- Exponential weighted means with lag 16, 30, 60, 90
- NO short lags (3, 7) that require test set data!

Expected: LB 0.37-0.38 (top leaderboard range)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb

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


def create_long_lag_features(df, is_train=True):
    """Create features with lag 16+ that work for 16-day test period."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Day features (always available)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if is_train:
        # Lag 16 features (safe for all 16 test days)
        df["Lag_16"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(16)

        # Rolling means with lag 16 (from top notebook)
        for window in [20, 30, 60, 90]:
            df[f"SMA{window}_lag16"] = (
                df.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(16))
            )

        # Exponential weighted means with different alphas and lags
        for alpha in [0.95, 0.9, 0.8]:
            for lag in [16, 30]:
                df[f"EWM_{int(alpha*100)}_lag{lag}"] = (
                    df.groupby(["store_nbr", "family"], observed=True)["sales"]
                    .transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
                )

        # Rolling std with lag 16
        df["Roll_std_30_lag16"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=30, min_periods=1).std().shift(16))
        )

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    print("="*70)
    print("LightGBM v65 - Long Lags (16+) for Full Test Coverage")
    print("="*70)
    print("Breakthrough: Use lag 16+ to avoid needing test set data")
    print()

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    # Combine train + test to create features (test lags reference training data)
    print("Creating long-lag features...")
    test_df["sales"] = np.nan  # Add sales column for feature creation
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = create_long_lag_features(combined, is_train=True)
    combined = add_holidays(combined, holidays)

    # Split back
    train_df = combined[combined["sales"].notna()].copy()
    test_df = combined[combined["sales"].isna()].copy()

    # Drop NaN rows from training
    train_df = train_df.dropna()

    # Feature columns
    feature_cols = [
        "onpromotion", "day_of_week", "is_weekend", "is_holiday",
        "Lag_16",
        "SMA20_lag16", "SMA30_lag16", "SMA60_lag16", "SMA90_lag16",
        "EWM_95_lag16", "EWM_95_lag30",
        "EWM_90_lag16", "EWM_90_lag30",
        "EWM_80_lag16", "EWM_80_lag30",
        "Roll_std_30_lag16"
    ]

    print(f"Features: {len(feature_cols)}")
    print("Key change: All lags 16+ (accessible from training for all test days)")
    print()

    # 30-day holdout validation
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

    # v19 hyperparameters (proven to work)
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
    print("LONG LAG RESULTS")
    print(f"{'='*70}")
    print(f"v19 (lag 3/7):    CV 0.3572, LB 0.498")
    print(f"v65 (lag 16+):    CV {holdout_cv:.4f}")
    print(f"Expected LB:      0.37-0.38 (top scorer range)")
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
    submission_path = SUBMISSION_DIR / f"lgbm_v65_long_lags_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # Hypothesis DB
    if HAS_DB:
        try:
            db = HypothesisDatabase()
            hyp_id = db.add_hypothesis(
                competition="store-sales-time-series-forecasting",
                experiment_id="v65",
                hypothesis="Long lags (16+) allow all test days to reference training data, avoiding NaN/constant fills",
                rationale="Top scorers use lag 16+ instead of lag 3/7. Test is 16 days, so lag 16+ always references training data. Discovered from analyzing top public notebook (2895 votes).",
                category="feature_engineering"
            )
            db.record_result(
                hypothesis_id=hyp_id,
                cv_score=holdout_cv,
                lb_score=None,
                baseline_cv=0.3572,
                succeeded=(holdout_cv < 0.40),
                params=params,
                metadata={
                    "num_features": len(feature_cols),
                    "min_lag": 16,
                    "max_lag": 30,
                    "discovery_source": "top_public_notebook"
                }
            )
        except Exception as e:
            print(f"Note: Could not log: {e}")

    return {
        "holdout_cv": holdout_cv,
        "vs_v19": holdout_cv - 0.3572,
        "expected_lb": "0.37-0.38",
        "success": holdout_cv < 0.40
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
