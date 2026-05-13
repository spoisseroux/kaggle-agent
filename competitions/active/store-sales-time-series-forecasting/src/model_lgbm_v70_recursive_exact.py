#!/usr/bin/env python3
"""LightGBM v70 - Recursive Forecasting (EXACT Pattern from Successful Notebook)

v66 failed (LB 0.603) with:
- 60-day sliding window
- Lag [3, 7]
- Simple rolling means

v70 uses EXACT pattern from successful recursive notebook:
- 20-day sliding window (critical!)
- Lag [1, 5, 7, 14] (not [3, 7])
- Moving averages [3, 7, 14] (shifted by 1)
- Item-specific mean/std statistics

Expected: LB 0.40-0.45 if implementation pattern is correct
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
from tqdm import tqdm
import gc

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


def feature_engineering(df):
    """EXACT pattern from successful recursive notebook."""
    # Date features
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['week'] = df['date'].dt.isocalendar().week.astype(int)

    # Lagged features [1, 5, 7, 14]
    for lag in [1, 5, 7, 14]:
        df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family'], observed=True)['sales'].shift(lag)

    # Moving average features with one-step shift
    for window in [3, 7, 14]:
        df[f'sales_ma_{window}'] = df.groupby(['store_nbr', 'family'], observed=True)['sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    # Target encoding (store-family statistics)
    df['item_mean'] = df.groupby(['store_nbr', 'family'], observed=True)['sales'].transform('mean')
    df['item_std'] = df.groupby(['store_nbr', 'family'], observed=True)['sales'].transform('std')

    # Holidays
    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def recursive_forecast_all(model, train_df, test_df, holidays, features):
    """
    EXACT recursive pattern from successful notebook.

    Critical differences from v66:
    - 20-day window (not 60)
    - lag [1,5,7,14] (not [3,7])
    - MA with shift(1)
    """
    print("\nRecursive forecasting (exact pattern)...")

    # Get all store-family pairs
    pairs = test_df[["store_nbr", "family"]].drop_duplicates()
    print(f"Total pairs: {len(pairs)}")

    # Test dates
    test_dates = sorted(test_df["date"].unique())
    print(f"Test dates: {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")

    all_submissions = []

    # Process each pair
    for idx, (_, pair_row) in enumerate(tqdm(pairs.iterrows(), total=len(pairs), desc="Pairs")):
        store = pair_row["store_nbr"]
        family = pair_row["family"]

        # Get history (last 20 days before test)
        history_df = train_df[(train_df["store_nbr"] == store) & (train_df["family"] == family)].copy()
        history_df = history_df.tail(20)  # CRITICAL: Only 20 days

        test_pair_df = test_df[(test_df["store_nbr"] == store) & (test_df["family"] == family)].copy()

        if len(test_pair_df) == 0:
            continue

        # Forecast each day recursively
        for test_date in test_dates:
            test_rows = test_pair_df[test_pair_df["date"] == test_date]
            if len(test_rows) == 0:
                continue

            test_row = test_rows.iloc[0].to_dict()
            test_row["sales"] = np.nan

            # Add to history
            temp_df = pd.concat([history_df, pd.DataFrame([test_row])], ignore_index=True)

            # Recreate ALL features
            temp_df = temp_df.sort_values("date").reset_index(drop=True)
            temp_df = feature_engineering(temp_df)
            temp_df = add_holidays(temp_df, holidays)

            # Get features for test date
            test_features = temp_df[temp_df["date"] == test_date][features]

            # Fill NaN with last known value
            for col in features:
                if test_features[col].isna().any():
                    if col in history_df.columns and len(history_df) > 0:
                        last_val = history_df[col].iloc[-1]
                        if pd.notna(last_val):
                            test_features[col].fillna(last_val, inplace=True)
                    test_features[col].fillna(0, inplace=True)

            # Predict
            pred = model.predict(test_features)[0]
            pred = max(pred, 0)

            all_submissions.append({"id": test_row["id"], "sales": pred})

            # Add prediction to history
            test_row["sales"] = pred
            history_df = pd.concat([history_df, pd.DataFrame([test_row])], ignore_index=True)

            # CRITICAL: Keep only last 20 days
            history_df = history_df.tail(20).reset_index(drop=True)

        # Memory cleanup
        if (idx + 1) % 100 == 0:
            gc.collect()

    submission = pd.DataFrame(all_submissions)
    print(f"\nGenerated {len(submission)} predictions")

    return submission


def main():
    print("="*70)
    print("LightGBM v70 - Recursive Forecasting (EXACT Pattern)")
    print("="*70)
    print("Using exact implementation from successful notebook")
    print()

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    # Create features for training
    print("Creating features...")
    train_df = feature_engineering(train_df)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    # Feature columns (EXACT from notebook)
    features = ['day', 'month', 'year', 'dayofweek', 'quarter', 'week',
                'sales_lag_1', 'sales_lag_5', 'sales_lag_7', 'sales_lag_14',
                'sales_ma_3', 'sales_ma_7', 'sales_ma_14',
                'item_mean', 'item_std', 'onpromotion', 'is_holiday']

    print(f"Features: {len(features)}")
    print()

    # 30-day holdout for batch validation
    cutoff_date = train_df["date"].max() - pd.Timedelta(days=30)
    val_mask = train_df["date"] >= cutoff_date
    train_mask = ~val_mask

    X = train_df[features]
    y = train_df["sales"]
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    print()

    # Notebook params (simpler than v19)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "random_state": 42,
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": -1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "n_jobs": -1,
        "verbosity": -1
    }

    print("Training LightGBM...")
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    # Batch validation (for comparison)
    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)
    batch_cv = np.sqrt(mean_squared_log_error(y_val, preds))

    print(f"\nBatch CV (30-day holdout): {batch_cv:.4f}")
    print("Note: Recursive forecast may differ from batch")
    print()

    # Recursive forecasting
    submission = recursive_forecast_all(model, train_df, test_df, holidays, features)

    # Save submission
    cv_str = f"{batch_cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"lgbm_v70_recursive_exact_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSaved: {submission_path}")

    print(f"\n{'='*70}")
    print("EXACT RECURSIVE PATTERN RESULTS")
    print(f"{'='*70}")
    print(f"Batch CV: {batch_cv:.4f}")
    print(f"Predictions: {len(submission):,}")
    print(f"Mean prediction: {submission['sales'].mean():.1f}")
    print(f"Expected LB: 0.40-0.45 (if pattern is correct)")
    print()
    print("Key differences from v66:")
    print("  - 20-day window (not 60)")
    print("  - Lag [1,5,7,14] (not [3,7])")
    print("  - MA with shift(1)")

    # Hypothesis DB
    if HAS_DB:
        try:
            db = HypothesisDatabase()
            hyp_id = db.add_hypothesis(
                competition="store-sales-time-series-forecasting",
                experiment_id="v70",
                hypothesis="Exact recursive pattern from successful notebook (20-day window, lag 1/5/7/14)",
                rationale="v66 failed with 60-day window and lag 3/7. v70 uses EXACT pattern from successful recursive forecasting notebook including 20-day sliding window and correct lag selection.",
                category="time_series"
            )
            db.record_result(
                hypothesis_id=hyp_id,
                cv_score=batch_cv,
                lb_score=None,
                baseline_cv=0.3572,
                succeeded=(batch_cv < 0.45),
                params=params,
                metadata={
                    "recursive": True,
                    "window_days": 20,
                    "lags": [1, 5, 7, 14],
                    "source": "successful_recursive_notebook"
                }
            )
        except Exception as e:
            print(f"Note: Could not log: {e}")

    return {
        "batch_cv": batch_cv,
        "recursive": "exact_pattern",
        "predictions": len(submission),
        "expected_lb": "0.40-0.45"
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
