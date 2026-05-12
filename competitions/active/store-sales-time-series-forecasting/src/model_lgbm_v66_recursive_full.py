#!/usr/bin/env python3
"""LightGBM v66 - Full Recursive Forecasting

Scales up successful v66 minimal (single pair) to all store-family pairs.

Expected: LB 0.37-0.40 (top leaderboard range)
Previous recursive attempts (v32/v33/v34) failed due to implementation bugs.
This version uses validated pattern from single-pair test.
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


def create_features(df):
    """Create v1 features - proven set from v19."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Lag features
    df["Lag_3"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(3)
    df["Lag_7"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(7)

    # Rolling means
    for window in [7, 14, 30, 60, 90]:
        df[f"Roll_mean_{window}"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )

    # Rolling std
    df["Roll_std_7"] = (
        df.groupby(["store_nbr", "family"], observed=True)["sales"]
        .transform(lambda x: x.rolling(window=7, min_periods=1).std())
    )

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def recursive_forecast_all(model, train_df, test_df, holidays, feature_cols):
    """
    Recursive forecasting for ALL store-family pairs.

    Uses pattern validated in v66 minimal (single pair test).
    """
    print("\nRecursive forecasting for all store-family pairs...")

    # Get all store-family pairs
    pairs = test_df[["store_nbr", "family"]].drop_duplicates()
    print(f"Total pairs: {len(pairs)}")

    # Test dates (same for all pairs)
    test_dates = sorted(test_df["date"].unique())
    print(f"Test dates: {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")

    all_submissions = []

    # Process each pair
    for idx, (_, pair_row) in enumerate(tqdm(pairs.iterrows(), total=len(pairs), desc="Forecasting pairs")):
        store = pair_row["store_nbr"]
        family = pair_row["family"]

        # Filter to this pair
        history_df = train_df[(train_df["store_nbr"] == store) & (train_df["family"] == family)].copy()
        test_pair_df = test_df[(test_df["store_nbr"] == store) & (test_df["family"] == family)].copy()

        if len(test_pair_df) == 0:
            continue  # Skip if no test data for this pair

        pair_predictions = []

        # Forecast each day recursively
        for test_date in test_dates:
            # Get test row for this date
            test_rows = test_pair_df[test_pair_df["date"] == test_date]
            if len(test_rows) == 0:
                continue

            test_row = test_rows.iloc[0].to_dict()
            test_row["sales"] = np.nan

            # Add to history
            temp_df = pd.concat([history_df, pd.DataFrame([test_row])], ignore_index=True)

            # Recreate features
            temp_df = temp_df.sort_values("date").reset_index(drop=True)
            temp_df = create_features(temp_df)
            temp_df = add_holidays(temp_df, holidays)

            # Get features for test date
            test_features = temp_df[temp_df["date"] == test_date][feature_cols]

            # Handle NaN features (fill with last known value or 0)
            if test_features.isna().any().any():
                for col in feature_cols:
                    if test_features[col].isna().any():
                        # Fill with last known value from history
                        if col in history_df.columns and len(history_df) > 0:
                            last_val = history_df[col].iloc[-1]
                            if pd.notna(last_val):
                                test_features[col].fillna(last_val, inplace=True)
                        # If still NaN, fill with 0
                        test_features[col].fillna(0, inplace=True)

            # Predict
            pred = model.predict(test_features)[0]
            pred = max(pred, 0)  # No negative sales

            pair_predictions.append({"id": test_row["id"], "sales": pred})

            # Add prediction to history for next iteration
            test_row["sales"] = pred
            history_df = pd.concat([history_df, pd.DataFrame([test_row])], ignore_index=True)

            # Keep only last 60 days (memory management)
            history_df = history_df.tail(60).reset_index(drop=True)

        all_submissions.extend(pair_predictions)

        # Memory cleanup every 100 pairs
        if (idx + 1) % 100 == 0:
            gc.collect()

    # Create submission dataframe
    submission = pd.DataFrame(all_submissions)
    print(f"\nGenerated {len(submission)} predictions")

    return submission


def main():
    print("="*70)
    print("LightGBM v66 - Full Recursive Forecasting")
    print("="*70)
    print("Scaling up from successful single-pair test")
    print()

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    # Create features for training
    print("Creating training features...")
    train_df = create_features(train_df)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    # Feature columns
    exclude_cols = ["id", "date", "sales", "store_nbr", "family"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"Features: {len(feature_cols)}")

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

    # v19 hyperparameters
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

    # Validation (batch prediction for validation set)
    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)
    holdout_cv = np.sqrt(mean_squared_log_error(y_val, preds))

    print(f"\nHoldout CV (batch): {holdout_cv:.4f} (v19: 0.3572)")
    print()

    # Recursive forecasting for test set
    submission = recursive_forecast_all(model, train_df, test_df, holidays, feature_cols)

    # Save submission
    cv_str = f"{holdout_cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"lgbm_v66_recursive_full_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSaved: {submission_path}")

    # Hypothesis DB
    if HAS_DB:
        try:
            db = HypothesisDatabase()
            hyp_id = db.add_hypothesis(
                competition="store-sales-time-series-forecasting",
                experiment_id="v66_full",
                hypothesis="Full recursive forecasting across all store-family pairs using validated pattern",
                rationale="Single-pair test (v66 minimal) succeeded with reasonable predictions. Scaling up carefully with same pattern. Expected LB 0.37-0.40 based on top scorer analysis.",
                category="time_series"
            )
            db.record_result(
                hypothesis_id=hyp_id,
                cv_score=holdout_cv,
                lb_score=None,
                baseline_cv=0.3572,
                succeeded=(holdout_cv < 0.40),
                params=params,
                metadata={
                    "recursive": True,
                    "pairs": 1782,
                    "test_days": 16,
                    "discovery": "scaled_from_successful_debug"
                }
            )
        except Exception as e:
            print(f"Note: Could not log: {e}")

    print("\n" + "="*70)
    print("RECURSIVE FORECASTING COMPLETE")
    print("="*70)
    print(f"Total predictions: {len(submission)}")
    print(f"Expected LB: 0.37-0.40 (if implementation correct)")
    print(f"Submission: {submission_path}")

    return {
        "holdout_cv": holdout_cv,
        "recursive": "full",
        "predictions": len(submission),
        "expected_lb": "0.37-0.40"
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
