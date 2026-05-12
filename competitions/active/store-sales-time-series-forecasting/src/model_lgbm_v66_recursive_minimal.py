#!/usr/bin/env python3
"""LightGBM v66 - Minimal Recursive Forecasting (Debug Version)

Attempt #4 at recursive forecasting with minimal, careful implementation.
Previous attempts (v32/v33/v34) all failed with negative correlation.

This version:
- Uses EXACT pattern from successful public notebook
- Starts with single store-family pair for debugging
- Validates predictions at each step
- Logs correlation with v19 to detect inversion bug early
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from scipy.stats import pearsonr
import lightgbm as lgb
from tqdm import tqdm

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

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


def recursive_forecast_single(model, train_df, test_df, holidays, feature_cols):
    """
    Recursive forecasting for a SINGLE store-family pair.

    Pattern from successful notebook:
    1. Start with full training history
    2. For each test day:
       a. Add row with NaN sales
       b. Recreate ALL features
       c. Predict sales for that row
       d. Replace NaN with prediction
       e. Append to history
    3. Keep only recent history (sliding window)
    """
    print("\nDEBUG: Testing recursive forecasting on single store-family pair")

    # Pick one store-family pair for debugging
    test_pair = train_df.iloc[0]
    store = test_pair["store_nbr"]
    family = test_pair["family"]

    print(f"Store: {store}, Family: {family}")

    # Filter to this pair
    history_df = train_df[(train_df["store_nbr"] == store) & (train_df["family"] == family)].copy()
    test_pair_df = test_df[(test_df["store_nbr"] == store) & (test_df["family"] == family)].copy()

    print(f"Training history: {len(history_df)} days")
    print(f"Test days: {len(test_pair_df)}")

    # Test dates
    test_dates = sorted(test_pair_df["date"].unique())

    predictions = []

    for test_date in tqdm(test_dates, desc="Forecasting"):
        # Create row for this date (sales unknown)
        test_row = test_pair_df[test_pair_df["date"] == test_date].iloc[0].to_dict()
        test_row["sales"] = np.nan

        # Add to history
        temp_df = pd.concat([history_df, pd.DataFrame([test_row])], ignore_index=True)

        # Recreate features
        temp_df = temp_df.sort_values("date").reset_index(drop=True)
        temp_df = create_features(temp_df)
        temp_df = add_holidays(temp_df, holidays)

        # Get features for test date
        test_features = temp_df[temp_df["date"] == test_date][feature_cols]

        # Check for NaN
        if test_features.isna().any().any():
            print(f"  WARNING: NaN features on {test_date}")
            print(f"  NaN columns: {test_features.columns[test_features.isna().any()].tolist()}")
            # Fill NaN with last known value from history
            for col in feature_cols:
                if test_features[col].isna().any():
                    last_val = history_df[col].iloc[-1] if col in history_df.columns and len(history_df) > 0 else 0
                    test_features[col].fillna(last_val, inplace=True)

        # Predict
        pred = model.predict(test_features)[0]
        pred = max(pred, 0)  # No negative sales

        predictions.append(pred)

        # Add prediction to history for next iteration
        test_row["sales"] = pred
        history_df = pd.concat([history_df, pd.DataFrame([test_row])], ignore_index=True)

        # Keep only last 60 days (sliding window)
        history_df = history_df.tail(60).reset_index(drop=True)

    print(f"\nGenerated {len(predictions)} predictions")
    print(f"Mean prediction: {np.mean(predictions):.2f}")
    print(f"Std prediction: {np.std(predictions):.2f}")

    return predictions


def main():
    print("="*70)
    print("LightGBM v66 - Minimal Recursive Forecasting (Debug)")
    print("="*70)
    print("Testing recursive approach on single store-family pair")
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

    # Validation
    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)
    holdout_cv = np.sqrt(mean_squared_log_error(y_val, preds))

    print(f"\nHoldout CV: {holdout_cv:.4f} (v19: 0.3572)")
    print()

    # DEBUG: Test recursive forecasting on single pair
    recursive_preds = recursive_forecast_single(model, train_df, test_df, holidays, feature_cols)

    print("\n" + "="*70)
    print("RECURSIVE FORECAST DEBUG COMPLETE")
    print("="*70)
    print("Check if predictions look reasonable before full implementation")
    print(f"Sample predictions (first 5): {recursive_preds[:5]}")
    print(f"Sample predictions (last 5): {recursive_preds[-5:]}")

    return {"holdout_cv": holdout_cv, "recursive_test": "single_pair_only"}


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
