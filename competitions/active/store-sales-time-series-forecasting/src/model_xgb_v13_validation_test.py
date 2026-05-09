#!/usr/bin/env python3
"""XGBoost v13 - Validation Strategy Test

Testing different CV strategies to understand CV-LB gap:
1. TimeSeriesSplit (current)
2. Last N days hold-out (mimics test set timing)
3. Store-stratified splits

Goal: Find validation strategy that better predicts LB performance.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

DATA_DIR = Path("data/store-sales-time-series-forecasting")

# Top 12 features from v11
TOP_FEATURES = [
    "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "is_weekend",
    "Roll_mean_60", "day_of_week", "is_holiday", "Lag_3",
    "Roll_mean_90", "onpromotion", "Lag_7", "Roll_std_7"
]


def load_data():
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
    )
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    return train, holidays


def create_features(df):
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

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

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def evaluate_split(X_train, y_train, X_val, y_val, name):
    """Train and evaluate on a split."""
    params = {
        "objective": "reg:squaredlogerror",
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 500,
        "random_state": 42,
        "verbosity": 0,
    }

    model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)

    mask = y_val > 0
    if mask.sum() > 0:
        score = np.sqrt(mean_squared_log_error(y_val[mask], preds[mask]))
    else:
        score = float('inf')

    return score


def main():
    print("="*70)
    print("XGBoost v13 - Validation Strategy Test")
    print("="*70)
    print()

    train_df, holidays = load_data()

    print("Creating features...")
    train_df = create_features(train_df)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    X = train_df[TOP_FEATURES]
    y = train_df["sales"]

    print(f"Dataset: {len(train_df):,} rows")
    print(f"Date range: {train_df['date'].min()} to {train_df['date'].max()}")
    print()

    # Strategy 1: TimeSeriesSplit (current baseline)
    print("Strategy 1: TimeSeriesSplit (5 folds)")
    print("-" * 50)
    tscv = TimeSeriesSplit(n_splits=5)
    tscv_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        score = evaluate_split(
            X.iloc[train_idx], y.iloc[train_idx],
            X.iloc[val_idx], y.iloc[val_idx],
            f"TSCV_Fold{fold}"
        )
        tscv_scores.append(score)
        print(f"  Fold {fold}: {score:.4f}")

    tscv_mean = np.mean(tscv_scores)
    print(f"  Mean: {tscv_mean:.4f} ± {np.std(tscv_scores):.4f}")
    print()

    # Strategy 2: Last 16 days hold-out (mimics test period)
    print("Strategy 2: Last 16 days hold-out")
    print("-" * 50)
    print("  (Test set is Aug 16-31, 2017 = 16 days)")

    # Get last 16 days as validation
    train_dates_sorted = sorted(train_df["date"].unique())
    cutoff_date = train_dates_sorted[-16]

    val_mask = train_df["date"] >= cutoff_date
    train_mask = ~val_mask

    holdout_score = evaluate_split(
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        "Holdout_16days"
    )
    print(f"  Score: {holdout_score:.4f}")
    print(f"  Val period: {train_df[val_mask]['date'].min()} to {train_df[val_mask]['date'].max()}")
    print()

    # Strategy 3: Last 30 days hold-out
    print("Strategy 3: Last 30 days hold-out")
    print("-" * 50)
    cutoff_date_30 = train_dates_sorted[-30]
    val_mask_30 = train_df["date"] >= cutoff_date_30
    train_mask_30 = ~val_mask_30

    holdout_30_score = evaluate_split(
        X[train_mask_30], y[train_mask_30],
        X[val_mask_30], y[val_mask_30],
        "Holdout_30days"
    )
    print(f"  Score: {holdout_30_score:.4f}")
    print(f"  Val period: {train_df[val_mask_30]['date'].min()} to {train_df[val_mask_30]['date'].max()}")
    print()

    # Summary
    print("="*70)
    print("VALIDATION STRATEGY COMPARISON")
    print("="*70)
    print(f"TimeSeriesSplit (5-fold):     {tscv_mean:.4f}")
    print(f"Last 16 days hold-out:        {holdout_score:.4f}")
    print(f"Last 30 days hold-out:        {holdout_30_score:.4f}")
    print()
    print("Analysis:")
    print(f"  Best XGBoost LB score: 0.464 (xgb_optuna)")
    print(f"  Best TSCV score:       {tscv_mean:.4f}")
    print(f"  Gap:                   {abs(0.464 - tscv_mean):.4f}")
    print()

    if abs(0.464 - holdout_score) < abs(0.464 - tscv_mean):
        print("✓ Hold-out validation closer to LB than TSCV")
        print("  Recommendation: Use last-N-days hold-out for future validation")
    else:
        print("✗ TSCV still better predictor of LB")
        print("  Recommendation: CV-LB gap likely from other factors (data shift, metric calc)")

    return {
        "tscv_mean": tscv_mean,
        "holdout_16": holdout_score,
        "holdout_30": holdout_30_score,
        "success": True
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
