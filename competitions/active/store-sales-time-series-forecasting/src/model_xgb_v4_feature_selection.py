#!/usr/bin/env python3
"""XGBoost v4 - Feature selection (reduced feature set for better generalization)

Testing hypothesis: Simpler model with fewer features may generalize better
Removing: long rolling windows (60, 90 days), redundant time features
Keeping: Core lags, short rolling windows, essential time features
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb

DATA_DIR = Path("data/store-sales-time-series-forecasting")

# Reduced feature set - 19 features vs 29 in v1
SELECTED_FEATURES = [
    'Time', 'day_of_week', 'month', 'is_weekend',
    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 'Lag_14', 'Lag_28',
    'Roll_mean_7', 'Roll_mean_14', 'Roll_mean_30',
    'Roll_std_7', 'Roll_std_14', 'Roll_std_30',
    'onpromotion', 'onpromotion_lag1', 'is_holiday'
]


def load_data():
    """Load datasets."""
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
    )
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    return train, holidays


def create_features(df, holidays):
    """Create features - only what we need for selected features."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Time index
    df["Time"] = df.groupby(["store_nbr", "family"], observed=True).cumcount()

    # Time features (selected subset)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Lags (selected subset: 1,2,3,7,14,28 - skip 21)
    for lag in [1, 2, 3, 7, 14, 28]:
        df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

    # Rolling features (selected subset: 7,14,30 - skip 60,90)
    for window in [7, 14, 30]:
        df[f"Roll_mean_{window}"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        df[f"Roll_std_{window}"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )

    # Promotion lags
    df["onpromotion_lag1"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(1).fillna(0)

    # Holidays
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)

    return df


def main():
    print("=" * 60)
    print("XGBoost v4 - Feature Selection Experiment")
    print("=" * 60)
    print(f"Selected features: {len(SELECTED_FEATURES)} (vs 29 in v1)")
    print(f"Removed: Lag_21, Roll_*_60, Roll_*_90, day_of_month,")
    print(f"         week_of_year, is_month_end, is_month_start")
    print()

    # Load data
    train_df, holidays = load_data()

    # Create features
    print("Creating features...")
    train_df = create_features(train_df, holidays)

    # Drop rows with NaN (from lags)
    train_df = train_df.dropna()

    print(f"Train shape after features: {train_df.shape}")
    print()

    # Train/val split (last 10% for validation)
    val_size = int(len(train_df) * 0.1)
    val_df = train_df.tail(val_size).copy()
    train_df = train_df.head(len(train_df) - val_size).copy()

    X_train = train_df[SELECTED_FEATURES]
    y_train = train_df['sales']
    X_val = val_df[SELECTED_FEATURES]
    y_val = val_df['sales']

    print(f"Train samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print()

    # Train XGBoost
    print("Training XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'eval_metric': 'rmse',
        'seed': 42
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # Evaluate
    y_pred = model.predict(dval)
    y_pred = np.maximum(y_pred, 0)

    # Calculate RMSLE
    mask = y_val > 0
    cv_rmsle = np.sqrt(mean_squared_log_error(y_val[mask], y_pred[mask]))

    print()
    print(f"Validation RMSLE: {cv_rmsle:.6f}")
    print(f"Best iteration: {model.best_iteration}")

    # Compare to baseline
    baseline_cv = 0.321032
    improvement = baseline_cv - cv_rmsle

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"XGBoost v4 (19 features):  {cv_rmsle:.6f}")
    print(f"XGBoost v1 (29 features):  {baseline_cv:.6f}")
    if improvement > 0:
        print(f"Improvement:                -{improvement:.6f} ✅")
    else:
        print(f"Change:                     +{abs(improvement):.6f}")
    print()

    # Feature importance
    importance = model.get_score(importance_type='gain')
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 features by importance:")
    for i, (feat, score) in enumerate(importance_sorted[:10], 1):
        print(f"  {i:2}. {feat:20s} {score:8.1f}")

    return cv_rmsle


if __name__ == "__main__":
    cv = main()
    print(f"\n✅ Complete - CV RMSLE: {cv:.6f}")
