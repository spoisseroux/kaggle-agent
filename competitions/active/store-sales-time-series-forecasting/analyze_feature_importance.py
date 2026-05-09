#!/usr/bin/env python3
"""Analyze feature importance from XGBoost v10 model."""
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import matplotlib.pyplot as plt

DATA_DIR = Path("data/store-sales-time-series-forecasting")


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
    df["Time"] = df.groupby(["store_nbr", "family"], observed=True).cumcount()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)
    for window in [7, 14, 30, 60, 90]:
        df[f"Roll_mean_{window}"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        df[f"Roll_std_{window}"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )
    df["onpromotion_lag1"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(1).fillna(0)
    df["onpromotion_lag7"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(7).fillna(0)
    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


print("="*70)
print("Feature Importance Analysis")
print("="*70)

# Load data
print("Loading data...")
train_df, holidays = load_data()

# Sample for speed (use 20% of data)
train_df = train_df.sample(frac=0.2, random_state=42)

print("Creating features...")
train_df = create_features(train_df)
train_df = add_holidays(train_df, holidays)
train_df = train_df.dropna()

feature_cols = [
    "Time", "day_of_week", "day_of_month", "month", "week_of_year",
    "is_month_end", "is_month_start", "is_weekend",
    "Lag_1", "Lag_2", "Lag_3", "Lag_7", "Lag_14", "Lag_21", "Lag_28",
    "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "Roll_mean_60", "Roll_mean_90",
    "Roll_std_7", "Roll_std_14", "Roll_std_30", "Roll_std_60", "Roll_std_90",
    "onpromotion", "onpromotion_lag1", "onpromotion_lag7", "is_holiday"
]

X = train_df[feature_cols]
y = train_df["sales"]

print(f"Training on {len(X):,} samples...")

# Train model (v10 params)
dtrain = xgb.DMatrix(X, label=y)
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
    'seed': 42
}

model = xgb.train(params, dtrain, num_boost_round=300)

# Get feature importance
importance = model.get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': list(importance.keys()),
    'importance': list(importance.values())
}).sort_values('importance', ascending=False)

print("\n" + "="*70)
print("Top 15 Most Important Features:")
print("="*70)
for idx, row in importance_df.head(15).iterrows():
    print(f"{row['feature']:25} {row['importance']:>10.1f}")

print("\n" + "="*70)
print("Bottom 10 Least Important Features:")
print("="*70)
for idx, row in importance_df.tail(10).iterrows():
    print(f"{row['feature']:25} {row['importance']:>10.1f}")

# Calculate cumulative importance
importance_df['cumulative'] = importance_df['importance'].cumsum() / importance_df['importance'].sum() * 100

# Find how many features needed for 95% importance
n_features_95 = (importance_df['cumulative'] <= 95).sum()
print(f"\n{'='*70}")
print(f"Features needed for 95% cumulative importance: {n_features_95}")
print(f"Current features: {len(feature_cols)}")
print(f"Potential reduction: {len(feature_cols) - n_features_95} features")

# Save results
importance_df.to_csv("feature_importance_analysis.csv", index=False)
print(f"\nFull results saved to: feature_importance_analysis.csv")
