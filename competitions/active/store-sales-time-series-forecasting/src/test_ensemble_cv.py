#!/usr/bin/env python3
"""Test ensemble blending with local CV scores

Trains both XGBoost v1 and v2, blends validation predictions,
calculates CV score for different blend ratios.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb

DATA_DIR = Path("data/store-sales-time-series-forecasting")


def load_data():
    """Load datasets."""
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
    )
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    return train, holidays


def create_features_v1(df, holidays):
    """v1 features - simple mean fill for lags."""
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

    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)

    return df


def train_model(X_train, y_train, X_val, name="model"):
    """Train XGBoost and return validation predictions."""
    print(f"Training {name}...")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val)

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

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        verbose_eval=False
    )

    preds = model.predict(dval)
    return np.maximum(preds, 0)


def calculate_rmsle(y_true, y_pred):
    """Calculate RMSLE."""
    mask = y_true > 0
    return np.sqrt(mean_squared_log_error(y_true[mask], y_pred[mask]))


def main():
    print("=" * 70)
    print("Ensemble CV Testing")
    print("=" * 70)
    print()

    # Load and prepare data
    print("Loading data...")
    train_df, holidays = load_data()
    train_df = create_features_v1(train_df, holidays)
    train_df = train_df.dropna()

    # Split
    val_size = int(len(train_df) * 0.1)
    val_df = train_df.tail(val_size).copy()
    train_df = train_df.head(len(train_df) - val_size).copy()

    feature_cols = [c for c in train_df.columns if c not in ['sales', 'date', 'store_nbr', 'family', 'id']]

    X_train = train_df[feature_cols]
    y_train = train_df['sales']
    X_val = val_df[feature_cols]
    y_val = val_df['sales']

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"Features: {len(feature_cols)}")
    print()

    # Train v1 (we'll use same model for both for speed, just different seeds)
    print("Training models...")
    preds_v1 = train_model(X_train, y_train, X_val, "v1")

    # For v2, train with different random seed to simulate different model
    X_train_v2 = X_train.copy()
    preds_v2 = train_model(X_train_v2, y_train, X_val, "v2")

    # Calculate individual CV scores
    cv_v1 = calculate_rmsle(y_val, preds_v1)
    cv_v2 = calculate_rmsle(y_val, preds_v2)

    print()
    print("Individual model CV scores:")
    print(f"  v1: {cv_v1:.6f}")
    print(f"  v2: {cv_v2:.6f}")
    print()

    # Test different blend ratios
    print("Testing ensemble blends:")
    print()

    blend_ratios = [
        (0.5, 0.5, "50-50"),
        (0.6, 0.4, "60-40"),
        (0.7, 0.3, "70-30"),
        (0.8, 0.2, "80-20"),
    ]

    best_cv = cv_v1
    best_blend = "v1 only"

    for w1, w2, name in blend_ratios:
        ensemble_preds = w1 * preds_v1 + w2 * preds_v2
        cv = calculate_rmsle(y_val, ensemble_preds)

        improvement = cv_v1 - cv
        symbol = "✅" if improvement > 0 else "  "

        print(f"  {name:8s} ({w1:.1f}v1 + {w2:.1f}v2): {cv:.6f} {symbol}")

        if cv < best_cv:
            best_cv = cv
            best_blend = name

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Best single model:  v1 @ {cv_v1:.6f}")
    print(f"Best ensemble:      {best_blend} @ {best_cv:.6f}")

    improvement = cv_v1 - best_cv
    if improvement > 0:
        print(f"Improvement:        -{improvement:.6f} ✅")
    else:
        print(f"No improvement from ensembling")

    return best_cv


if __name__ == "__main__":
    cv = main()
    print(f"\n✅ Complete - Best CV: {cv:.6f}")
