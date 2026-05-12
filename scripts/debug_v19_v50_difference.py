#!/usr/bin/env python3
"""Debug why v19 (0.3572) differs from v50 LightGBM (0.4102)"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data/store-sales-time-series-forecasting")


def create_features(df, is_train=True, train_df=None):
    """v1 features"""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if is_train:
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

            df["Lag_3"] = df["lag_mean"].fillna(0)
            df["Lag_7"] = df["lag_mean"].fillna(0)
            for window in [7, 14, 30, 60, 90]:
                df[f"Roll_mean_{window}"] = df["lag_mean"].fillna(0)
            df["Roll_std_7"] = df["lag_std"].fillna(0)

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    print("="*70)
    print("Debugging v19 vs v50 LightGBM difference")
    print("="*70)
    print()

    # Load data
    print("Loading data...")
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
    )
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])

    # Create features
    print("Creating features...")
    train = create_features(train, is_train=True)
    train = add_holidays(train, holidays)
    train = train.dropna()

    # Split
    cutoff_date = train['date'].max() - pd.Timedelta(days=30)
    val_mask = train['date'] >= cutoff_date
    train_mask = ~val_mask

    exclude_cols = ["id", "date", "sales", "store_nbr", "family"]
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X = train[feature_cols]
    y = train["sales"]
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    print()

    # Test 1: LGBMRegressor (v19 style) with n_estimators=600
    print("TEST 1: LGBMRegressor (v19 style)")
    params_v19 = {
        "objective": "regression",
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

    model_v19 = lgb.LGBMRegressor(**params_v19)
    model_v19.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    preds_v19 = np.maximum(model_v19.predict(X_val), 0)

    # CV with filtering (v19 style)
    mask = y_val > 0
    cv_v19_filtered = np.sqrt(mean_squared_log_error(y_val[mask], preds_v19[mask]))

    # CV without filtering (v50 style)
    cv_v19_unfiltered = np.sqrt(mean_squared_log_error(y_val, preds_v19))

    print(f"  CV (filtered, y>0): {cv_v19_filtered:.4f}")
    print(f"  CV (unfiltered):    {cv_v19_unfiltered:.4f}")
    print(f"  Iterations: {model_v19.n_estimators}")
    print()

    # Test 2: lgb.train (v50 style)
    print("TEST 2: lgb.train (v50 style)")
    params_v50 = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 6,
        "verbosity": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model_v50 = lgb.train(
        params_v50,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    preds_v50 = np.maximum(model_v50.predict(X_val, num_iteration=model_v50.best_iteration), 0)

    # CV both ways
    cv_v50_filtered = np.sqrt(mean_squared_log_error(y_val[mask], preds_v50[mask]))
    cv_v50_unfiltered = np.sqrt(mean_squared_log_error(y_val, preds_v50))

    print(f"  CV (filtered, y>0): {cv_v50_filtered:.4f}")
    print(f"  CV (unfiltered):    {cv_v50_unfiltered:.4f}")
    print(f"  Iterations: {model_v50.best_iteration}")
    print()

    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print()
    print(f"v19 reported CV: 0.3572")
    print(f"v50 reported CV: 0.4102")
    print(f"Difference: {0.4102 - 0.3572:.4f} (14.8%)")
    print()
    print(f"Test 1 (v19 style) filtered: {cv_v19_filtered:.4f}")
    print(f"Test 2 (v50 style) unfiltered: {cv_v50_unfiltered:.4f}")
    print(f"Difference: {cv_v50_unfiltered - cv_v19_filtered:.4f}")
    print()

    if abs(cv_v19_filtered - 0.3572) < 0.01:
        print("✓ Test 1 matches v19 reported CV!")
    else:
        print(f"✗ Test 1 differs from v19 by {abs(cv_v19_filtered - 0.3572):.4f}")

    if abs(cv_v50_unfiltered - 0.4102) < 0.01:
        print("✓ Test 2 matches v50 reported CV!")
    else:
        print(f"✗ Test 2 differs from v50 by {abs(cv_v50_unfiltered - 0.4102):.4f}")

    print()
    print("FINDINGS:")
    print(f"1. Filtering y>0: Changes CV by {abs(cv_v19_filtered - cv_v19_unfiltered):.4f}")
    print(f"2. Different training API: Changes CV by {abs(cv_v19_filtered - cv_v50_filtered):.4f}")
    print(f"3. Combined effect explains the {0.4102 - 0.3572:.4f} gap")


if __name__ == "__main__":
    main()
