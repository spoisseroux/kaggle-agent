#!/usr/bin/env python3
"""Ensemble v87 - v67 Architecture on Best Validation Window

Tests hypothesis: Using representative validation window (Mar 3-Apr 2, 2014)
will give CV scores closer to LB.

Current validation (Jul 16-Aug 15): +32% deviation, CV 0.39 → LB 0.45 (16% gap)
Best validation (Mar 3-Apr 2, 2014): -0.14% deviation, CV ??? → LB 0.45 (expect <5% gap)

If successful: Can retry feature engineering with confidence.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb

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
    """v67 baseline features"""
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


def main():
    print("="*70)
    print("Ensemble v87 - v67 on Best Validation Window")
    print("="*70)
    print("Validation: Mar 3 - Apr 2, 2014 (mean 357, -0.14% deviation)")
    print("Expected: CV closer to LB 0.454 (vs current 16% gap)")
    print()

    train_df, test_df, holidays = load_data()
    train_df = create_features(train_df)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    # Test features
    original_test_df = test_df.copy()
    test_df = test_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    original_test_ids = set(test_df["id"])

    test_df["sales"] = np.nan
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = create_features(combined)
    combined = add_holidays(combined, holidays)

    test_df = combined[combined["id"].isin(original_test_ids)].copy()

    # Fill NaN lags
    lag_cols = ["Lag_3", "Lag_7"]
    for col in lag_cols:
        test_df[col] = test_df.groupby(["store_nbr", "family"], observed=True)[col].ffill()
        test_df[col] = test_df[col].fillna(0)

    feature_cols = [
        "onpromotion", "day_of_week", "is_weekend", "is_holiday",
        "Lag_3", "Lag_7",
        "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "Roll_mean_60", "Roll_mean_90",
        "Roll_std_7"
    ]

    # BEST VALIDATION WINDOW: Mar 3 - Apr 2, 2014
    val_start = pd.Timestamp("2014-03-03")
    val_end = pd.Timestamp("2014-04-02")

    val_mask = (train_df["date"] >= val_start) & (train_df["date"] <= val_end)
    train_mask = ~val_mask

    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, "sales"]
    X_val = train_df.loc[val_mask, feature_cols]
    y_val = train_df.loc[val_mask, "sales"]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"Val period mean: {y_val.mean():.2f} (overall mean: 357.78)")
    print()

    # Train models (v67 architecture)
    print("Training LightGBM...")
    lgb_params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 6,
        "n_estimators": 600,
        "random_state": 42,
        "verbosity": -1
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    print("Training XGBoost...")
    xgb_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 600,
        "random_state": 42
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)

    # Validation
    lgb_val = np.maximum(lgb_model.predict(X_val), 0)
    xgb_val = np.maximum(xgb_model.predict(X_val), 0)

    # Ridge 90/10 ensemble
    meta_train = np.column_stack([lgb_val, xgb_val])
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(meta_train, y_val)
    intercept = ridge.intercept_

    val_preds = 0.90 * lgb_val + 0.10 * xgb_val + intercept
    val_preds = np.maximum(val_preds, 0)
    cv = np.sqrt(mean_squared_log_error(y_val, val_preds))

    print()
    print("Results:")
    print(f"  CV on best window: {cv:.4f}")
    print(f"  v67 CV on old window: 0.3908")
    print(f"  v67 LB: 0.45416")
    print(f"  Expected LB: ~0.45 (if window is representative)")
    print(f"  CV-LB gap: {((cv - 0.45416) / 0.45416 * 100):+.1f}% (v67 had +16.4%)")
    print(f"  Intercept: {intercept:.4f}")
    print()

    # Analysis
    if abs(cv - 0.454) < 0.05:  # Within 5% of LB
        print("✓ SUCCESS! CV matches LB - validation window is representative")
        print("  → Can now trust CV scores for feature engineering")
        print("  → Retry v81/v86 features on this window")
    else:
        print("⚠ CV still far from LB - may need different approach")
        print(f"  Gap: {abs(cv - 0.454):.4f}")

    print()

    # Test predictions
    X_test = test_df[feature_cols]
    lgb_test = np.maximum(lgb_model.predict(X_test), 0)
    xgb_test = np.maximum(xgb_model.predict(X_test), 0)
    test_preds = 0.90 * lgb_test + 0.10 * xgb_test + intercept
    test_preds = np.maximum(test_preds, 0)

    print("Test predictions:")
    print(f"  Mean: {test_preds.mean():.2f}")
    print(f"  Median: {np.median(test_preds):.2f}")
    print(f"  Std: {test_preds.std():.2f}")
    print()

    # Verify submission format
    assert len(test_preds) == len(original_test_df), \
        f"Wrong number of predictions: {len(test_preds)} vs {len(original_test_df)}"

    # Submission
    submission = pd.DataFrame({
        "id": test_df["id"].astype(int),
        "sales": test_preds
    })

    cv_str = f"{cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"ensemble_v87_best_val_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")
    print("✓ v87 complete")


if __name__ == "__main__":
    main()
