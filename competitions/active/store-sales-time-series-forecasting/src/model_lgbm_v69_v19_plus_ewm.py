#!/usr/bin/env python3
"""LightGBM v69 - v19 Features + EWM (Additive, Not Replacement)

CORRECTED HYPOTHESIS: EWM features ADD signal to existing features

v68 failed (CV 1.3507) because it REPLACED v19 features with EWM.
v69 ADDS EWM features to proven v19 feature set.

v19 features (proven):
- Lag 3, 7
- Rolling means 7/14/30/60/90
- Rolling std 7
- day_of_week, is_weekend, is_holiday, onpromotion

v69 additions:
- EWM with shorter lags: [3, 7, 14] instead of [16, 30, 60, 90]
- Fewer alphas: [0.95, 0.9, 0.8] instead of [0.95, 0.9, 0.8, 0.7, 0.5]
- Total: 9 EWM features (3 alphas x 3 lags)

Expected: CV 0.33-0.35 if EWM adds signal
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
PREDICTIONS_DIR = Path("competitions/active/store-sales-time-series-forecasting/predictions")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
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
    """Create v19 features + EWM additions."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # v19 features (PROVEN)
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

    # NEW: EWM features (additive)
    # Use shorter lags (3, 7, 14) to match v19's timeframes
    alphas = [0.95, 0.9, 0.8]
    lags = [3, 7, 14]

    for alpha in alphas:
        for lag in lags:
            df[f"EWM_{int(alpha*100)}_lag{lag}"] = (
                df.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
            )

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    print("="*70)
    print("LightGBM v69 - v19 Features + EWM (Additive)")
    print("="*70)
    print("Testing: EWM adds signal on top of proven v19 features")
    print()

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    # Create features for training
    print("Creating features (v19 + EWM)...")
    train_df = create_features(train_df)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    # Test features
    test_df["sales"] = np.nan  # Placeholder for fillna
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = create_features(combined)
    combined = add_holidays(combined, holidays)
    test_df = combined[combined["id"].isin(test_df["id"])].copy()

    # Fill test NaN with store-family mean from last 30 days
    lag_cols = ["Lag_3", "Lag_7"] + [f"EWM_{int(a*100)}_lag{l}" for a in [0.95, 0.9, 0.8] for l in [3, 7, 14]]
    for col in lag_cols:
        test_df[col] = test_df.groupby(["store_nbr", "family"], observed=True)[col].fillna(method="ffill")
        test_df[col] = test_df[col].fillna(0)

    # Feature columns
    feature_cols = [
        # v19 features (12)
        "onpromotion", "day_of_week", "is_weekend", "is_holiday",
        "Lag_3", "Lag_7",
        "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "Roll_mean_60", "Roll_mean_90",
        "Roll_std_7"
    ]

    # Add EWM features (9)
    for alpha in [0.95, 0.9, 0.8]:
        for lag in [3, 7, 14]:
            feature_cols.append(f"EWM_{int(alpha*100)}_lag{lag}")

    print(f"Total features: {len(feature_cols)}")
    print(f"  - v19 features: 12")
    print(f"  - EWM features: 9 (3 alphas x 3 lags)")
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

    improvement_pct = (0.3572 - holdout_cv) / 0.3572 * 100

    print(f"\n{'='*70}")
    print("v19 + EWM RESULTS")
    print(f"{'='*70}")
    print(f"v19 (baseline):       CV 0.3572")
    print(f"v69 (v19 + EWM):      CV {holdout_cv:.4f} ({improvement_pct:+.1f}%)")
    print()

    if holdout_cv >= 0.3572:
        print("❌ EWM features did NOT help - no improvement")
    elif holdout_cv < 0.35:
        print("✅ SIGNIFICANT IMPROVEMENT - EWM adds valuable signal!")
    else:
        print("⚠️  Minor improvement - EWM adds some signal")

    # Train on full data
    print("\nTraining on full dataset...")
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
    submission_path = SUBMISSION_DIR / f"lgbm_v69_v19_plus_ewm_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # Hypothesis DB
    if HAS_DB:
        try:
            db = HypothesisDatabase()
            hyp_id = db.add_hypothesis(
                competition="store-sales-time-series-forecasting",
                experiment_id="v69",
                hypothesis="EWM features add signal on top of proven v19 features (additive, not replacement)",
                rationale="v68 failed by replacing v19 features with EWM. v69 adds EWM (3 alphas x 3 short lags) to v19's 12 features. Tests if EWM provides complementary signal.",
                category="feature_engineering"
            )
            db.record_result(
                hypothesis_id=hyp_id,
                cv_score=holdout_cv,
                lb_score=None,
                baseline_cv=0.3572,
                succeeded=(holdout_cv < 0.35),
                params=params,
                metadata={
                    "num_features": len(feature_cols),
                    "v19_features": 12,
                    "ewm_features": 9,
                    "improvement_pct": improvement_pct
                }
            )
        except Exception as e:
            print(f"Note: Could not log: {e}")

    return {
        "holdout_cv": holdout_cv,
        "vs_v19": holdout_cv - 0.3572,
        "improvement_pct": improvement_pct,
        "success": holdout_cv < 0.35
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
