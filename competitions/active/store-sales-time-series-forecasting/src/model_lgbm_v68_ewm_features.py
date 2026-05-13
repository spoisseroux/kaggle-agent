#!/usr/bin/env python3
"""LightGBM v68 - Exponential Weighted Mean (EWM) Features

BREAKTHROUGH HYPOTHESIS: Top scorer (2895 votes) uses EWM instead of simple rolling means

EWM advantages over rolling mean:
- Exponential decay gives more weight to recent values
- Less memory (infinite window but weighted)
- Better captures changing trends
- Multiple alphas capture different timescales

Pattern from comprehensive guide:
- Alphas: [0.95, 0.9, 0.8, 0.7, 0.5]
- Lags: [16, 30, 60, 90]
- Plus lag [1, 16, 30, 60] basic features

Expected: LB 0.40-0.45 if EWM is the differentiator
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


def create_ewm_features(df):
    """Create EWM features following comprehensive guide pattern."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Basic lag features (1, 16, 30, 60)
    for lag in [1, 16, 30, 60]:
        df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

    # EWM features (alpha x lag combinations)
    alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
    lags = [16, 30, 60, 90]

    for alpha in alphas:
        for lag in lags:
            df[f"EWM_{int(alpha*100)}_lag{lag}"] = (
                df.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
            )

    # Day features (always available)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    print("="*70)
    print("LightGBM v68 - Exponential Weighted Mean (EWM) Features")
    print("="*70)
    print("Testing hypothesis: EWM is the secret sauce vs simple rolling means")
    print()

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    # Combine train + test for feature creation (test lags reference training)
    print("Creating EWM features...")
    test_df["sales"] = np.nan
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = create_ewm_features(combined)
    combined = add_holidays(combined, holidays)

    # Split back
    train_df = combined[combined["sales"].notna()].copy()
    test_df = combined[combined["sales"].isna()].copy()

    # Drop NaN rows from training
    train_df = train_df.dropna()

    # Feature columns
    feature_cols = [
        "onpromotion", "day_of_week", "is_weekend", "is_holiday",
        "Lag_1", "Lag_16", "Lag_30", "Lag_60"
    ]

    # Add all EWM features
    alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
    lags = [16, 30, 60, 90]
    for alpha in alphas:
        for lag in lags:
            feature_cols.append(f"EWM_{int(alpha*100)}_lag{lag}")

    print(f"Total features: {len(feature_cols)}")
    print(f"  - Basic: 4 (onpromotion, day_of_week, is_weekend, is_holiday)")
    print(f"  - Lags: 4 (1, 16, 30, 60)")
    print(f"  - EWM: {len(alphas) * len(lags)} ({len(alphas)} alphas x {len(lags)} lags)")
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

    # v19 hyperparameters (proven to work)
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

    print(f"\n{'='*70}")
    print("EWM FEATURES RESULTS")
    print(f"{'='*70}")
    print(f"v19 (rolling means):  CV 0.3572, LB 0.498")
    print(f"v68 (EWM features):   CV {holdout_cv:.4f}")
    print(f"Expected LB:          0.40-0.45 (if EWM is the differentiator)")
    print()

    # Train on full data
    print("Training on full dataset...")
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    # Test predictions
    print("Generating test predictions...")
    X_test = test_df[feature_cols]
    test_preds = final_model.predict(X_test)
    test_preds = np.maximum(test_preds, 0)

    # Save predictions for correlation check
    np.save(PREDICTIONS_DIR / "v68_ewm_test.npy", test_preds)
    print(f"Saved predictions: {PREDICTIONS_DIR / 'v68_ewm_test.npy'}")

    # Correlation with v19
    v19_preds = np.load(PREDICTIONS_DIR / "v19_lgbm_test.npy")
    correlation = np.corrcoef(test_preds, v19_preds)[0, 1]

    print(f"\nCorrelation with v19: {correlation:.4f}")
    print(f"Mean prediction: {np.mean(test_preds):.1f} (v19: {np.mean(v19_preds):.1f})")

    # Save submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": test_preds
    })

    cv_str = f"{holdout_cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"lgbm_v68_ewm_features_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # Hypothesis DB
    if HAS_DB:
        try:
            db = HypothesisDatabase()
            hyp_id = db.add_hypothesis(
                competition="store-sales-time-series-forecasting",
                experiment_id="v68",
                hypothesis="EWM (Exponential Weighted Mean) features are the secret sauce used by top scorers",
                rationale="Comprehensive guide (2895 votes) uses EWM with 5 alphas x 4 lags instead of simple rolling means. EWM gives exponential decay weight to recent values, better captures changing trends.",
                category="feature_engineering"
            )
            db.record_result(
                hypothesis_id=hyp_id,
                cv_score=holdout_cv,
                lb_score=None,
                baseline_cv=0.3572,
                succeeded=(holdout_cv < 0.40),
                params=params,
                metadata={
                    "num_features": len(feature_cols),
                    "ewm_alphas": alphas,
                    "ewm_lags": lags,
                    "source": "comprehensive_guide_2895_votes",
                    "correlation_v19": float(correlation)
                }
            )
        except Exception as e:
            print(f"Note: Could not log: {e}")

    return {
        "holdout_cv": holdout_cv,
        "vs_v19": holdout_cv - 0.3572,
        "correlation": correlation,
        "expected_lb": "0.40-0.45",
        "success": holdout_cv < 0.40
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
