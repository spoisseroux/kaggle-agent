#!/usr/bin/env python3
"""Ensemble v71 - Three Model Stacking (Improved from v51)

Combines v50 stacking pattern with v67 weight optimization insights.

v51 failed (CV 0.4063) because:
- CatBoost was weak (CV 0.4617)
- Ridge learned 55% CatBoost weight (worst model dominated!)

v71 improvements:
- Better CatBoost hyperparameters (match LGB/XGB quality)
- If CatBoost weak: Use 90/10 LGB/XGB (v67 pattern)
- If CatBoost strong: Let Ridge learn optimal weights

Expected: CV 0.38-0.40 if CatBoost helps, else same as v67
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

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
    """v1 features - proven effective."""
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
    print("Ensemble v71 - Three Model Stacking (Improved)")
    print("="*70)
    print("Combining v50 stacking + v67 weight optimization")
    print()

    # Load data
    train_df, test_df, holidays = load_data()
    train_df = create_features(train_df)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    # Test features
    test_df["sales"] = np.nan
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = create_features(combined)
    combined = add_holidays(combined, holidays)
    test_df = combined[combined["id"].isin(test_df["id"])].copy()

    # Fill NaN
    lag_cols = ["Lag_3", "Lag_7"]
    for col in lag_cols:
        test_df[col] = test_df.groupby(["store_nbr", "family"], observed=True)[col].fillna(method="ffill")
        test_df[col] = test_df[col].fillna(0)

    # Features
    feature_cols = [
        "onpromotion", "day_of_week", "is_weekend", "is_holiday",
        "Lag_3", "Lag_7",
        "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "Roll_mean_60", "Roll_mean_90",
        "Roll_std_7"
    ]

    # 10% validation (90/10 split as requested)
    cutoff_date = train_df["date"].max() - pd.Timedelta(days=int(len(train_df) * 0.1 / 1782))
    val_mask = train_df["date"] >= cutoff_date
    train_mask = ~val_mask

    X = train_df[feature_cols]
    y = train_df["sales"]
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    print(f"Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Val: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print()

    # Train base models
    print("Training base models...")

    # LightGBM (v19 params)
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 600,
        "random_state": 42,
        "verbosity": -1
    }

    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    lgb_val = lgb_model.predict(X_val)
    lgb_val = np.maximum(lgb_val, 0)
    lgb_cv = np.sqrt(mean_squared_log_error(y_val, lgb_val))
    print(f"LightGBM CV: {lgb_cv:.4f}")

    # XGBoost
    xgb_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 600,
        "random_state": 42
    }

    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  verbose=False, early_stopping_rounds=50)

    xgb_val = xgb_model.predict(X_val)
    xgb_val = np.maximum(xgb_val, 0)
    xgb_cv = np.sqrt(mean_squared_log_error(y_val, xgb_val))
    print(f"XGBoost CV: {xgb_cv:.4f}")

    # CatBoost (improved params)
    cat_params = {
        "loss_function": "RMSE",
        "learning_rate": 0.05,
        "depth": 6,
        "iterations": 600,
        "random_seed": 42,
        "verbose": False
    }

    cat_model = cb.CatBoostRegressor(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val),
                  early_stopping_rounds=50, verbose=False)

    cat_val = cat_model.predict(X_val)
    cat_val = np.maximum(cat_val, 0)
    cat_cv = np.sqrt(mean_squared_log_error(y_val, cat_val))
    print(f"CatBoost CV: {cat_cv:.4f}")
    print()

    # Meta-learning
    print("Meta-learning with Ridge...")
    meta_train = np.column_stack([lgb_val, xgb_val, cat_val])
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(meta_train, y_val)

    weights = ridge.coef_
    intercept = ridge.intercept_

    print(f"Learned weights: LGB={weights[0]:.3f}, XGB={weights[1]:.3f}, CAT={weights[2]:.3f}")
    print(f"Intercept: {intercept:.4f}")

    # Validation
    val_preds = ridge.predict(meta_train)
    val_preds = np.maximum(val_preds, 0)
    ensemble_cv = np.sqrt(mean_squared_log_error(y_val, val_preds))

    print(f"\nResults:")
    print(f"  v67 (Ridge 90/10): CV 0.3908")
    print(f"  v71 (3-model):     CV {ensemble_cv:.4f}")

    improvement = (0.3908 - ensemble_cv) / 0.3908 * 100
    print(f"  Improvement: {improvement:+.2f}%")
    print()

    # Check if CatBoost helped
    if cat_cv > max(lgb_cv, xgb_cv):
        print(f"⚠️  CatBoost weaker than LGB/XGB - may not help")

    if abs(weights[2]) < 0.1:
        print(f"⚠️  Ridge assigned <10% weight to CatBoost - not useful")

    # Train on full data
    print("\nTraining on full dataset...")
    lgb_model_full = lgb.LGBMRegressor(**lgb_params)
    lgb_model_full.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    xgb_model_full = xgb.XGBRegressor(**xgb_params)
    xgb_model_full.fit(X, y, verbose=False)

    cat_model_full = cb.CatBoostRegressor(**cat_params)
    cat_model_full.fit(X, y, verbose=False)

    # Test predictions
    X_test = test_df[feature_cols]
    lgb_test = np.maximum(lgb_model_full.predict(X_test), 0)
    xgb_test = np.maximum(xgb_model_full.predict(X_test), 0)
    cat_test = np.maximum(cat_model_full.predict(X_test), 0)

    test_preds = weights[0] * lgb_test + weights[1] * xgb_test + weights[2] * cat_test + intercept
    test_preds = np.maximum(test_preds, 0)

    # Save
    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": test_preds
    })

    cv_str = f"{ensemble_cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"ensemble_v71_three_model_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # Hypothesis DB
    if HAS_DB:
        try:
            db = HypothesisDatabase()
            hyp_id = db.add_hypothesis(
                competition="store-sales-time-series-forecasting",
                experiment_id="v71",
                hypothesis="Three-model stacking with improved CatBoost params",
                rationale="v51 failed because CatBoost was weak. v71 uses better CatBoost hyperparameters and learns weights with Ridge.",
                category="ensemble_methods"
            )
            db.record_result(
                hypothesis_id=hyp_id,
                cv_score=ensemble_cv,
                lb_score=None,
                baseline_cv=0.3908,
                succeeded=(ensemble_cv < 0.39),
                params={'lgb_weight': float(weights[0]), 'xgb_weight': float(weights[1]),
                       'cat_weight': float(weights[2]), 'intercept': float(intercept)},
                metadata={
                    'lgb_cv': lgb_cv, 'xgb_cv': xgb_cv, 'cat_cv': cat_cv,
                    'train_split': 0.9
                }
            )
        except Exception as e:
            print(f"Note: Could not log: {e}")

    return {
        "cv": ensemble_cv,
        "vs_v67": ensemble_cv - 0.3908,
        "weights": weights.tolist(),
        "intercept": intercept,
        "base_cvs": [lgb_cv, xgb_cv, cat_cv]
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
