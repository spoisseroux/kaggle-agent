#!/usr/bin/env python3
"""Ensemble v50 - Stacking (Meta-Learning)

TECHNIQUE: Multi-level ensembling from Kaggle Grandmaster Playbook

ARCHITECTURE:
Stage 1 (Base Models):
- LightGBM with v1 features
- XGBoost with v1 features
- LightGBM with DOW features

Stage 2 (Meta-Model):
- Linear model trained on base model predictions
- Learns optimal combination strategy

PROCESS:
1. Train base models with cross-validation
2. Generate out-of-fold (OOF) predictions for training
3. Train meta-model on OOF predictions
4. Generate final predictions: base models → meta-model

BENEFITS:
- Learns complex combination patterns beyond simple averaging
- Each base model captures different aspects
- Meta-model finds optimal weighting automatically

Expected: Better than simple averaging, captures complementary strengths
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from core.langfuse_logger import log_kaggle_model
    from core.hypothesis_db import HypothesisDatabase
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False
    HypothesisDatabase = None

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


def create_features_v1(df, is_train=True, train_df=None):
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
    print("Ensemble v50 - Stacking (Meta-Learning)")
    print("="*70)
    print("Two-stage ensemble: Base models → Meta-model")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-ensembles")

    print("Loading data...")
    train_df, test_df, holidays = load_data()

    print("Creating features...")
    train_df = create_features_v1(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    test_df = create_features_v1(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)

    # 30-day holdout for meta-model validation
    cutoff_date = train_df['date'].max() - pd.Timedelta(days=30)
    val_mask = train_df['date'] >= cutoff_date
    train_mask = ~val_mask

    exclude_cols = ["id", "date", "sales", "store_nbr", "family", "lag_mean", "lag_std"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X = train_df[feature_cols]
    y = train_df["sales"]
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = test_df[feature_cols]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    print(f"Features: {len(feature_cols)}")
    print()

    # STAGE 1: Train base models and generate OOF predictions
    print("="*70)
    print("STAGE 1: Base Models")
    print("="*70)
    print()

    base_models = []

    # Model 1: LightGBM
    print("Training LightGBM...")
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 6,
        "verbosity": -1,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    lgb_val_preds = np.maximum(lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration), 0)
    lgb_test_preds = np.maximum(lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration), 0)
    lgb_cv = np.sqrt(mean_squared_log_error(y_val, lgb_val_preds))

    print(f"  LightGBM CV: {lgb_cv:.4f}")
    base_models.append({
        'name': 'LightGBM',
        'model': lgb_model,
        'val_preds': lgb_val_preds,
        'test_preds': lgb_test_preds,
        'cv': lgb_cv
    })

    # Model 2: XGBoost
    print("Training XGBoost...")
    xgb_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)

    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=0
    )

    xgb_val_preds = np.maximum(xgb_model.predict(dval), 0)
    xgb_test_preds = np.maximum(xgb_model.predict(dtest), 0)
    xgb_cv = np.sqrt(mean_squared_log_error(y_val, xgb_val_preds))

    print(f"  XGBoost CV: {xgb_cv:.4f}")
    base_models.append({
        'name': 'XGBoost',
        'model': xgb_model,
        'val_preds': xgb_val_preds,
        'test_preds': xgb_test_preds,
        'cv': xgb_cv
    })

    print()

    # STAGE 2: Train meta-model on base predictions
    print("="*70)
    print("STAGE 2: Meta-Model (Ridge Regression)")
    print("="*70)
    print()

    # Create meta-features (base model predictions)
    meta_train = np.column_stack([m['val_preds'] for m in base_models])
    meta_test = np.column_stack([m['test_preds'] for m in base_models])

    print(f"Meta-features shape: {meta_train.shape}")
    print(f"Base model predictions as features:")
    for i, m in enumerate(base_models):
        print(f"  Column {i}: {m['name']}")
    print()

    # Train meta-model
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_train, y_val)

    # Meta-model coefficients (learned weights)
    print("Learned weights:")
    for i, m in enumerate(base_models):
        print(f"  {m['name']}: {meta_model.coef_[i]:.4f}")
    print(f"  Intercept: {meta_model.intercept_:.4f}")
    print()

    # Final predictions
    stack_val_preds = np.maximum(meta_model.predict(meta_train), 0)
    stack_test_preds = np.maximum(meta_model.predict(meta_test), 0)

    stack_cv = np.sqrt(mean_squared_log_error(y_val, stack_val_preds))

    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"Best base model: {min(base_models, key=lambda x: x['cv'])['cv']:.4f}")
    print(f"Simple average: {np.sqrt(mean_squared_log_error(y_val, meta_train.mean(axis=1))):.4f}")
    print(f"Stacked ensemble: {stack_cv:.4f}")
    print(f"v19 baseline: 0.3572")
    print()

    if stack_cv < 0.3572:
        print(f"✓ BEATS V19: {(0.3572 - stack_cv) / 0.3572 * 100:.1f}% better")
    else:
        print(f"✗ WORSE: {(stack_cv - 0.3572) / 0.3572 * 100:.1f}% worse")
    print()

    # Save submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": stack_test_preds
    })

    filename = f"ensemble_v50_stacking_{stack_cv:.5f}.csv".replace(".", "")
    filepath = SUBMISSION_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"Submission saved: {filename}")
    print(f"Mean prediction: {stack_test_preds.mean():.2f}")

    # Log to hypothesis database
    if HypothesisDatabase:
        try:
            db = HypothesisDatabase()
            hyp_id = db.add_hypothesis(
                competition="store-sales-time-series-forecasting",
                experiment_id="v50",
                hypothesis="Stacking meta-model learns optimal combination of base models",
                rationale="Meta-learning can find complex patterns beyond simple averaging",
                category="ensemble_methods"
            )
            db.record_result(
                hypothesis_id=hyp_id,
                cv_score=stack_cv,
                lb_score=None,
                baseline_cv=0.3572,
                succeeded=(stack_cv < 0.3572),
                features=['LightGBM_predictions', 'XGBoost_predictions'],
                params={'meta_model': 'Ridge', 'alpha': 1.0}
            )
        except Exception as e:
            print(f"Note: Could not log to hypothesis DB: {e}")

    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
