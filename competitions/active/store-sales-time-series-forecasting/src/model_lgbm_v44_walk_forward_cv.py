#!/usr/bin/env python3
"""LightGBM v44 - Walk-Forward Cross-Validation

STRATEGY: Apply walk-forward CV to best model (LightGBM)
- v19: Single 30-day holdout, CV 0.3572, LB 0.498
- v43: XGBoost walk-forward, CV 0.4490 (worse than v19)
- v44: LightGBM walk-forward (test if CV strategy improves best model)

Benefits:
- Reduces bias from single Aug 2017 validation window
- Multiple temporal folds span different market conditions
- More robust CV estimate
- Should reduce the 39.5% CV-LB gap from v19

Implementation:
- 3 temporal folds (May, Jun, Jul-Aug)
- Same v1 features as v19
- Same LightGBM hyperparameters as v19
- Average CV scores across folds
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from core.langfuse_logger import log_kaggle_model
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False

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


def create_features(df, is_train=True, train_df=None):
    """v1 proven feature set"""
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
    print("LightGBM v44 - Walk-Forward Cross-Validation")
    print("="*70)
    print("Strategy: Multiple temporal folds with best model (LightGBM)")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-lightgbm-v1")

    print("Loading data...")
    train_df, test_df, holidays = load_data()

    print("Creating v1 features...")
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)

    exclude_cols = ["id", "date", "sales", "store_nbr", "family"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Features: {len(feature_cols)} (v1 proven set)")
    print()

    # LightGBM parameters (same as v19)
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
        "random_state": 42,
        "verbosity": -1,
    }

    print("LightGBM parameters (v19 proven):")
    for k, v in params.items():
        if k not in ["verbosity"]:
            print(f"  {k}: {v}")
    print()

    # Walk-forward CV: 3 folds with 30-day validation each
    max_date = train_df['date'].max()
    n_folds = 3
    fold_size_days = 30

    cv_scores = []
    fold_results = []

    print("Walk-Forward Cross-Validation:")
    print(f"Number of folds: {n_folds}")
    print(f"Validation size per fold: {fold_size_days} days")
    print()

    for fold in range(n_folds):
        # Calculate cutoff for this fold
        days_back = (n_folds - fold) * fold_size_days
        cutoff_date = max_date - pd.Timedelta(days=days_back)
        val_end_date = cutoff_date + pd.Timedelta(days=fold_size_days)

        # Create train/val split for this fold
        train_mask = train_df['date'] < cutoff_date
        val_mask = (train_df['date'] >= cutoff_date) & (train_df['date'] < val_end_date)

        X = train_df[feature_cols]
        y = train_df["sales"]
        X_train_fold = X[train_mask]
        y_train_fold = y[train_mask]
        X_val_fold = X[val_mask]
        y_val_fold = y[val_mask]

        print(f"Fold {fold+1}:")
        print(f"  Train: {cutoff_date.date()} and before ({len(X_train_fold):,} samples)")
        print(f"  Val:   {cutoff_date.date()} to {val_end_date.date()} ({len(X_val_fold):,} samples)")
        print(f"  Val mean sales: {y_val_fold.mean():.2f}")

        # Train model for this fold
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )

        # Validate
        val_preds = model.predict(X_val_fold, num_iteration=model.best_iteration)
        val_preds = np.maximum(val_preds, 0)

        fold_cv = np.sqrt(mean_squared_log_error(y_val_fold, val_preds))
        cv_scores.append(fold_cv)

        fold_results.append({
            'fold': fold + 1,
            'train_end': cutoff_date.date(),
            'val_start': cutoff_date.date(),
            'val_end': val_end_date.date(),
            'cv_score': fold_cv,
            'val_mean_sales': y_val_fold.mean()
        })

        print(f"  CV: {fold_cv:.4f}")
        print()

    # Calculate average CV
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    print("="*70)
    print("WALK-FORWARD CV RESULTS:")
    print("="*70)
    for result in fold_results:
        print(f"Fold {result['fold']}: {result['cv_score']:.4f} (val period: {result['val_start']} to {result['val_end']})")
    print()
    print(f"Mean CV: {mean_cv:.4f}")
    print(f"Std CV:  {std_cv:.4f}")
    print(f"v19 (single 30-day holdout): 0.3572")
    print()

    # Train final model on all training data
    print("Training final model on all training data...")
    X_full = train_df[feature_cols]
    y_full = train_df["sales"]

    train_data = lgb.Dataset(X_full, label=y_full)

    with mlflow.start_run(run_name="lgbm_v44_walk_forward"):
        mlflow.log_params(params)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("cv_strategy", "walk_forward")
        mlflow.log_param("n_folds", n_folds)
        mlflow.log_param("fold_size_days", fold_size_days)

        final_model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )

        mlflow.log_metric("mean_cv", mean_cv)
        mlflow.log_metric("std_cv", std_cv)
        mlflow.log_metric("vs_v19", mean_cv - 0.3572)

    print()
    print("Generating test predictions...")
    X_test = test_df[feature_cols]
    test_preds = final_model.predict(X_test, num_iteration=final_model.best_iteration)
    test_preds = np.maximum(test_preds, 0)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": test_preds
    })

    filename = f"lgbm_v44_walkforward_{mean_cv:.5f}.csv".replace(".", "")
    filepath = SUBMISSION_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"Submission saved: {filename}")
    print(f"Mean prediction: {test_preds.mean():.2f}")
    print()

    if HAS_LANGFUSE:
        try:
            log_kaggle_model(
                competition="store-sales-time-series-forecasting",
                model_name="LightGBM v44 Walk-Forward",
                cv_score=mean_cv,
                model_type="lightgbm",
                params=params,
                features=feature_cols
            )
        except Exception:
            pass

    print("="*70)
    print("ANALYSIS:")
    print(f"Walk-forward CV with LightGBM (best model)")
    print(f"Mean CV: {mean_cv:.4f} ± {std_cv:.4f}")
    if mean_cv < 0.3572:
        print(f"✓ IMPROVED: {(0.3572 - mean_cv) / 0.3572 * 100:.1f}% better than v19")
    else:
        print(f"✗ WORSE: {(mean_cv - 0.3572) / 0.3572 * 100:.1f}% worse than v19")
    print(f"Validation periods span different market conditions")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
