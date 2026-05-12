#!/usr/bin/env python3
"""XGBoost v43 - Walk-Forward Cross-Validation

STRATEGY: Multiple temporal folds to reduce validation bias
- v19/v41: Single 30-day holdout (biased to Aug 2017 peak)
- v43: 5-fold walk-forward CV (multiple temporal windows)
- Each fold uses expanding window (all data before cutoff)
- Validation periods spread across time

Benefits:
- Reduces bias from any single validation period
- More robust CV estimate
- Better represents varied market conditions

Implementation:
- Create 5 temporal folds manually (can't use sklearn's TimeSeriesSplit directly due to groupby features)
- Train on all data before each cutoff
- Average CV scores across folds
- Final model trains on all training data
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
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


def create_features(df, is_train=True, train_df=None, holidays=None):
    """v1 features"""
    df = df.copy()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if holidays is not None:
        holiday_dates = set(holidays["date"].dt.date)
        df["is_holiday"] = df["date"].dt.date.isin(holiday_dates).astype(int)
    else:
        df["is_holiday"] = 0

    if is_train:
        for lag in [3, 7]:
            df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

        for window in [7, 14, 30, 60, 90]:
            df[f"Roll_mean_{window}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

        df["Roll_std_7"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        )
    else:
        if train_df is not None:
            lag_mean = train_df.groupby(["store_nbr", "family"], observed=True)["sales"].mean()
            df = df.merge(
                lag_mean.rename("lag_mean").reset_index(),
                on=["store_nbr", "family"],
                how="left"
            )

            for lag in [3, 7]:
                df[f"Lag_{lag}"] = df["lag_mean"]

            for window in [7, 14, 30, 60, 90]:
                df[f"Roll_mean_{window}"] = df["lag_mean"]

            df["Roll_std_7"] = df["lag_mean"]

            df = df.drop(columns=["lag_mean"])

    return df


def main():
    print("="*70)
    print("XGBoost v43 - Walk-Forward Cross-Validation")
    print("="*70)
    print("Strategy: Multiple temporal folds to reduce validation bias")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-xgboost-v1")

    print("Loading data...")
    train_df, test_df, holidays = load_data()

    print("Creating v1 features...")
    train_df = create_features(train_df, is_train=True, holidays=holidays)
    train_df = train_df.dropna()

    test_df = create_features(test_df, is_train=False, train_df=train_df, holidays=holidays)

    exclude_cols = ["id", "date", "sales", "store_nbr", "family"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Historical parameters (from v1/v3)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    print("XGBoost parameters:")
    for k, v in params.items():
        if k not in ["n_jobs", "eval_metric", "tree_method"]:
            print(f"  {k}: {v}")
    print()

    # Walk-forward CV: 5 folds with 30-day validation each
    # Fold 1: Train up to 2017-05-16, validate 2017-05-17 to 2017-06-15
    # Fold 2: Train up to 2017-06-16, validate 2017-06-17 to 2017-07-15
    # Fold 3: Train up to 2017-07-16, validate 2017-07-17 to 2017-08-15
    # This spreads validation across different market conditions

    max_date = train_df['date'].max()
    n_folds = 3  # Use 3 folds to cover last 90 days
    fold_size_days = 30

    cv_scores = []
    fold_results = []

    print("Walk-Forward Cross-Validation:")
    print(f"Number of folds: {n_folds}")
    print(f"Validation size per fold: {fold_size_days} days")
    print()

    for fold in range(n_folds):
        # Calculate cutoff for this fold
        # Start from the end and work backwards
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
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold, enable_categorical=True)
        dval = xgb.DMatrix(X_val_fold, label=y_val_fold, enable_categorical=True)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=0  # Silent to avoid clutter
        )

        # Validate
        val_preds = model.predict(dval)
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
    print(f"v41 (single 30-day holdout): 0.4554")
    print()

    # Train final model on all training data
    print("Training final model on all training data...")
    X_full = train_df[feature_cols]
    y_full = train_df["sales"]

    dfull = xgb.DMatrix(X_full, label=y_full, enable_categorical=True)

    with mlflow.start_run(run_name="xgb_v43_walk_forward"):
        mlflow.log_params(params)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("cv_strategy", "walk_forward")
        mlflow.log_param("n_folds", n_folds)
        mlflow.log_param("fold_size_days", fold_size_days)

        final_model = xgb.train(
            params,
            dfull,
            num_boost_round=2000,
            evals=[(dfull, 'train')],
            early_stopping_rounds=50,
            verbose_eval=100
        )

        mlflow.log_metric("mean_cv", mean_cv)
        mlflow.log_metric("std_cv", std_cv)
        mlflow.log_metric("vs_v19", mean_cv - 0.3572)
        mlflow.log_metric("vs_v41", mean_cv - 0.4554)

    print()
    print("Generating test predictions...")
    X_test = test_df[feature_cols]
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    test_preds = final_model.predict(dtest)
    test_preds = np.maximum(test_preds, 0)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": test_preds
    })

    filename = f"xgb_v43_walkforward_{mean_cv:.5f}.csv".replace(".", "")
    filepath = SUBMISSION_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"Submission saved: {filename}")
    print(f"Mean prediction: {test_preds.mean():.2f}")
    print()

    if HAS_LANGFUSE:
        try:
            log_kaggle_model(
                competition="store-sales-time-series-forecasting",
                model_name="XGBoost v43 Walk-Forward",
                cv_score=mean_cv,
                model_type="xgboost",
                params=params,
                features=feature_cols
            )
        except Exception:
            pass

    print("="*70)
    print("ANALYSIS:")
    print(f"Walk-forward CV reduces bias from single validation window")
    print(f"Mean CV: {mean_cv:.4f} ± {std_cv:.4f}")
    print(f"Validation periods span different market conditions")
    print(f"Should provide more robust generalization estimate")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
