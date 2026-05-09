#!/usr/bin/env python3
"""XGBoost v11 - Top 12 Features Only

Based on feature importance analysis:
- Reduced from 29 to 12 features (59% reduction)
- Keeps 95% of predictive power
- Should improve generalization and reduce overfitting
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import mlflow

DATA_DIR = Path("data/store-sales-time-series-forecasting")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")

# Top 12 features from importance analysis
TOP_FEATURES = [
    "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "is_weekend",
    "Roll_mean_60", "day_of_week", "is_holiday", "Lag_3",
    "Roll_mean_90", "onpromotion", "Lag_7", "Roll_std_7"
]


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
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if is_train:
        # Only create features we actually use
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
    print("XGBoost v11 - Top 12 Features")
    print("="*70)
    print("59% feature reduction (29 → 12)")
    print("Targeting better generalization")
    print()

    train_df, test_df, holidays = load_data()

    print("Creating top 12 features...")
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)
    train_df = train_df.dropna()

    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    X = train_df[TOP_FEATURES]
    y = train_df["sales"]
    X_test = test_df[TOP_FEATURES]

    print(f"Features: {len(TOP_FEATURES)}, Samples: {len(X):,}")

    # TimeSeriesSplit CV
    print("\n" + "="*70)
    print("Training with 5-fold TimeSeriesSplit...")
    print("="*70)

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    # Use v10 params
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

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"\nFold {fold}/5 - Train: {len(train_idx):,}, Val: {len(val_idx):,}")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params, dtrain, num_boost_round=500,
            evals=[(dval, 'val')], early_stopping_rounds=50,
            verbose_eval=False
        )

        y_pred = model.predict(dval)
        y_pred = np.maximum(y_pred, 0)

        # Masked RMSLE
        mask = y_val > 0
        rmsle = np.sqrt(mean_squared_log_error(y_val[mask], y_pred[mask]))
        cv_scores.append(rmsle)
        print(f"  Fold {fold} RMSLE: {rmsle:.6f}")

    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    print(f"\n{'='*70}")
    print(f"CV Results: {mean_cv:.6f} (+/- {std_cv:.6f})")
    print(f"All folds: {[f'{s:.6f}' for s in cv_scores]}")
    print(f"{'='*70}")

    print(f"\nComparison:")
    print(f"  v10 (29 features): CV 0.370, LB 0.48159")
    print(f"  v11 (12 features): CV {mean_cv:.3f}")

    if mean_cv < 0.370:
        print(f"\n✅ IMPROVED! {((0.370 - mean_cv) / 0.370 * 100):.1f}% better")
    else:
        diff = mean_cv - 0.370
        print(f"\n⚠️ Slightly worse ({diff:.3f} higher)")

    # Train final model
    print("\nTraining final model...")
    dtrain_full = xgb.DMatrix(X, label=y)
    final_model = xgb.train(params, dtrain_full, num_boost_round=500)

    # Predict
    dtest = xgb.DMatrix(X_test)
    test_preds = final_model.predict(dtest)
    test_preds = np.maximum(test_preds, 0)

    # Save
    submission = pd.DataFrame({"id": test_df["id"], "sales": test_preds})
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / f"xgb_v11_top12_{mean_cv:.4f}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\n✅ Saved: {submission_path.name}")
    print(f"📊 CV: {mean_cv:.6f}")

    # MLflow
    mlflow.set_experiment("store-sales-forecasting")
    with mlflow.start_run(run_name="xgb_v11_top12_features"):
        mlflow.log_param("model", "xgboost_v11")
        mlflow.log_param("n_features", len(TOP_FEATURES))
        mlflow.log_param("features", ",".join(TOP_FEATURES))
        mlflow.log_metric("cv_rmsle", mean_cv)
        mlflow.log_metric("cv_std", std_cv)

    return mean_cv


if __name__ == "__main__":
    cv_score = main()
