#!/usr/bin/env python3
"""LightGBM v35 - Hierarchical Models (FIXED)

FIXES v33 CRITICAL BUG: fillna(0) on training data → dropna()

v33 FAILURE ANALYSIS:
- v33 LB: 0.590 (worse than v19's 0.498)
- Root cause: Used fillna(0) for training lag features (lines 160, 162)
- Filling NaN lags with 0 means "zero sales history"
- Model learned to predict low when seeing 0 lags
- Result: Systematic 20% underprediction

v35 FIX:
- Drop rows with NaN lag features from training (like v19)
- Only train on rows with valid historical data
- Should match v19's prediction scale

Expected: LB ~0.47 (proper hierarchical approach)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import mlflow
from tqdm import tqdm

# Add parent directory to path
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


def create_features(df, is_train=True, train_df=None):
    """Create v1 proven features

    Args:
        df: DataFrame to create features for
        is_train: Whether this is training data
        train_df: Training data (used for test set lag/rolling feature approximation)
    """
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Date features (always available)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if is_train:
        # Training: calculate lag/rolling features from sales
        df["Lag_3"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(3)
        df["Lag_7"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(7)

        # Rolling means
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
        # Test: approximate lag/rolling features with training means (v19 approach)
        # Calculate mean and std for each store-family from training data
        lag_mean = train_df.groupby(["store_nbr", "family"], observed=True)["sales"].mean()
        lag_std = train_df.groupby(["store_nbr", "family"], observed=True)["sales"].std()

        # Map to test set
        df["lag_mean"] = df.set_index(["store_nbr", "family"]).index.map(lag_mean).values
        df["lag_std"] = df.set_index(["store_nbr", "family"]).index.map(lag_std).values

        # Fill lag/rolling features with approximations
        df["Lag_3"] = df["lag_mean"].fillna(0)
        df["Lag_7"] = df["lag_mean"].fillna(0)
        for window in [7, 14, 30, 60, 90]:
            df[f"Roll_mean_{window}"] = df["lag_mean"].fillna(0)
        df["Roll_std_7"] = df["lag_std"].fillna(0)

        # Drop temporary columns
        df = df.drop(columns=["lag_mean", "lag_std"])

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    print("="*70)
    print("LightGBM v35 - Hierarchical Models (FIXED)")
    print("="*70)
    print("Training separate model for each of 33 product families")
    print("FIX: Using dropna() for training data (not fillna(0))")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-lightgbm-hierarchical-fixed")

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    print(f"Training: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Test: {test_df['date'].min()} to {test_df['date'].max()}")
    print()

    # Create features
    print("Creating features...")
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)

    # Feature columns
    exclude_cols = ["id", "date", "sales", "store_nbr", "family"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"Features: {len(feature_cols)} (v1 proven set)")
    print()

    # Get unique families
    families = sorted(train_df['family'].unique())
    print(f"Product families: {len(families)}")
    print()

    # Store models and predictions
    family_models = {}
    family_cv_scores = {}
    all_val_predictions = []
    all_test_predictions = []

    # Train separate model for each family
    print("="*70)
    print("TRAINING HIERARCHICAL MODELS")
    print("="*70)

    for family in tqdm(families, desc="Training families"):
        # Filter data for this family
        train_family = train_df[train_df['family'] == family].copy()
        test_family = test_df[test_df['family'] == family].copy()

        # 30-day holdout validation
        cutoff_date = train_family['date'].max() - pd.Timedelta(days=30)
        val_mask = train_family['date'] >= cutoff_date
        train_mask = ~val_mask

        # FIX v33 BUG: Drop NaN rows instead of filling with 0 (like v19)
        train_subset = train_family.loc[train_mask].dropna(subset=feature_cols)
        val_subset = train_family.loc[val_mask].dropna(subset=feature_cols)

        X_train = train_subset[feature_cols]
        y_train = train_subset["sales"]
        X_val = val_subset[feature_cols]
        y_val = val_subset["sales"]

        # Train LightGBM (same hyperparameters as v19)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 63,  # depth=6 → 2^6-1 = 63
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
        }

        train_set = lgb.Dataset(X_train, y_train)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)],
        )

        # Validate
        val_preds = model.predict(X_val)
        val_preds = np.maximum(val_preds, 0)
        cv_score = np.sqrt(mean_squared_log_error(y_val, val_preds))

        # Store model and score
        family_models[family] = model
        family_cv_scores[family] = cv_score

        # Store validation predictions for overall CV
        val_data = train_family[val_mask].copy()
        val_data['predictions'] = val_preds
        all_val_predictions.append(val_data[['sales', 'predictions']])

        # Test predictions
        X_test = test_family[feature_cols].fillna(0)
        test_preds = model.predict(X_test)
        test_preds = np.maximum(test_preds, 0)

        test_data = test_family.copy()
        test_data['sales'] = test_preds
        all_test_predictions.append(test_data[['id', 'sales']])

    # Combine validation predictions for overall CV
    all_val = pd.concat(all_val_predictions, ignore_index=True)
    overall_cv = np.sqrt(mean_squared_log_error(all_val['sales'], all_val['predictions']))

    # Combine test predictions
    submission = pd.concat(all_test_predictions, ignore_index=True)
    submission = submission.sort_values('id').reset_index(drop=True)

    # Save submission
    cv_str = f"{overall_cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"lgbm_v35_hierarchical_fixed_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)

    # Print results
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Overall CV (30-day holdout): {overall_cv:.4f}")
    print()
    print("Per-family CV scores:")
    for family in sorted(family_cv_scores.keys(), key=lambda f: family_cv_scores[f]):
        print(f"  {family:35} {family_cv_scores[family]:.4f}")
    print()
    print(f"Saved: {submission_path.name}")

    # Log to MLflow
    with mlflow.start_run(run_name=f"lgbm_v35_hierarchical_fixed_{cv_str}"):
        mlflow.log_params({
            "model": "lightgbm_hierarchical_fixed",
            "num_families": len(families),
            "features": "v1_proven",
            "validation": "30day_holdout_dropna",
            "learning_rate": 0.05,
            "max_depth": 6,
        })
        mlflow.log_metrics({
            "cv_score": overall_cv,
            "cv_mean": overall_cv,
        })
        mlflow.log_artifact(str(submission_path))
        mlflow.set_tag("competition", "store-sales-time-series-forecasting")

    print()
    print(f"Submission: {submission_path.name}")
    print(f"Expected LB: ~0.47 (3-5% better than v19's 0.498)")


if __name__ == "__main__":
    main()
