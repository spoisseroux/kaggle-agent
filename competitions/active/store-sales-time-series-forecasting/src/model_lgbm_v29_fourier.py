#!/usr/bin/env python3
"""LightGBM v29 - Fourier Transform Features

STRATEGY: Capture periodic patterns via frequency domain
- FFT (Fast Fourier Transform) on rolling windows
- Extract dominant frequencies
- Sin/cos transforms for cyclical features
- Expected: 3-7% improvement on v19

Features added:
- FFT coefficients from 7, 14, 30-day windows
- Dominant frequency components
- Phase information
- Original v1 features retained

vs v19: Same LightGBM, better features
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from scipy import fft
import lightgbm as lgb
import mlflow

# Add parent directory to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

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


def fourier_features(series, n_components=3):
    """Extract Fourier features from a time series.

    Args:
        series: Time series data
        n_components: Number of frequency components to extract

    Returns:
        Dictionary of Fourier features
    """
    if len(series) < 4 or series.isna().all():
        return {f'fft_mag_{i}': 0 for i in range(n_components)}

    # Remove NaN and normalize
    series_clean = series.dropna()
    if len(series_clean) < 4:
        return {f'fft_mag_{i}': 0 for i in range(n_components)}

    # FFT
    fft_vals = fft.fft(series_clean.values)
    fft_mag = np.abs(fft_vals)

    # Extract top n_components (excluding DC component)
    features = {}
    for i in range(min(n_components, len(fft_mag)//2)):
        features[f'fft_mag_{i}'] = fft_mag[i+1]  # Skip DC

    # Pad if needed
    for i in range(len(features), n_components):
        features[f'fft_mag_{i}'] = 0

    return features


def create_features(df, is_train=True, train_df=None):
    """Create v1 features + Fourier features."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Original v1 features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if is_train:
        # Lag features
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

        # Fourier features on rolling windows
        print("Computing Fourier features (this may take a few minutes)...")
        for window in [7, 14, 30]:
            # Get rolling window data
            rolling = df.groupby(["store_nbr", "family"], observed=True)["sales"].rolling(
                window=window, min_periods=window//2
            )

            # Extract FFT features for each window
            fft_feats = []
            for idx, (name, group) in enumerate(df.groupby(["store_nbr", "family"], observed=True)):
                series = group["sales"]
                group_ffts = []

                for i in range(len(series)):
                    start = max(0, i - window + 1)
                    window_data = series.iloc[start:i+1]
                    feats = fourier_features(window_data, n_components=2)
                    group_ffts.append(feats)

                fft_feats.extend(group_ffts)

                if idx % 200 == 0:
                    print(f"  Processed {idx}/1782 store-family groups...")

            # Convert to DataFrame and merge
            fft_df = pd.DataFrame(fft_feats)
            for col in fft_df.columns:
                df[f"{col}_w{window}"] = fft_df[col].values

    else:
        # Test set - use train statistics
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

            # Lag and rolling features
            df["Lag_3"] = df["lag_mean"].fillna(0)
            df["Lag_7"] = df["lag_mean"].fillna(0)
            for window in [7, 14, 30, 60, 90]:
                df[f"Roll_mean_{window}"] = df["lag_mean"].fillna(0)
            df["Roll_std_7"] = df["lag_std"].fillna(0)

            # Fourier features from last 30 days
            print("Computing Fourier features for test set...")
            fft_feats_test = []
            for idx, (name, group) in enumerate(train_df.groupby(["store_nbr", "family"], observed=True)):
                last_sales = group.tail(30)["sales"]

                # Compute FFT features for different windows
                feats = {}
                for window in [7, 14, 30]:
                    window_data = last_sales.tail(window)
                    fft_f = fourier_features(window_data, n_components=2)
                    for k, v in fft_f.items():
                        feats[f"{k}_w{window}"] = v

                fft_feats_test.append((name[0], name[1], feats))

            # Create FFT dataframe
            fft_rows = []
            for store, family, feats in fft_feats_test:
                row = {"store_nbr": store, "family": family}
                row.update(feats)
                fft_rows.append(row)

            fft_df_test = pd.DataFrame(fft_rows)
            df = df.merge(fft_df_test, on=["store_nbr", "family"], how="left")

            # Fill any remaining NaN in FFT features
            fft_cols = [c for c in df.columns if 'fft' in c]
            for col in fft_cols:
                df[col] = df[col].fillna(0)

    return df


def add_holidays(df, holidays):
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    print("="*70)
    print("LightGBM v29 - Fourier Transform Features")
    print("="*70)
    print("Capturing periodic patterns via frequency domain")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-lightgbm-fourier")

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    # Create features
    print("Creating v1 + Fourier features...")
    train_df = create_features(train_df, is_train=True)
    train_df = add_holidays(train_df, holidays)
    train_df = train_df.dropna()

    test_df = create_features(test_df, is_train=False, train_df=train_df)
    test_df = add_holidays(test_df, holidays)

    # Feature columns
    exclude_cols = ["id", "date", "sales", "store_nbr", "family", "lag_mean", "lag_std"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"Features: {len(feature_cols)} (v1: 12, Fourier: {len(feature_cols)-12})")
    print()

    # 30-day holdout
    cutoff_date = train_df['date'].max() - pd.Timedelta(days=30)
    val_mask = train_df['date'] >= cutoff_date
    train_mask = ~val_mask

    X = train_df[feature_cols]
    y = train_df["sales"]
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    print()

    # v19 params (proven to work)
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

    print("Training LightGBM with Fourier features...")
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # Validation
    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)

    mask = y_val > 0
    if mask.sum() > 0:
        holdout_cv = np.sqrt(mean_squared_log_error(y_val[mask], preds[mask]))
    else:
        holdout_cv = float('inf')

    print(f"\nHoldout CV: {holdout_cv:.4f}")
    print(f"LightGBM v19 (baseline): 0.3572")

    if holdout_cv < 0.3572:
        improvement = 0.3572 - holdout_cv
        print(f"✓ IMPROVEMENT: -{improvement:.4f} ({improvement/0.3572*100:.1f}%)")
    else:
        decline = holdout_cv - 0.3572
        print(f"✗ Declined: +{decline:.4f}")
    print()

    # Train on full data
    print("Training on full dataset...")
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    # Generate predictions
    print("Generating test predictions...")
    X_test = test_df[feature_cols]
    predictions = final_model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    # Submission
    submission = pd.DataFrame({"id": test_df["id"], "sales": predictions})
    cv_str = f"{holdout_cv:.4f}".replace(".", "")
    submission_path = SUBMISSION_DIR / f"lgbm_v29_fourier_{cv_str}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")

    # MLflow
    with mlflow.start_run(run_name="lgbm_v29_fourier"):
        mlflow.log_params(params)
        mlflow.log_metric("holdout_cv", holdout_cv)
        mlflow.log_metric("vs_v19", holdout_cv - 0.3572)
        mlflow.log_metric("num_features", len(feature_cols))
        mlflow.log_artifact(str(submission_path))

    # Langfuse
    if HAS_LANGFUSE:
        log_kaggle_model(
            name="v29",
            competition="store-sales-time-series-forecasting",
            model_type="LightGBM",
            cv_score=holdout_cv,
            lb_score=None,
            features=len(feature_cols),
            hyperparameters=params,
            status="fourier_features"
        )

    print()
    print("="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Holdout CV: {holdout_cv:.4f}")
    print(f"Submission: {submission_path.name}")

    return {
        "holdout_cv": holdout_cv,
        "vs_v19": holdout_cv - 0.3572,
        "success": True
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
