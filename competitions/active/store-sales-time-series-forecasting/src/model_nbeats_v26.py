#!/usr/bin/env python3
"""N-BEATS v26 - Neural Basis Expansion for Time Series

STRATEGY: Deep learning approach to time series forecasting
- N-BEATS: SOTA architecture for univariate forecasting
- Learns from raw sequences, minimal feature engineering
- Expected: 5-15% improvement over LightGBM v19

Architecture:
- 2 stacks (trend + seasonality)
- Lookback: 30 days
- Forecast horizon: 16 days
- Per store-family models (or global with embeddings)

vs v19 LightGBM:
- v19: Hand-crafted features + gradient boosting
- v26: End-to-end learned representations
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import mlflow
import torch

# Add parent directory to path for core imports
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

try:
    from core.langfuse_logger import log_kaggle_model
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False
    print("⚠️  Langfuse logger not available")

DATA_DIR = Path("data/store-sales-time-series-forecasting")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def load_data():
    """Load competition data."""
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


def prepare_nbeats_data(train_df, test_df):
    """Prepare data for N-BEATS (requires specific format).

    N-BEATS expects:
    - unique_id: identifier for each time series
    - ds: timestamp
    - y: target value
    """
    # Create unique ID for each store-family combination
    train_df["unique_id"] = train_df["store_nbr"].astype(str) + "_" + train_df["family"].astype(str)
    test_df["unique_id"] = test_df["store_nbr"].astype(str) + "_" + test_df["family"].astype(str)

    # Rename columns for N-BEATS
    train_formatted = train_df[["unique_id", "date", "sales"]].copy()
    train_formatted.columns = ["unique_id", "ds", "y"]

    test_formatted = test_df[["unique_id", "date"]].copy()
    test_formatted.columns = ["unique_id", "ds"]

    return train_formatted, test_formatted


def main():
    print("="*70)
    print("N-BEATS v26 - Neural Time Series Forecasting")
    print("="*70)
    print("Deep learning approach with minimal feature engineering")
    print()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("store-sales-nbeats")

    # Load data
    print("Loading data...")
    train_df, test_df, holidays = load_data()

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Store-family combinations: {train_df.groupby(['store_nbr', 'family']).ngroups}")
    print()

    # Prepare for N-BEATS
    print("Formatting for N-BEATS...")
    train_formatted, test_formatted = prepare_nbeats_data(train_df, test_df)

    # 30-day holdout validation
    cutoff_date = train_formatted['ds'].max() - pd.Timedelta(days=30)
    val_mask = train_formatted['ds'] >= cutoff_date

    train_data = train_formatted[~val_mask].copy()
    val_data = train_formatted[val_mask].copy()

    print(f"Train period: {train_data['ds'].min()} to {train_data['ds'].max()}")
    print(f"Val period: {val_data['ds'].min()} to {val_data['ds'].max()}")
    print(f"Test period: {test_formatted['ds'].min()} to {test_formatted['ds'].max()}")
    print()

    # Train N-BEATS
    print("Training N-BEATS...")
    print("⚠️  This may take 1-2 hours on GPU")
    print()

    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS

        # N-BEATS configuration
        horizon = 16  # Forecast 16 days ahead

        model = NBEATS(
            h=horizon,
            input_size=30,  # Lookback 30 days
            stack_types=["trend", "seasonality"],
            n_blocks=[3, 3],
            mlp_units=[[512, 512], [512, 512]],
            learning_rate=1e-3,
            max_steps=1000,
            batch_size=32,
            windows_batch_size=128,
            scaler_type="robust",
        )

        nf = NeuralForecast(
            models=[model],
            freq='D',  # Daily frequency
        )

        # Fit on training data
        nf.fit(train_data)

        # Validate
        print("\nValidating on holdout...")
        val_preds = nf.predict()

        # Merge predictions with actuals
        val_results = val_data.merge(
            val_preds,
            on=["unique_id", "ds"],
            how="left"
        )

        # Calculate RMSLE
        mask = val_results["y"] > 0
        preds_clean = np.maximum(val_results["NBEATS"].values, 0)

        if mask.sum() > 0:
            holdout_cv = np.sqrt(mean_squared_log_error(
                val_results.loc[mask, "y"],
                preds_clean[mask]
            ))
        else:
            holdout_cv = float('inf')

        print(f"\nHoldout CV: {holdout_cv:.4f}")
        print(f"LightGBM v19: 0.3572")

        if holdout_cv < 0.3572:
            improvement = 0.3572 - holdout_cv
            print(f"✓ IMPROVEMENT: -{improvement:.4f} ({improvement/0.3572*100:.1f}%)")
        else:
            decline = holdout_cv - 0.3572
            print(f"✗ Declined: +{decline:.4f}")
        print()

        # Train on full data
        print("Training on full dataset...")
        nf_full = NeuralForecast(models=[model], freq='D')
        nf_full.fit(train_formatted)

        # Generate test predictions
        print("Generating test predictions...")
        test_preds = nf_full.predict()

        # Merge with test IDs
        submission = test_df[["id", "unique_id", "date"]].copy()
        submission.columns = ["id", "unique_id", "ds"]
        submission = submission.merge(test_preds, on=["unique_id", "ds"], how="left")

        # Clip negative predictions
        submission["sales"] = np.maximum(submission["NBEATS"].fillna(0), 0)
        final_submission = submission[["id", "sales"]]

        # Save submission
        cv_str = f"{holdout_cv:.4f}".replace(".", "")
        submission_path = SUBMISSION_DIR / f"nbeats_v26_{cv_str}.csv"
        final_submission.to_csv(submission_path, index=False)
        print(f"Saved: {submission_path}")

        # Log to MLflow
        with mlflow.start_run(run_name="nbeats_v26"):
            mlflow.log_param("architecture", "N-BEATS")
            mlflow.log_param("horizon", horizon)
            mlflow.log_param("input_size", 30)
            mlflow.log_param("stack_types", "trend+seasonality")
            mlflow.log_metric("holdout_cv", holdout_cv)
            mlflow.log_metric("vs_v19", holdout_cv - 0.3572)
            mlflow.log_artifact(str(submission_path))

        # Log to Langfuse
        if HAS_LANGFUSE:
            log_kaggle_model(
                name="v26",
                competition="store-sales-time-series-forecasting",
                model_type="N-BEATS",
                cv_score=holdout_cv,
                lb_score=None,
                features=None,  # End-to-end learning
                hyperparameters={
                    "horizon": horizon,
                    "input_size": 30,
                    "stacks": "trend+seasonality"
                },
                status="neural_baseline"
            )

        print()
        print("="*70)
        print("COMPLETE")
        print("="*70)
        print(f"Holdout CV: {holdout_cv:.4f}")
        print(f"Submission: {submission_path.name}")
        print("Ready for leaderboard submission")

        return {
            "holdout_cv": holdout_cv,
            "vs_v19": holdout_cv - 0.3572,
            "success": True
        }

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
