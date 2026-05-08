"""Baseline model v1: LinearRegression with Time + Lag_1 features

Based on top notebooks - simple linear model with temporal features.
Target: <0.45 RMSLE on CV
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
import mlflow
import yaml

# Paths
DATA_DIR = Path("data/store-sales-time-series-forecasting")
CONFIG_DIR = Path("competitions/active/store-sales-time-series-forecasting/configs")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")

def load_data():
    """Load train and test data."""
    print("Loading data...")
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
        infer_datetime_format=True,
    )
    test = pd.read_csv(
        DATA_DIR / "test.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
        infer_datetime_format=True,
    )
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    return train, test


def create_features(df, is_train=True):
    """Create Time and Lag_1 features."""
    # Sort by store, family, date
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Time feature: sequential integer per store-family group
    df["Time"] = df.groupby(["store_nbr", "family"]).cumcount()

    if is_train:
        # Lag_1: previous day's sales per store-family
        df["Lag_1"] = df.groupby(["store_nbr", "family"])["sales"].shift(1)
    else:
        # For test, we need to join with last training values
        # For now, use 0 (will need proper implementation)
        df["Lag_1"] = 0  # TODO: Fix this with proper train-test lag handling

    return df


def train_model(X_train, y_train, X_val, y_val):
    """Train LinearRegression model."""
    print("\nTraining LinearRegression...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Validate
    y_pred_val = model.predict(X_val)
    y_pred_val = np.clip(y_pred_val, 0, None)  # Sales can't be negative

    # Calculate RMSLE
    rmsle = np.sqrt(mean_squared_log_error(y_val + 1, y_pred_val + 1))
    print(f"Validation RMSLE: {rmsle:.6f}")

    return model, rmsle


def main():
    """Run baseline experiment."""
    # MLflow setup
    mlflow.set_experiment("store-sales-baseline")

    with mlflow.start_run(run_name="baseline_v1_time_lag1"):
        # Load data
        train_df, test_df = load_data()

        # Create features
        train_df = create_features(train_df, is_train=True)
        test_df = create_features(test_df, is_train=False)

        # Remove NaN from lag (first row per group)
        train_df = train_df.dropna(subset=["Lag_1"])

        # Time-based split: last 16 days for validation (same as test period)
        split_date = train_df["date"].max() - pd.Timedelta(days=15)
        train_mask = train_df["date"] <= split_date
        val_mask = train_df["date"] > split_date

        X_train = train_df.loc[train_mask, ["Time", "Lag_1"]]
        y_train = train_df.loc[train_mask, "sales"]
        X_val = train_df.loc[val_mask, ["Time", "Lag_1"]]
        y_val = train_df.loc[val_mask, "sales"]

        print(f"\nTrain samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        # Log parameters
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("features", "Time,Lag_1")
        mlflow.log_param("val_days", 16)

        # Train
        model, rmsle = train_model(X_train, y_train, X_val, y_val)

        # Log metrics
        mlflow.log_metric("cv_rmsle", rmsle)

        # Generate submission
        print("\nGenerating submission...")
        X_test = test_df[["Time", "Lag_1"]]
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)

        submission = test_df[["id"]].copy()
        submission["sales"] = y_pred

        submission_path = SUBMISSION_DIR / "baseline_v1_submission.csv"
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to: {submission_path}")

        mlflow.log_artifact(str(submission_path))

        print(f"\n✅ Baseline v1 complete - CV RMSLE: {rmsle:.6f}")

        return rmsle


if __name__ == "__main__":
    main()
