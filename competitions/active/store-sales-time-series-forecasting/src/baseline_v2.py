"""Baseline model v2: LinearRegression with comprehensive features

Improvements over v1:
- More lag features (Lag_1, Lag_2, Lag_3, Lag_7, Lag_14)
- Rolling statistics (7-day, 30-day, 90-day means)
- Seasonality features (day_of_week, month, etc.)
- Holiday indicators
- Proper lag handling for test data

Target: <0.50 RMSLE (hopefully better)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
import mlflow

# Paths
DATA_DIR = Path("data/store-sales-time-series-forecasting")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")


def load_data():
    """Load all datasets."""
    print("Loading data...")
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

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Holidays shape: {holidays.shape}")

    return train, test, holidays


def create_features(df, is_train=True, train_df=None):
    """Create comprehensive time series features."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Time feature
    df["Time"] = df.groupby(["store_nbr", "family"], observed=True).cumcount()

    # Seasonality features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if is_train:
        # Lag features
        for lag in [1, 2, 3, 7, 14]:
            df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

        # Rolling statistics
        for window in [7, 30, 90]:
            df[f"Roll_mean_{window}"] = (
                df.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )

        # Promotion lag
        df["onpromotion_lag1"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(1)
        df["onpromotion_lag1"] = df["onpromotion_lag1"].fillna(0)

    else:
        # For test data, we need to use last training values
        if train_df is not None:
            # Calculate average sales from last 30 days per store-family
            last_30_stats = (
                train_df.groupby(["store_nbr", "family"], observed=True)
                .tail(30)
                .groupby(["store_nbr", "family"], observed=True)["sales"]
                .mean()
                .reset_index()
                .rename(columns={"sales": "lag_fill"})
            )

            # Merge with test data
            df = df.merge(last_30_stats, on=["store_nbr", "family"], how="left")

            # Fill lag features with average
            for lag in [1, 2, 3, 7, 14]:
                df[f"Lag_{lag}"] = df["lag_fill"].fillna(0)

            # Rolling features: use same average
            for window in [7, 30, 90]:
                df[f"Roll_mean_{window}"] = df["lag_fill"].fillna(0)

            # Promotion lag
            df["onpromotion_lag1"] = df["onpromotion"]

            df = df.drop(columns=["lag_fill"])
        else:
            # Fallback: fill with 0
            for lag in [1, 2, 3, 7, 14]:
                df[f"Lag_{lag}"] = 0
            for window in [7, 30, 90]:
                df[f"Roll_mean_{window}"] = 0
            df["onpromotion_lag1"] = 0

    return df


def add_holidays(df, holidays):
    """Add holiday indicator."""
    # National holidays only (most impactful)
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def train_model(X_train, y_train, X_val, y_val):
    """Train LinearRegression model."""
    print("\nTraining LinearRegression...")
    print(f"Features ({len(X_train.columns)}): {list(X_train.columns)}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Validate
    y_pred_val = model.predict(X_val)
    y_pred_val = np.clip(y_pred_val, 0, None)

    # Calculate RMSLE
    rmsle = np.sqrt(mean_squared_log_error(y_val + 1, y_pred_val + 1))
    print(f"Validation RMSLE: {rmsle:.6f}")

    return model, rmsle


def main():
    """Run improved baseline experiment."""
    mlflow.set_experiment("store-sales-baseline")

    with mlflow.start_run(run_name="baseline_v2_full_features"):
        # Load data
        train_df, test_df, holidays = load_data()

        # Create features
        train_df = create_features(train_df, is_train=True)
        test_df = create_features(test_df, is_train=False, train_df=train_df)

        # Add holidays
        train_df = add_holidays(train_df, holidays)
        test_df = add_holidays(test_df, holidays)

        # Remove NaN from lags
        train_df = train_df.dropna()

        print(f"\nAfter feature engineering:")
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")

        # Time-based split
        split_date = train_df["date"].max() - pd.Timedelta(days=15)
        train_mask = train_df["date"] <= split_date
        val_mask = train_df["date"] > split_date

        feature_cols = [
            "Time",
            "day_of_week", "day_of_month", "month", "is_month_end", "is_month_start", "is_weekend",
            "Lag_1", "Lag_2", "Lag_3", "Lag_7", "Lag_14",
            "Roll_mean_7", "Roll_mean_30", "Roll_mean_90",
            "onpromotion", "onpromotion_lag1",
            "is_holiday",
        ]

        X_train = train_df.loc[train_mask, feature_cols]
        y_train = train_df.loc[train_mask, "sales"]
        X_val = train_df.loc[val_mask, feature_cols]
        y_val = train_df.loc[val_mask, "sales"]

        print(f"\nTrain samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        # Log parameters
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("val_days", 16)

        # Train
        model, rmsle = train_model(X_train, y_train, X_val, y_val)

        # Log metrics
        mlflow.log_metric("cv_rmsle", rmsle)

        # Generate submission
        print("\nGenerating submission...")
        X_test = test_df[feature_cols]
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)

        submission = test_df[["id"]].copy()
        submission["sales"] = y_pred

        submission_path = SUBMISSION_DIR / "baseline_v2_submission.csv"
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to: {submission_path}")

        mlflow.log_artifact(str(submission_path))

        print(f"\n✅ Baseline v2 complete - CV RMSLE: {rmsle:.6f}")

        return rmsle


if __name__ == "__main__":
    main()
