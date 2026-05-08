"""XGBoost v2: Fixed lag feature handling for test set

Key fix: Properly propagate predictions day-by-day for test set lags
instead of using simple means from training data.

This should close the CV/LB gap.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import mlflow
from tqdm import tqdm

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
    return train, test, holidays


def create_base_features(df):
    """Create non-lag features (seasonality, etc.)."""
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def create_lag_features_train(df):
    """Create lag and rolling features for training data."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    df["Time"] = df.groupby(["store_nbr", "family"], observed=True).cumcount()

    # Lag features
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

    # Rolling features
    for window in [7, 14, 30, 60, 90]:
        df[f"Roll_mean_{window}"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        df[f"Roll_std_{window}"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )

    # Promotion lags
    df["onpromotion_lag1"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(1).fillna(0)
    df["onpromotion_lag7"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(7).fillna(0)

    return df


def predict_test_with_lags(model, train_df, test_df, feature_cols):
    """
    Predict test set day-by-day, using previous predictions for lag features.

    This is the correct way to handle time series test predictions.
    """
    print("\nPredicting test set with proper lag propagation...")

    # Combine train and test for continuous time series
    train_df = train_df.copy()
    test_df = test_df.copy()
    test_df["sales"] = 0  # Placeholder

    # Get all unique dates in test set (sorted)
    test_dates = sorted(test_df["date"].unique())

    # Initialize predictions
    predictions = []

    # Predict day by day
    for test_date in tqdm(test_dates, desc="Predicting test days"):
        # Get rows for this date
        test_day_mask = test_df["date"] == test_date
        test_day = test_df[test_day_mask].copy()

        # Combine with historical data (train + already-predicted test days)
        historical = pd.concat([train_df, test_df[test_df["date"] < test_date]], ignore_index=True)
        combined = pd.concat([historical, test_day], ignore_index=True)

        # Sort and create lag features
        combined = combined.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
        combined["Time"] = combined.groupby(["store_nbr", "family"], observed=True).cumcount()

        # Create lag features
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            combined[f"Lag_{lag}"] = combined.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

        # Rolling features
        for window in [7, 14, 30, 60, 90]:
            combined[f"Roll_mean_{window}"] = (
                combined.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )
            combined[f"Roll_std_{window}"] = (
                combined.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))
            )

        # Promotion lags
        combined["onpromotion_lag1"] = combined.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(1).fillna(0)
        combined["onpromotion_lag7"] = combined.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(7).fillna(0)

        # Get only the rows for current test date
        test_day_features = combined[combined["date"] == test_date].copy()

        # Predict
        X_test_day = test_day_features[feature_cols]
        dtest = xgb.DMatrix(X_test_day)
        preds = model.predict(dtest, iteration_range=(0, model.best_iteration))
        preds = np.clip(preds, 0, None)

        # Store predictions back into test_df for next iteration
        test_df.loc[test_day_mask, "sales"] = preds

        # Save for final submission
        for idx, pred in zip(test_day_features["id"].values, preds):
            predictions.append({"id": idx, "sales": pred})

    return pd.DataFrame(predictions)


def add_holidays(df, holidays):
    """Add holiday indicator."""
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def main():
    """Run XGBoost with fixed lag handling."""
    mlflow.set_experiment("store-sales-xgboost-v2")

    with mlflow.start_run(run_name="xgb_v2_fixed_lags"):
        # Load data
        train_df, test_df, holidays = load_data()

        # Create features
        print("Creating training features...")
        train_df = create_base_features(train_df)
        train_df = create_lag_features_train(train_df)
        train_df = add_holidays(train_df, holidays)
        train_df = train_df.dropna()

        test_df = create_base_features(test_df)
        test_df = add_holidays(test_df, holidays)

        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")

        # Time-based split
        split_date = train_df["date"].max() - pd.Timedelta(days=15)
        train_mask = train_df["date"] <= split_date
        val_mask = train_df["date"] > split_date

        feature_cols = [
            "Time",
            "day_of_week", "day_of_month", "month", "week_of_year",
            "is_month_end", "is_month_start", "is_weekend",
            "Lag_1", "Lag_2", "Lag_3", "Lag_7", "Lag_14", "Lag_21", "Lag_28",
            "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "Roll_mean_60", "Roll_mean_90",
            "Roll_std_7", "Roll_std_14", "Roll_std_30", "Roll_std_60", "Roll_std_90",
            "onpromotion", "onpromotion_lag1", "onpromotion_lag7",
            "is_holiday",
        ]

        X_train = train_df.loc[train_mask, feature_cols]
        y_train = train_df.loc[train_mask, "sales"]
        X_val = train_df.loc[val_mask, feature_cols]
        y_val = train_df.loc[val_mask, "sales"]

        print(f"\nTrain samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")

        # Train XGBoost
        print("\nTraining XGBoost...")
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
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=100,
        )

        # Validate
        y_pred_val = model.predict(dval, iteration_range=(0, model.best_iteration))
        y_pred_val = np.clip(y_pred_val, 0, None)
        rmsle = np.sqrt(mean_squared_log_error(y_val + 1, y_pred_val + 1))

        print(f"\nValidation RMSLE: {rmsle:.6f}")

        # Log to MLflow
        mlflow.log_param("model", "XGBoost_v2_fixed_lags")
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_metric("cv_rmsle", rmsle)
        mlflow.log_metric("best_iteration", model.best_iteration)

        # Generate submission with proper lag handling
        submission = predict_test_with_lags(model, train_df, test_df, feature_cols)

        # Save
        submission_path = SUBMISSION_DIR / f"xgb_v2_fixed_lags_{rmsle:.4f}.csv"
        submission.to_csv(submission_path, index=False)
        print(f"\nSubmission saved: {submission_path}")

        mlflow.log_artifact(str(submission_path))

        print(f"\n✅ XGBoost v2 complete - CV RMSLE: {rmsle:.6f}")
        print("This version properly handles test lag features - should match CV better!")

        return rmsle


if __name__ == "__main__":
    main()
