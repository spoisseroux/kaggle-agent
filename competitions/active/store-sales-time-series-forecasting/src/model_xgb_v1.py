"""XGBoost model v1: Alternative to LightGBM

Same features as LightGBM v1, different algorithm.
Sometimes XGBoost performs better on time series.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
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
    return train, test, holidays


def create_features(df, is_train=True, train_df=None):
    """Create comprehensive time series features."""
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    df["Time"] = df.groupby(["store_nbr", "family"], observed=True).cumcount()

    # Seasonality
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    if is_train:
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

        for window in [7, 14, 30, 60, 90]:
            df[f"Roll_mean_{window}"] = (
                df.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )
            df[f"Roll_std_{window}"] = (
                df.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.rolling(window=window, min_periods=1).std())
            )

        df["onpromotion_lag1"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(1).fillna(0)
        df["onpromotion_lag7"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(7).fillna(0)
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

            for lag in [1, 2, 3, 7, 14, 21, 28]:
                df[f"Lag_{lag}"] = df["lag_mean"].fillna(0)

            for window in [7, 14, 30, 60, 90]:
                df[f"Roll_mean_{window}"] = df["lag_mean"].fillna(0)
                df[f"Roll_std_{window}"] = df["lag_std"].fillna(0)

            df["onpromotion_lag1"] = df["onpromotion"]
            df["onpromotion_lag7"] = df["onpromotion"]
            df = df.drop(columns=["lag_mean", "lag_std"])

    return df


def add_holidays(df, holidays):
    """Add holiday indicator."""
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model."""
    print("\nTraining XGBoost...")
    print(f"Features ({len(X_train.columns)}): {list(X_train.columns)}")

    # XGBoost parameters
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

    evals = [(dtrain, "train"), (dval, "val")]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    # Validate
    y_pred_val = model.predict(dval, iteration_range=(0, model.best_iteration))
    y_pred_val = np.clip(y_pred_val, 0, None)

    # Calculate RMSLE
    rmsle = np.sqrt(mean_squared_log_error(y_val + 1, y_pred_val + 1))
    print(f"\nValidation RMSLE: {rmsle:.6f}")
    print(f"Best iteration: {model.best_iteration}")

    return model, rmsle


def main():
    """Run XGBoost experiment."""
    mlflow.set_experiment("store-sales-xgboost")

    with mlflow.start_run(run_name="xgb_v1_full_features"):
        # Load data
        train_df, test_df, holidays = load_data()

        # Create features
        train_df = create_features(train_df, is_train=True)
        test_df = create_features(test_df, is_train=False, train_df=train_df)
        train_df = add_holidays(train_df, holidays)
        test_df = add_holidays(test_df, holidays)
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
        print(f"Validation samples: {len(X_val)}")

        # Log parameters
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_param("num_features", len(feature_cols))

        # Train
        model, rmsle = train_model(X_train, y_train, X_val, y_val)

        # Log metrics
        mlflow.log_metric("cv_rmsle", rmsle)
        mlflow.log_metric("best_iteration", model.best_iteration)

        # Generate submission
        print("\nGenerating submission...")
        dtest = xgb.DMatrix(test_df[feature_cols])
        y_pred = model.predict(dtest, iteration_range=(0, model.best_iteration))
        y_pred = np.clip(y_pred, 0, None)

        submission = test_df[["id"]].copy()
        submission["sales"] = y_pred

        submission_path = SUBMISSION_DIR / "xgb_v1_submission.csv"
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to: {submission_path}")

        mlflow.log_artifact(str(submission_path))

        print(f"\n✅ XGBoost v1 complete - CV RMSLE: {rmsle:.6f}")

        return rmsle


if __name__ == "__main__":
    main()
