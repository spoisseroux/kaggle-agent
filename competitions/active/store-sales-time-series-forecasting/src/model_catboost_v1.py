"""CatBoost v1: Alternative gradient boosting with proper lag handling

CatBoost often outperforms XGBoost/LightGBM on time series.
Key advantages:
- Better handling of categorical features
- Less prone to overfitting
- Symmetric tree structure
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor, Pool
import mlflow
from tqdm import tqdm

# Paths
DATA_DIR = Path("/home/keehar/kaggle-agent/data/store-sales-time-series-forecasting")
SUBMISSION_DIR = Path("/home/keehar/kaggle-agent/competitions/active/store-sales-time-series-forecasting/submissions")


def load_data():
    """Load all datasets."""
    print("Loading data...")
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        parse_dates=["date"],
    )
    test = pd.read_csv(
        DATA_DIR / "test.csv",
        parse_dates=["date"],
    )
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    return train, test, holidays, stores


def create_base_features(df, stores):
    """Create non-lag features."""
    # Merge store metadata
    df = df.merge(stores, on="store_nbr", how="left")

    # Convert categorical columns after merge
    df["store_nbr"] = df["store_nbr"].astype("category")
    df["family"] = df["family"].astype("category")
    df["city"] = df["city"].astype("category")
    df["state"] = df["state"].astype("category")
    df["type"] = df["type"].astype("category")
    df["cluster"] = df["cluster"].astype("category")

    # Time features
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
    for window in [7, 14, 30]:
        df[f"Roll_mean_{window}"] = (
            df.groupby(["store_nbr", "family"], observed=True)["sales"]
            .transform(lambda x: x.rolling(window=window, min_periods=max(1, window//2)).mean())
        )

    # Promotion lags
    df["onpromotion_lag1"] = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(1).fillna(0)

    return df


def add_holidays(df, holidays):
    """Add holiday indicator."""
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def predict_test_with_lags(model, train_df, test_df, feature_cols, cat_features):
    """Predict test set day-by-day with proper lag propagation."""
    print("\nPredicting test set with proper lag propagation...")

    train_df = train_df.copy()
    test_df = test_df.copy()
    test_df["sales"] = 0

    test_dates = sorted(test_df["date"].unique())
    predictions = []

    for test_date in tqdm(test_dates, desc="Predicting test days"):
        test_day_mask = test_df["date"] == test_date
        test_day = test_df[test_day_mask].copy()

        historical = pd.concat([train_df, test_df[test_df["date"] < test_date]], ignore_index=True)
        combined = pd.concat([historical, test_day], ignore_index=True)

        combined = combined.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
        combined["Time"] = combined.groupby(["store_nbr", "family"], observed=True).cumcount()

        for lag in [1, 2, 3, 7, 14, 21, 28]:
            combined[f"Lag_{lag}"] = combined.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

        for window in [7, 14, 30]:
            combined[f"Roll_mean_{window}"] = (
                combined.groupby(["store_nbr", "family"], observed=True)["sales"]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )

        combined["onpromotion_lag1"] = combined.groupby(["store_nbr", "family"], observed=True)["onpromotion"].shift(1).fillna(0)

        test_day_features = combined[combined["date"] == test_date].copy()
        X_test_day = test_day_features[feature_cols]

        preds = model.predict(X_test_day)
        preds = np.clip(preds, 0, None)

        test_df.loc[test_day_mask, "sales"] = preds

        for idx, pred in zip(test_day_features["id"].values, preds):
            predictions.append({"id": idx, "sales": pred})

    return pd.DataFrame(predictions)


def main():
    """Run CatBoost experiment."""
    mlflow.set_experiment("store-sales-catboost")

    with mlflow.start_run(run_name="catboost_v1_fixed_lags"):
        # Load data
        train_df, test_df, holidays, stores = load_data()

        # Create features
        print("Creating features...")
        train_df = create_base_features(train_df, stores)
        train_df = create_lag_features_train(train_df)
        train_df = add_holidays(train_df, holidays)
        train_df = train_df.dropna()

        test_df = create_base_features(test_df, stores)
        test_df = add_holidays(test_df, holidays)

        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")

        # Time-based split
        split_date = train_df["date"].max() - pd.Timedelta(days=15)
        train_mask = train_df["date"] <= split_date
        val_mask = train_df["date"] > split_date

        # Categorical features
        cat_features = ["store_nbr", "family", "city", "state", "type", "cluster"]

        feature_cols = [
            "store_nbr", "family", "city", "state", "type", "cluster",
            "Time",
            "day_of_week", "day_of_month", "month", "week_of_year",
            "is_month_end", "is_month_start", "is_weekend",
            "Lag_1", "Lag_2", "Lag_3", "Lag_7", "Lag_14", "Lag_21", "Lag_28",
            "Roll_mean_7", "Roll_mean_14", "Roll_mean_30",
            "onpromotion", "onpromotion_lag1",
            "is_holiday",
        ]

        X_train = train_df.loc[train_mask, feature_cols]
        y_train = train_df.loc[train_mask, "sales"]
        X_val = train_df.loc[val_mask, feature_cols]
        y_val = train_df.loc[val_mask, "sales"]

        print(f"\nTrain samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")

        # Train CatBoost
        print("\nTraining CatBoost...")
        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            early_stopping_rounds=50,
            verbose=100,
            cat_features=cat_features,
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=100,
        )

        # Validate
        y_pred_val = model.predict(X_val)
        y_pred_val = np.clip(y_pred_val, 0, None)
        rmsle = np.sqrt(mean_squared_log_error(y_val + 1, y_pred_val + 1))

        print(f"\nValidation RMSLE: {rmsle:.6f}")
        print(f"Best iteration: {model.best_iteration_}")

        # Log to MLflow
        mlflow.log_param("model", "CatBoost_v1")
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_metric("cv_rmsle", rmsle)
        mlflow.log_metric("best_iteration", model.best_iteration_)

        # Generate submission
        submission = predict_test_with_lags(model, train_df, test_df, feature_cols, cat_features)

        submission_path = SUBMISSION_DIR / f"catboost_v1_{rmsle:.4f}.csv"
        submission.to_csv(submission_path, index=False)
        print(f"\nSubmission saved: {submission_path}")

        mlflow.log_artifact(str(submission_path))

        print(f"\nCatBoost v1 complete - CV RMSLE: {rmsle:.6f}")

        return rmsle


if __name__ == "__main__":
    main()
