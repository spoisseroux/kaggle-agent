"""XGBoost v3 Fixed: External data WITHOUT transactions (data doesn't cover test period)

Builds on v2 by adding:
- Store type, cluster from stores.csv
- Oil price and rolling means from oil.csv

NOTE: Transactions data ends 2017-08-15, but test starts 2017-08-16 (no overlap).
Removed transaction features to fix LB score.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import mlflow
from tqdm import tqdm

# Paths
DATA_DIR = Path("data/store-sales-time-series-forecasting")
SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")


def load_data():
    """Load all datasets including external data."""
    print("Loading data...")
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "int16", "family": "category"},
        parse_dates=["date"],
    )
    test = pd.read_csv(
        DATA_DIR / "test.csv",
        dtype={"store_nbr": "int16", "family": "category"},
        parse_dates=["date"],
    )
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])

    return train, test, stores, oil, holidays


def prepare_oil_data(oil):
    """Preprocess oil data once."""
    oil = oil.copy()
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    oil = oil.sort_values("date")
    oil["oil_roll_7"] = oil["dcoilwtico"].rolling(window=7, min_periods=1).mean()
    oil["oil_roll_14"] = oil["dcoilwtico"].rolling(window=14, min_periods=1).mean()
    oil["oil_roll_30"] = oil["dcoilwtico"].rolling(window=30, min_periods=1).mean()
    return oil


def add_external_data(df, stores, oil_prepared):
    """Merge external datasets (NO transactions - data doesn't cover test period)."""
    # Drop existing external columns if they exist (from previous merges)
    for col in ["type", "cluster", "dcoilwtico", "oil_roll_7", "oil_roll_14", "oil_roll_30"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Add store metadata
    df = df.merge(stores[["store_nbr", "type", "cluster"]], on="store_nbr", how="left")

    # Add oil features (oil_prepared already has all features)
    oil_mean = oil_prepared["dcoilwtico"].mean()
    df = df.merge(oil_prepared, on="date", how="left")
    for col in ["dcoilwtico", "oil_roll_7", "oil_roll_14", "oil_roll_30"]:
        if col in df.columns:
            df[col] = df[col].fillna(oil_mean)
        else:
            df[col] = oil_mean  # If merge failed, fill with mean

    return df


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

    # Sales lags
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

    # Sales rolling features
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


def add_holidays(df, holidays):
    """Add holiday indicator."""
    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def predict_test_with_lags(model, train_df, test_df, stores, oil_prepared, holidays, feature_cols, le_type):
    """
    Predict test set day-by-day, using previous predictions for lag features.
    """
    print("\nPredicting test set with proper lag propagation...")

    # Combine train and test
    train_df = train_df.copy()
    test_df = test_df.copy()
    test_df["sales"] = 0  # Placeholder

    test_dates = sorted(test_df["date"].unique())
    predictions = []

    for test_date in tqdm(test_dates, desc="Predicting test days"):
        test_day_mask = test_df["date"] == test_date
        test_day = test_df[test_day_mask].copy()

        # Combine historical + current test day
        historical = pd.concat([train_df, test_df[test_df["date"] < test_date]], ignore_index=True)
        combined = pd.concat([historical, test_day], ignore_index=True)

        # Add external data
        combined = add_external_data(combined, stores, oil_prepared)

        # Encode store type
        combined["type"] = le_type.transform(combined["type"])

        # Create features
        combined = combined.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
        combined["Time"] = combined.groupby(["store_nbr", "family"], observed=True).cumcount()

        # Sales lags
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            combined[f"Lag_{lag}"] = combined.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

        # Sales rolling
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

        # Add holidays
        combined = add_holidays(combined, holidays)

        # Get test day rows
        test_day_features = combined[combined["date"] == test_date].copy()

        # Fill NaNs
        for col in feature_cols:
            if test_day_features[col].isna().any():
                test_day_features[col] = test_day_features[col].fillna(test_day_features[col].mean())

        # Predict
        X_test_day = test_day_features[feature_cols]
        dtest = xgb.DMatrix(X_test_day)
        preds = model.predict(dtest, iteration_range=(0, model.best_iteration))
        preds = np.clip(preds, 0, None)

        # Store predictions
        test_df.loc[test_day_mask, "sales"] = preds

        for idx, pred in zip(test_day_features["id"].values, preds):
            predictions.append({"id": idx, "sales": pred})

    return pd.DataFrame(predictions)


def main():
    """Run XGBoost v3 with external data features."""
    mlflow.set_experiment("store-sales-xgboost-v3")

    with mlflow.start_run(run_name="xgb_v3_fixed_no_trans"):
        # Load data
        train_df, test_df, stores, oil, holidays = load_data()

        # Prepare external data once
        print("Preparing external data...")
        oil_prepared = prepare_oil_data(oil)

        # Encode store type
        le_type = LabelEncoder()
        le_type.fit(stores["type"].unique())

        # Add external data
        print("Merging external data...")
        train_df = add_external_data(train_df, stores, oil_prepared)
        train_df["type"] = le_type.transform(train_df["type"])

        # Create features
        print("Creating features...")
        train_df = create_base_features(train_df)
        train_df = create_lag_features_train(train_df)
        train_df = add_holidays(train_df, holidays)
        train_df = train_df.dropna()

        print(f"Train shape: {train_df.shape}")

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
            "type", "cluster",
            "dcoilwtico", "oil_roll_7", "oil_roll_14", "oil_roll_30",
        ]

        X_train = train_df.loc[train_mask, feature_cols]
        y_train = train_df.loc[train_mask, "sales"]
        X_val = train_df.loc[val_mask, feature_cols]
        y_val = train_df.loc[val_mask, "sales"]

        print(f"\nTrain samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")
        print(f"Features: {len(feature_cols)}")

        # Train XGBoost
        print("\nTraining XGBoost...")
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "device": "cuda",
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
        mlflow.log_param("model", "XGBoost_v3_fixed_no_trans")
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_metric("cv_rmsle", rmsle)
        mlflow.log_metric("best_iteration", model.best_iteration)

        # Generate submission
        submission = predict_test_with_lags(
            model, train_df, test_df, stores, oil_prepared, holidays,
            feature_cols, le_type
        )

        # Save
        submission_path = SUBMISSION_DIR / f"xgb_v3_fixed_no_trans_{rmsle:.4f}.csv"
        submission.to_csv(submission_path, index=False)
        print(f"\nSubmission saved: {submission_path}")

        mlflow.log_artifact(str(submission_path))

        print(f"\n✅ XGBoost v3 fixed complete - CV RMSLE: {rmsle:.6f}")
        print(f"Features: oil prices + store metadata (no transactions - data doesn't cover test)")

        return rmsle


if __name__ == "__main__":
    from core.vram_manager import request_training_vram, release_training_vram
    request_training_vram()
    try:
        main()
    finally:
        release_training_vram()
