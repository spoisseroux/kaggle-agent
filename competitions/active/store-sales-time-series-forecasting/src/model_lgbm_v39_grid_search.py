#!/usr/bin/env python3
"""LightGBM v39 - Grid Search: L1 vs L2 Regularization

STRATEGY: Test less aggressive regularization
- v38 used both L1=0.1 and L2=0.1 → LB 0.533 (worse than v19)
- v39: Test L1 ONLY vs L2 ONLY with grid search
- Find optimal regularization strength

Grid:
- L1 only: reg_alpha in [0.01, 0.05, 0.1], reg_lambda=0
- L2 only: reg_lambda in [0.01, 0.05, 0.1], reg_alpha=0
- learning_rate: [0.03, 0.05]
- num_leaves: [31, 64]
- max_depth: [5, 6]
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import itertools

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

DATA_DIR = Path("data/store-sales-time-series-forecasting")

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
    return train, test

def create_features(df, is_train=True, train_df=None):
    """v19 features (proven best)"""
    df = df.copy()

    # Temporal features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Lags
    for lag in [3, 7]:
        df[f"Lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)

    # Rolling means
    for window in [7, 14, 30, 60, 90]:
        df[f"Roll_mean_{window}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    # Rolling std
    df["Roll_std_7"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )

    # For test set, fill NaN with store-family mean from training
    if not is_train and train_df is not None:
        lag_mean = train_df.groupby(["store_nbr", "family"], observed=True)["sales"].mean()
        df = df.merge(
            lag_mean.rename("lag_mean").reset_index(),
            on=["store_nbr", "family"],
            how="left"
        )
        lag_cols = [c for c in df.columns if c.startswith(("Lag_", "Roll_"))]
        for col in lag_cols:
            df[col] = df[col].fillna(df["lag_mean"])
        df = df.drop(columns=["lag_mean"])

    return df

def train_and_evaluate(X_train, y_train, X_val, y_val, params):
    """Train LightGBM and return CV score"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration)
    preds = np.maximum(preds, 0)  # Clip negatives

    rmsle = np.sqrt(mean_squared_log_error(y_val, preds))
    return rmsle, model

def main():
    print("="*70)
    print("LightGBM v39 - Grid Search: L1 vs L2 Regularization")
    print("="*70)

    # Load data
    print("Loading data...")
    train_df, test_df = load_data()

    # Create features
    print("Creating features...")
    train_df = create_features(train_df, is_train=True)
    train_df = train_df.dropna()

    # 30-day holdout
    cutoff_date = train_df['date'].max() - pd.Timedelta(days=30)
    val_mask = train_df['date'] >= cutoff_date
    train_mask = ~val_mask

    exclude_cols = ["id", "date", "sales", "store_nbr", "family"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X = train_df[feature_cols]
    y = train_df["sales"]
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"Features: {len(feature_cols)}")
    print()

    # Base parameters
    base_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "n_estimators": 1000,
        "random_state": 42,
        "verbosity": -1,
    }

    # Grid search
    results = []

    # L1 only (reg_lambda=0)
    print("Testing L1 regularization only...")
    for reg_alpha in [0.01, 0.05, 0.1]:
        for lr in [0.03, 0.05]:
            for num_leaves in [31, 64]:
                for max_depth in [5, 6]:
                    params = base_params.copy()
                    params.update({
                        "reg_alpha": reg_alpha,
                        "reg_lambda": 0.0,
                        "learning_rate": lr,
                        "num_leaves": num_leaves,
                        "max_depth": max_depth,
                    })

                    cv, _ = train_and_evaluate(X_train, y_train, X_val, y_val, params)

                    results.append({
                        "type": "L1",
                        "reg_alpha": reg_alpha,
                        "reg_lambda": 0.0,
                        "lr": lr,
                        "num_leaves": num_leaves,
                        "max_depth": max_depth,
                        "cv": cv
                    })

                    print(f"  L1={reg_alpha:.2f}, lr={lr:.2f}, leaves={num_leaves}, depth={max_depth} → CV {cv:.4f}")

    print()

    # L2 only (reg_alpha=0)
    print("Testing L2 regularization only...")
    for reg_lambda in [0.01, 0.05, 0.1]:
        for lr in [0.03, 0.05]:
            for num_leaves in [31, 64]:
                for max_depth in [5, 6]:
                    params = base_params.copy()
                    params.update({
                        "reg_alpha": 0.0,
                        "reg_lambda": reg_lambda,
                        "learning_rate": lr,
                        "num_leaves": num_leaves,
                        "max_depth": max_depth,
                    })

                    cv, _ = train_and_evaluate(X_train, y_train, X_val, y_val, params)

                    results.append({
                        "type": "L2",
                        "reg_alpha": 0.0,
                        "reg_lambda": reg_lambda,
                        "lr": lr,
                        "num_leaves": num_leaves,
                        "max_depth": max_depth,
                        "cv": cv
                    })

                    print(f"  L2={reg_lambda:.2f}, lr={lr:.2f}, leaves={num_leaves}, depth={max_depth} → CV {cv:.4f}")

    print()
    print("="*70)
    print("GRID SEARCH RESULTS")
    print("="*70)

    # Sort by CV score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("cv")

    print("\nTop 10 configurations:")
    print(results_df.head(10).to_string(index=False))

    print(f"\nBest CV: {results_df.iloc[0]['cv']:.4f}")
    print(f"v19 CV: 0.3572 (baseline)")
    print(f"Improvement: {(0.3572 - results_df.iloc[0]['cv']) / 0.3572 * 100:.2f}%")

    # Save results
    results_df.to_csv("v39_grid_search_results.csv", index=False)
    print("\nFull results saved to: v39_grid_search_results.csv")

    return 0

if __name__ == "__main__":
    sys.exit(main())
