"""Debug why v7 performs worse than v5 and v6."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/store-sales-time-series-forecasting")

# Load data
train = pd.read_csv(
    DATA_DIR / "train.csv",
    dtype={"store_nbr": "category", "family": "category"},
    parse_dates=["date"],
)

print("="*80)
print("Debugging XGBoost v7 Poor Performance")
print("="*80)
print()

# Sort data
train = train.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

# Create sample features for one store/family to inspect
sample = train[(train["store_nbr"] == 1) & (train["family"] == "AUTOMOTIVE")].copy()
sample = sample.sort_values("date").reset_index(drop=True)

print(f"Sample data: Store 1, Family AUTOMOTIVE")
print(f"Date range: {sample['date'].min()} to {sample['date'].max()}")
print(f"Rows: {len(sample)}")
print()

# Create v7 features
sample["lag_1"] = sample["sales"].shift(1)
sample["lag_7"] = sample["sales"].shift(7)
sample["lag_14"] = sample["sales"].shift(14)
sample["lag_28"] = sample["sales"].shift(28)

sample["roll_mean_7"] = sample["sales"].rolling(window=7, min_periods=1).mean()
sample["roll_mean_14"] = sample["sales"].rolling(window=14, min_periods=1).mean()
sample["roll_mean_28"] = sample["sales"].rolling(window=28, min_periods=1).mean()

print("First 30 rows with features:")
print(sample[["date", "sales", "lag_1", "lag_7", "lag_28", "roll_mean_7"]].head(30))
print()

# Check for NaN values
print("NaN counts:")
print(sample[["lag_1", "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_14", "roll_mean_28"]].isnull().sum())
print()

# After dropna, how much data remains?
sample_clean = sample.dropna()
print(f"After dropna: {len(sample_clean)} rows (lost {len(sample) - len(sample_clean)} rows)")
print()

# Now check across ALL store/family groups
print("Checking lag creation across all groups...")
train["lag_28"] = train.groupby(["store_nbr", "family"], observed=True)["sales"].shift(28)

total_rows = len(train)
rows_with_nan = train["lag_28"].isnull().sum()
pct_nan = (rows_with_nan / total_rows) * 100

print(f"Total rows: {total_rows:,}")
print(f"Rows with NaN in lag_28: {rows_with_nan:,} ({pct_nan:.1f}%)")
print()

# After dropna on full dataset
train_clean = train.dropna(subset=["lag_28"])
print(f"After dropping NaN: {len(train_clean):,} rows remaining ({(len(train_clean)/total_rows)*100:.1f}%)")
print()

# TimeSeriesSplit analysis
from sklearn.model_selection import TimeSeriesSplit

# Simulate what happens in CV
X = train_clean.index.values.reshape(-1, 1)
tscv = TimeSeriesSplit(n_splits=5)

print("TimeSeriesSplit fold sizes:")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    fold_train = train_clean.iloc[train_idx]
    fold_val = train_clean.iloc[val_idx]

    # Check date ranges
    train_dates = pd.to_datetime(fold_train["date"]) if "date" in fold_train.columns else None
    val_dates = pd.to_datetime(fold_val["date"]) if "date" in fold_val.columns else None

    print(f"\nFold {fold}:")
    print(f"  Train: {len(train_idx):,} rows")
    print(f"  Val: {len(val_idx):,} rows")
    if train_dates is not None:
        print(f"  Train dates: {train_dates.min()} to {train_dates.max()}")
        print(f"  Val dates: {val_dates.min()} to {val_dates.max()}")

    # Check for data quality issues in validation set
    val_data = fold_val[["sales", "lag_28"]].copy()
    if "lag_28" in val_data.columns:
        mean_sales = val_data["sales"].mean()
        mean_lag28 = val_data["lag_28"].mean()
        print(f"  Val mean sales: {mean_sales:.2f}")
        print(f"  Val mean lag_28: {mean_lag28:.2f}")

        # Check if lag_28 is zero for many rows (would indicate issue)
        zero_lags = (val_data["lag_28"] == 0).sum()
        print(f"  Val rows with lag_28=0: {zero_lags:,} ({(zero_lags/len(val_data))*100:.1f}%)")

print("\n" + "="*80)
print("FINDINGS:")
print("="*80)
