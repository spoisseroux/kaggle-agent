"""Exploratory Data Analysis for Titanic competition."""
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
data_dir = Path(__file__).parent.parent / "data"
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")

print("=" * 80)
print("TITANIC DATASET — EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\n📊 Dataset Shape")
print(f"Train: {train.shape[0]} rows × {train.shape[1]} columns")
print(f"Test:  {test.shape[0]} rows × {test.shape[1]} columns")

print("\n🎯 Target Distribution (Survived)")
print(train['Survived'].value_counts())
print(f"Survival rate: {train['Survived'].mean():.2%}")

print("\n📋 Column Types")
print(train.dtypes)

print("\n❓ Missing Values (Train)")
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    for col, count in missing.items():
        pct = 100 * count / len(train)
        print(f"  {col:15s}: {count:4d} ({pct:5.1f}%)")
else:
    print("  No missing values")

print("\n❓ Missing Values (Test)")
missing_test = test.isnull().sum()
missing_test = missing_test[missing_test > 0].sort_values(ascending=False)
if len(missing_test) > 0:
    for col, count in missing_test.items():
        pct = 100 * count / len(test)
        print(f"  {col:15s}: {count:4d} ({pct:5.1f}%)")
else:
    print("  No missing values")

print("\n🔢 Numeric Features Summary")
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['PassengerId', 'Survived']]
print(train[numeric_cols].describe())

print("\n📝 Categorical Features")
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    print(f"\n{col}:")
    print(f"  Unique values: {train[col].nunique()}")
    if train[col].nunique() <= 10:
        print(f"  Value counts:\n{train[col].value_counts()}")

print("\n🎲 Survival Rates by Key Features")
print(f"\nBy Sex:")
print(train.groupby('Sex')['Survived'].agg(['count', 'mean']))

print(f"\nBy Pclass:")
print(train.groupby('Pclass')['Survived'].agg(['count', 'mean']))

print(f"\nBy Embarked:")
print(train.groupby('Embarked')['Survived'].agg(['count', 'mean']))

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)

# Save summary
summary = {
    "train_rows": len(train),
    "test_rows": len(test),
    "target_balance": train['Survived'].mean(),
    "numeric_features": numeric_cols,
    "categorical_features": categorical_cols,
    "missing_train": missing.to_dict() if len(missing) > 0 else {},
    "missing_test": missing_test.to_dict() if len(missing_test) > 0 else {},
}

import json
(data_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
print("\n✅ Summary saved to data/eda_summary.json")
