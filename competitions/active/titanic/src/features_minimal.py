"""Minimal feature engineering - only core features that generalize.

Features used:
- Sex (one-hot encoded)
- Pclass (as numeric)
- Age (imputed)
- Fare (imputed)
- FamilySize (SibSp + Parch + 1)
- IsAlone (FamilySize == 1)

Excluded:
- Title extraction (may not generalize to test)
- Cabin deck (too many missing values)
- Ticket prefix (may be test-specific)
- Interaction features (Age*Class, etc.)
"""
import pandas as pd
import numpy as np

def engineer_features_minimal(df, is_train=True):
    """Minimal feature engineering for better generalization."""
    data = df.copy()

    # 1. Sex - simple one-hot encoding
    data['Sex_male'] = (data['Sex'] == 'male').astype(int)

    # 2. Pclass - keep as is (ordinal: 1, 2, 3)

    # 3. Age - impute with median
    if is_train:
        global AGE_MEDIAN
        AGE_MEDIAN = data['Age'].median()
    data['Age'] = data['Age'].fillna(AGE_MEDIAN)

    # 4. Fare - impute with median
    if is_train:
        global FARE_MEDIAN
        FARE_MEDIAN = data['Fare'].median()
    data['Fare'] = data['Fare'].fillna(FARE_MEDIAN)

    # 5. FamilySize - simple count
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # 6. IsAlone - binary indicator
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    # Select only these features
    features = ['Pclass', 'Sex_male', 'Age', 'Fare', 'FamilySize', 'IsAlone']

    return data[features]
