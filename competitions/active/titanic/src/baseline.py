"""Baseline model for Titanic competition.

Simple feature engineering + RandomForest to establish baseline CV score.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Setup paths
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.experiment_tracker import run as mlflow_run

def engineer_features(df, is_train=True):
    """Basic feature engineering."""
    df = df.copy()

    # Fill missing Age with median
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Fill missing Embarked with mode
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Fill missing Fare with median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Create FamilySize feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Create IsAlone feature
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Extract Title from Name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    # Simplify titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                         'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                         'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Create Age bins
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100],
                          labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

    # Create Fare bins
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Med', 'High', 'VeryHigh'],
                            duplicates='drop')

    # Has Cabin or not
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # Select features for model
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                    'Embarked', 'FamilySize', 'IsAlone', 'Title',
                    'AgeBin', 'FareBin', 'HasCabin']

    df_features = df[feature_cols].copy()

    # Encode categorical variables
    for col in ['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin']:
        if col in df_features.columns:
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col].astype(str))

    return df_features

def main():
    print("=" * 80)
    print("TITANIC BASELINE MODEL")
    print("=" * 80)

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    print(f"\n✓ Loaded {len(train)} train samples, {len(test)} test samples")

    # Engineer features
    print("\n🔧 Engineering features...")
    X = engineer_features(train, is_train=True)
    y = train['Survived']
    X_test = engineer_features(test, is_train=False)

    print(f"  Features: {list(X.columns)}")
    print(f"  Shape: {X.shape}")

    # Train baseline model
    print("\n🌲 Training RandomForest baseline...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    print("\n📊 Running 5-fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    print(f"\n  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full training set
    print("\n🎯 Training on full dataset...")
    model.fit(X, y)

    # Feature importance
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n📈 Top 10 Feature Importances:")
    for idx, row in importances.head(10).iterrows():
        print(f"  {row['feature']:15s}: {row['importance']:.4f}")

    # Generate predictions
    print("\n🔮 Generating predictions...")
    predictions = model.predict(X_test)

    # Save submission
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })

    submission_path = data_dir.parent / "submissions" / "baseline_submission.csv"
    submission_path.parent.mkdir(exist_ok=True)
    submission.to_csv(submission_path, index=False)

    print(f"\n✅ Submission saved to {submission_path.relative_to(ROOT)}")
    print(f"   Predicted survival rate: {predictions.mean():.2%}")

    # Log to MLflow
    print("\n📝 Logging to MLflow...")
    config = {
        'model': 'RandomForest',
        'n_estimators': 100,
        'max_depth': 7,
        'features': len(X.columns),
        'train_acc': model.score(X, y)
    }

    with mlflow_run(
        competition_slug='titanic',
        model_type='RandomForest',
        feature_set='baseline',
        config=config,
        notes='Initial baseline with basic feature engineering'
    ) as (run, captured):
        captured['cv_mean'] = cv_scores.mean()
        captured['cv_std'] = cv_scores.std()
        print(f"   Logged as run: {run.info.run_id}")

    print("\n" + "=" * 80)
    print(f"BASELINE COMPLETE — CV: {cv_scores.mean():.4f}")
    print("=" * 80)

    return cv_scores.mean()

if __name__ == '__main__':
    score = main()
