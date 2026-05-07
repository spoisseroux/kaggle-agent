"""Selective feature engineering — V3.

Keep baseline features + only the best new features:
- FarePerPerson
- NameLength
- Age_Class interaction
- Better title grouping
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

def engineer_features_v3(df, is_train=True):
    """Selective feature engineering - best of baseline + new."""
    df = df.copy()

    # ===== Missing values =====
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # ===== Baseline features =====
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Better title grouping
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Capt': 'Rare', 'Don': 'Rare', 'Sir': 'Rare', 'Lady': 'Rare',
        'Countess': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare',
        'Mme': 'Mrs', 'Mlle': 'Miss', 'Ms': 'Miss'
    }
    df['Title'] = df['Title'].map(title_mapping).fillna('Rare')

    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100],
                          labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Med', 'High', 'VeryHigh'],
                            duplicates='drop')
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # ===== NEW: Best performing features from V2 =====
    # Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # Name length (proxy for social status)
    df['NameLength'] = df['Name'].str.len()

    # Age * Pclass interaction
    df['Age_Class'] = df['Age'] * df['Pclass']

    # ===== Select features =====
    feature_cols = [
        # Baseline
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
        'FamilySize', 'IsAlone', 'Title', 'AgeBin', 'FareBin', 'HasCabin',
        # New
        'FarePerPerson', 'NameLength', 'Age_Class'
    ]

    df_features = df[feature_cols].copy()

    # Encode categoricals
    for col in ['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin']:
        if col in df_features.columns:
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col].astype(str))

    return df_features

def main():
    print("=" * 80)
    print("TITANIC FEATURES V3 — SELECTIVE ENGINEERING")
    print("=" * 80)

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    print(f"\n✓ Loaded {len(train)} train samples, {len(test)} test samples")

    # Engineer features
    print("\n🔧 Engineering selective features...")
    X = engineer_features_v3(train, is_train=True)
    y = train['Survived']
    X_test = engineer_features_v3(test, is_train=False)

    print(f"  Features: {list(X.columns)}")
    print(f"  Shape: {X.shape} (baseline: 13, v2: 23)")

    # Train model - try both RF and LightGBM
    print("\n🌲 Training RandomForest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    print("\n📊 Running 5-fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy')

    print(f"\n  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Check improvement
    baseline_cv = 0.8339
    improvement = cv_scores.mean() - baseline_cv
    print(f"\n  Improvement over baseline: {improvement:+.4f} ({improvement*100:+.2f}%)")

    # Train on full dataset
    print("\n🎯 Training on full dataset...")
    rf_model.fit(X, y)

    # Feature importance
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n📈 Feature Importances:")
    for idx, row in importances.iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.4f}")

    # Generate predictions
    print("\n🔮 Generating predictions...")
    predictions = rf_model.predict(X_test)

    # Save submission
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })

    submission_path = data_dir.parent / "submissions" / "features_v3_submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\n✅ Submission saved to {submission_path.relative_to(ROOT)}")
    print(f"   Predicted survival rate: {predictions.mean():.2%}")

    # Log to MLflow
    print("\n📝 Logging to MLflow...")
    config = {
        'model': 'RandomForest',
        'n_estimators': 200,
        'max_depth': 7,
        'features': len(X.columns),
        'feature_set': 'v3_selective',
        'train_acc': rf_model.score(X, y),
        'improvement_over_baseline': improvement
    }

    with mlflow_run(
        competition_slug='titanic',
        model_type='RandomForest',
        feature_set='v3_selective',
        config=config,
        notes='Selective features: baseline + FarePerPerson + NameLength + Age_Class'
    ) as (run, captured):
        captured['cv_mean'] = cv_scores.mean()
        captured['cv_std'] = cv_scores.std()
        print(f"   Logged as run: {run.info.run_id}")

    print("\n" + "=" * 80)
    print(f"FEATURES V3 COMPLETE — CV: {cv_scores.mean():.4f}")
    print("=" * 80)

    return cv_scores.mean()

if __name__ == '__main__':
    score = main()
