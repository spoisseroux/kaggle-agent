"""Advanced feature engineering for Titanic — V2.

Improvements over baseline:
- Better title grouping
- Cabin deck extraction
- Ticket prefix/frequency features
- Fare per person
- Name length as proxy for social status
- Age * Pclass interaction
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

def engineer_features_v2(df, is_train=True):
    """Advanced feature engineering."""
    df = df.copy()

    # ===== Missing value handling =====
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # ===== Family features =====
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Family size categories
    df['FamilySize_Cat'] = pd.cut(df['FamilySize'],
                                   bins=[0, 1, 4, 20],
                                   labels=['Alone', 'Small', 'Large'])

    # ===== Title extraction and grouping =====
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # More sophisticated title grouping
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Professional',
        'Rev': 'Professional',
        'Col': 'Military',
        'Major': 'Military',
        'Capt': 'Military',
        'Don': 'Nobility',
        'Sir': 'Nobility',
        'Lady': 'Nobility',
        'Countess': 'Nobility',
        'Jonkheer': 'Nobility',
        'Dona': 'Nobility',
        'Mme': 'Mrs',
        'Mlle': 'Miss',
        'Ms': 'Miss'
    }
    df['Title'] = df['Title'].map(title_mapping).fillna('Rare')

    # ===== Cabin features =====
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # Extract cabin deck (first letter)
    df['Deck'] = df['Cabin'].str[0]
    df['Deck'] = df['Deck'].fillna('Unknown')

    # Number of cabins booked
    df['NumCabins'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else len(str(x).split()))

    # ===== Ticket features =====
    # Extract ticket prefix
    df['TicketPrefix'] = df['Ticket'].str.extract(r'([A-Za-z./]+)', expand=False)
    df['TicketPrefix'] = df['TicketPrefix'].fillna('None')

    # Ticket frequency (shared tickets often indicate groups)
    ticket_counts = df['Ticket'].value_counts()
    df['TicketFreq'] = df['Ticket'].map(ticket_counts)

    # Ticket number
    df['TicketNumber'] = df['Ticket'].str.extract(r'(\d+)', expand=False)
    df['TicketNumber'] = pd.to_numeric(df['TicketNumber'], errors='coerce').fillna(0)

    # ===== Fare features =====
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # Fare bins (quantile-based)
    df['FareBin'] = pd.qcut(df['Fare'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                            duplicates='drop')

    # ===== Age features =====
    # Age bins
    df['AgeBin'] = pd.cut(df['Age'],
                          bins=[0, 12, 18, 30, 50, 80],
                          labels=['Child', 'Teen', 'Young', 'Middle', 'Senior'])

    # Is child
    df['IsChild'] = (df['Age'] < 12).astype(int)

    # Age * Pclass interaction (young first class vs old third class)
    df['Age_Class'] = df['Age'] * df['Pclass']

    # ===== Name length (proxy for social status) =====
    df['NameLength'] = df['Name'].str.len()

    # ===== Sex * Pclass interaction =====
    # Create a combined feature
    df['Sex_Pclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)

    # ===== Select and encode features =====
    feature_cols = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
        'FamilySize', 'IsAlone', 'FamilySize_Cat',
        'Title', 'HasCabin', 'Deck', 'NumCabins',
        'TicketPrefix', 'TicketFreq',
        'FarePerPerson', 'FareBin',
        'AgeBin', 'IsChild', 'Age_Class',
        'NameLength', 'Sex_Pclass'
    ]

    df_features = df[feature_cols].copy()

    # Encode categorical variables
    categorical_cols = ['Sex', 'Embarked', 'FamilySize_Cat', 'Title',
                       'Deck', 'TicketPrefix', 'FareBin', 'AgeBin', 'Sex_Pclass']

    for col in categorical_cols:
        if col in df_features.columns:
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col].astype(str))

    return df_features

def main():
    print("=" * 80)
    print("TITANIC FEATURES V2 — ADVANCED FEATURE ENGINEERING")
    print("=" * 80)

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    print(f"\n✓ Loaded {len(train)} train samples, {len(test)} test samples")

    # Engineer features
    print("\n🔧 Engineering advanced features...")
    X = engineer_features_v2(train, is_train=True)
    y = train['Survived']
    X_test = engineer_features_v2(test, is_train=False)

    print(f"  Features: {len(X.columns)}")
    print(f"  Shape: {X.shape}")

    # Train model
    print("\n🌲 Training RandomForest with tuned params...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    print("\n📊 Running 5-fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    print(f"\n  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Check improvement over baseline
    baseline_cv = 0.8339
    improvement = cv_scores.mean() - baseline_cv
    print(f"\n  Improvement over baseline: {improvement:+.4f} ({improvement*100:+.2f}%)")

    # Train on full dataset
    print("\n🎯 Training on full dataset...")
    model.fit(X, y)

    # Feature importance
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n📈 Top 15 Feature Importances:")
    for idx, row in importances.head(15).iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.4f}")

    # Generate predictions
    print("\n🔮 Generating predictions...")
    predictions = model.predict(X_test)

    # Save submission
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })

    submission_path = data_dir.parent / "submissions" / "features_v2_submission.csv"
    submission_path.parent.mkdir(exist_ok=True)
    submission.to_csv(submission_path, index=False)

    print(f"\n✅ Submission saved to {submission_path.relative_to(ROOT)}")
    print(f"   Predicted survival rate: {predictions.mean():.2%}")

    # Log to MLflow
    print("\n📝 Logging to MLflow...")
    config = {
        'model': 'RandomForest',
        'n_estimators': 200,
        'max_depth': 8,
        'features': len(X.columns),
        'feature_set': 'v2_advanced',
        'train_acc': model.score(X, y),
        'improvement_over_baseline': improvement
    }

    with mlflow_run(
        competition_slug='titanic',
        model_type='RandomForest',
        feature_set='v2_advanced',
        config=config,
        notes='Advanced features: title groups, deck, ticket freq, fare per person, interactions'
    ) as (run, captured):
        captured['cv_mean'] = cv_scores.mean()
        captured['cv_std'] = cv_scores.std()
        print(f"   Logged as run: {run.info.run_id}")

    print("\n" + "=" * 80)
    print(f"FEATURES V2 COMPLETE — CV: {cv_scores.mean():.4f} (improvement: {improvement:+.4f})")
    print("=" * 80)

    return cv_scores.mean()

if __name__ == '__main__':
    score = main()
