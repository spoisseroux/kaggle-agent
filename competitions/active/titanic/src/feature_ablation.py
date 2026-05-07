"""Feature ablation study - systematically remove feature groups.

Goal: Identify which features hurt generalization.

Feature groups to test:
1. All features (baseline from features_v3)
2. Without Title features
3. Without Cabin features
4. Without interaction features (Age*Class, Fare_per_person)
5. Without Ticket features
6. Core features only (already tested)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Setup paths
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

def engineer_features_ablation(df, is_train=True, exclude_group=None):
    """Feature engineering with ablation - exclude specific feature group."""
    data = df.copy()

    # Basic preprocessing
    data['Age'] = data['Age'].fillna(data.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Embarked'] = data['Embarked'].fillna('S')

    # Feature group 1: Title features
    if exclude_group != 'title':
        data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
        data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                               'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                               'Jonkheer', 'Dona'], 'Rare')
        data['Title'] = data['Title'].replace('Mlle', 'Miss')
        data['Title'] = data['Title'].replace('Ms', 'Miss')
        data['Title'] = data['Title'].replace('Mme', 'Mrs')
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        data['Title'] = data['Title'].map(title_mapping).fillna(0)

    # Feature group 2: Family features (always include)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    # Feature group 3: Cabin features
    if exclude_group != 'cabin':
        data['Cabin_deck'] = data['Cabin'].str[0]
        data['Cabin_deck'] = data['Cabin_deck'].fillna('Unknown')
        deck_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'Unknown': 0}
        data['Cabin_deck'] = data['Cabin_deck'].map(deck_mapping)

    # Feature group 4: Interaction features
    if exclude_group != 'interactions':
        data['Age_Class'] = data['Age'] * data['Pclass']
        data['Fare_per_person'] = data['Fare'] / data['FamilySize']

    # Feature group 5: Ticket features
    if exclude_group != 'ticket':
        data['Ticket_prefix'] = data['Ticket'].str.split().str[0]
        data['Ticket_prefix'] = data['Ticket_prefix'].str.replace('.', '').str.replace('/', '')
        ticket_mapping = {}
        for i, prefix in enumerate(data['Ticket_prefix'].unique()):
            ticket_mapping[prefix] = i
        data['Ticket_prefix'] = data['Ticket_prefix'].map(ticket_mapping)

    # One-hot encode categorical
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

    # Select features
    feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
                   'FamilySize', 'IsAlone',
                   'Sex_male', 'Embarked_Q', 'Embarked_S']

    if exclude_group != 'title' and 'Title' in data.columns:
        feature_cols.append('Title')
    if exclude_group != 'cabin' and 'Cabin_deck' in data.columns:
        feature_cols.append('Cabin_deck')
    if exclude_group != 'interactions':
        if 'Age_Class' in data.columns:
            feature_cols.append('Age_Class')
        if 'Fare_per_person' in data.columns:
            feature_cols.append('Fare_per_person')
    if exclude_group != 'ticket' and 'Ticket_prefix' in data.columns:
        feature_cols.append('Ticket_prefix')

    available_cols = [c for c in feature_cols if c in data.columns]
    return data[available_cols]

def evaluate_model(X, y, cv, name):
    """Train and evaluate model."""
    params = {
        'n_estimators': 150,
        'learning_rate': 0.01,
        'max_depth': 4,
        'min_child_weight': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    model = xgb.XGBClassifier(**params)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    return cv_mean, cv_std, len(X.columns)

def main():
    # Load data
    data_dir = ROOT / "competitions" / "active" / "titanic" / "data"
    train = pd.read_csv(data_dir / "train.csv")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y = train['Survived']

    print("="*80)
    print("FEATURE ABLATION STUDY")
    print("="*80)

    results = []

    # Test different exclusions
    exclusions = [None, 'title', 'cabin', 'interactions', 'ticket']
    names = ['All features', 'Without Title', 'Without Cabin',
             'Without Interactions', 'Without Ticket']

    for exclude, name in zip(exclusions, names):
        print(f"\n⏳ Testing: {name}...")
        X = engineer_features_ablation(train, is_train=True, exclude_group=exclude)
        cv_mean, cv_std, n_features = evaluate_model(X, y, cv, name)
        results.append({
            'name': name,
            'exclude': exclude if exclude else 'none',
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'n_features': n_features
        })
        print(f"   Features: {n_features}, CV: {cv_mean:.4f} (±{cv_std:.4f})")

    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"\n{'Configuration':<25s} {'Features':<10s} {'CV Score':<15s}")
    print("-" * 60)

    for r in results:
        print(f"{r['name']:<25s} {r['n_features']:<10d} {r['cv_mean']:.4f} (±{r['cv_std']:.4f})")

    # Find best
    best = max(results, key=lambda x: x['cv_mean'])
    print(f"\n🏆 Best CV: {best['name']} with {best['cv_mean']:.4f}")

    # Insights
    print("\n💡 Insights:")
    baseline_cv = results[0]['cv_mean']
    for r in results[1:]:
        diff = r['cv_mean'] - baseline_cv
        direction = "improved" if diff > 0 else "degraded"
        print(f"   {r['name']:25s}: CV {direction:8s} by {diff:+.4f}")

    # Save results
    import json
    output_path = ROOT / "competitions" / "active" / "titanic" / "ablation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved: {output_path}")

    return results

if __name__ == "__main__":
    main()
