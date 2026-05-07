"""XGBoost with minimal features - testing if simpler features generalize better.

Using only 6 core features:
- Pclass, Sex_male, Age, Fare, FamilySize, IsAlone

Using regularized parameters that worked well previously.
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

from features_minimal import engineer_features_minimal

def main():
    # Load data
    data_dir = ROOT / "competitions" / "active" / "titanic" / "data"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    X = engineer_features_minimal(train, is_train=True)
    y = train['Survived']
    X_test = engineer_features_minimal(test, is_train=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("="*80)
    print("MINIMAL FEATURES - XGBoost with Core Features Only")
    print("="*80)
    print(f"\n📋 Features ({len(X.columns)}):")
    print(f"   {list(X.columns)}")

    # Use regularized parameters
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

    # Train and evaluate with CV
    print("\n⏳ Evaluating with 5-fold CV...")
    model = xgb.XGBClassifier(**params)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"\n✅ CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"   Folds: {cv_scores}")

    # Train on full dataset
    print("\n⏳ Training on full dataset...")
    model.fit(X, y)

    # Feature importance
    print("\n📊 Feature Importance:")
    importance = sorted(zip(X.columns, model.feature_importances_),
                       key=lambda x: x[1], reverse=True)
    for feat, imp in importance:
        print(f"   {feat:15s}: {imp:.4f}")

    # Make predictions
    print("\n⏳ Making predictions...")
    predictions = model.predict(X_test)

    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })

    output_dir = ROOT / "competitions" / "active" / "titanic" / "submissions"
    output_path = output_dir / "xgb_minimal_features_submission.csv"
    submission.to_csv(output_path, index=False)

    survival_rate = predictions.mean() * 100
    print(f"\n✅ Submission saved: {output_path}")
    print(f"   Predicted survival rate: {survival_rate:.2f}%")

    print("\n" + "="*80)
    print(f"MINIMAL FEATURES COMPLETE — CV: {cv_mean:.4f}")
    print("="*80)

    return cv_mean

if __name__ == "__main__":
    main()
