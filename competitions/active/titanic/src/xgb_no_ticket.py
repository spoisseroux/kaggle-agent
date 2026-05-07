"""XGBoost without Ticket features - based on ablation study.

Ablation study showed Ticket features hurt CV by -0.0011.
This model excludes Ticket but keeps all other engineered features.
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

from feature_ablation import engineer_features_ablation

def main():
    # Load data
    data_dir = ROOT / "competitions" / "active" / "titanic" / "data"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    X = engineer_features_ablation(train, is_train=True, exclude_group='ticket')
    y = train['Survived']
    X_test = engineer_features_ablation(test, is_train=False, exclude_group='ticket')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("="*80)
    print("XGBOOST WITHOUT TICKET FEATURES")
    print("="*80)
    print(f"\n📋 Features ({len(X.columns)}): {list(X.columns)}")

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

    # Train and evaluate
    print("\n⏳ Evaluating with 5-fold CV...")
    model = xgb.XGBClassifier(**params)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"\n✅ CV Score: {cv_mean:.4f} (±{cv_std:.4f})")

    # Train on full dataset
    print("\n⏳ Training on full dataset...")
    model.fit(X, y)

    # Make predictions
    print("⏳ Making predictions...")
    predictions = model.predict(X_test)

    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })

    output_dir = ROOT / "competitions" / "active" / "titanic" / "submissions"
    output_path = output_dir / "xgb_no_ticket_submission.csv"
    submission.to_csv(output_path, index=False)

    survival_rate = predictions.mean() * 100
    print(f"\n✅ Submission saved: {output_path}")
    print(f"   Predicted survival rate: {survival_rate:.2f}%")

    print("\n" + "="*80)
    print(f"NO TICKET COMPLETE — CV: {cv_mean:.4f}")
    print("="*80)

    return cv_mean

if __name__ == "__main__":
    main()
