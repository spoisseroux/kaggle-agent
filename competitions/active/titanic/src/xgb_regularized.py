"""Regularized XGBoost with reduced complexity to improve generalization.

Changes from optuna_best:
- Stronger regularization: reg_alpha=0.1, reg_lambda=1.0
- Reduced complexity: max_depth=4 (was 7), n_estimators=150 (was 294)
- Lower learning_rate: 0.01 (was 0.021)
- Goal: Better generalization, less overfitting
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

from core.experiment_tracker import run as mlflow_run
from features_v3 import engineer_features_v3

def main():
    # Load data
    data_dir = ROOT / "competitions" / "active" / "titanic" / "data"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    X = engineer_features_v3(train, is_train=True)
    y = train['Survived']
    X_test = engineer_features_v3(test, is_train=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("="*80)
    print("REGULARIZED XGBOOST - Reduced Complexity")
    print("="*80)

    # Regularized parameters
    params = {
        'n_estimators': 150,          # Reduced from 294
        'learning_rate': 0.01,        # Reduced from 0.021
        'max_depth': 4,               # Reduced from 7
        'min_child_weight': 10,       # Increased from 5 (more conservative)
        'subsample': 0.8,             # Slightly reduced from 0.92
        'colsample_bytree': 0.8,      # Slightly reduced from 0.90
        'gamma': 0.1,                 # Increased from ~0 (more pruning)
        'reg_alpha': 0.1,             # Increased from 0.015 (L1 regularization)
        'reg_lambda': 1.0,            # Increased from ~0 (L2 regularization)
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    print("\n📋 Parameters:")
    for k, v in params.items():
        if k != 'eval_metric':
            print(f"  {k:20s}: {v}")

    # Train and evaluate with CV
    print("\n⏳ Evaluating with 5-fold CV...")
    model = xgb.XGBClassifier(**params)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"\n✅ CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"   Folds: {cv_scores}")

    # Compare with optuna best (0.8485)
    improvement = cv_mean - 0.8485
    print(f"\n   {'⚠️  Lower CV' if improvement < 0 else '🚀 Higher CV'}: {improvement:+.4f}")
    print("   (Lower CV might generalize better!)")

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
    output_path = output_dir / "xgb_regularized_submission.csv"
    submission.to_csv(output_path, index=False)

    survival_rate = predictions.mean() * 100
    print(f"\n✅ Submission saved: {output_path}")
    print(f"   Predicted survival rate: {survival_rate:.2f}%")

    # Log to MLflow
    print("\n📝 Logging to MLflow...")
    with mlflow_run("titanic", model_type="xgb_regularized",
                    notes="Reduced complexity + stronger regularization") as run:
        # Log parameters
        run.log_params(params)

        # Log metrics
        run.log_metric("cv_score", cv_mean)
        run.log_metric("cv_std", cv_std)
        run.log_metric("improvement_vs_optuna", improvement)
        run.log_metric("survival_rate", survival_rate)

        # Log artifact
        run.log_artifact(str(output_path))

    print("\n" + "="*80)
    print(f"REGULARIZED XGBOOST COMPLETE — CV: {cv_mean:.4f}")
    print("="*80)

    return cv_mean

if __name__ == "__main__":
    main()
