"""Ensemble the top 3 models from Optuna tuning.

Weighted voting based on CV scores:
- XGBoost: 0.8485 (weight: 0.40)
- LightGBM: 0.8451 (weight: 0.35)
- RandomForest: 0.8440 (weight: 0.25)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
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
    print("ENSEMBLE: XGBoost + LightGBM + RandomForest")
    print("="*80)

    # Best parameters from Optuna
    xgb_params = {
        'n_estimators': 294,
        'learning_rate': 0.020796927448298978,
        'max_depth': 7,
        'min_child_weight': 5,
        'subsample': 0.9197121567728955,
        'colsample_bytree': 0.8979777408556597,
        'gamma': 3.081587442535289e-07,
        'reg_alpha': 0.015105023850298839,
        'reg_lambda': 1.1626943463555593e-08,
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    lgb_params = {
        'n_estimators': 397,
        'learning_rate': 0.02363046106767058,
        'num_leaves': 32,
        'max_depth': 6,
        'min_child_samples': 31,
        'subsample': 0.7789097092688662,
        'colsample_bytree': 0.7972878823733976,
        'reg_alpha': 9.862585374076386e-05,
        'reg_lambda': 0.0031317775291180446,
        'random_state': 42,
        'verbose': -1
    }

    rf_params = {
        'n_estimators': 361,
        'max_depth': 8,
        'min_samples_split': 9,
        'min_samples_leaf': 2,
        'max_features': 'log2',
        'random_state': 42,
        'n_jobs': -1
    }

    # Create models
    xgb_model = xgb.XGBClassifier(**xgb_params)
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    rf_model = RandomForestClassifier(**rf_params)

    # Weighted voting based on CV scores
    # XGBoost: 0.8485, LightGBM: 0.8451, RF: 0.8440
    # Normalize to weights: 0.40, 0.35, 0.25
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('rf', rf_model)
        ],
        voting='soft',
        weights=[0.40, 0.35, 0.25]
    )

    print("\n⏳ Evaluating ensemble with 5-fold CV...")
    cv_scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"   CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"   Individual folds: {cv_scores}")

    # Compare with best individual model (XGBoost: 0.8485)
    improvement = cv_mean - 0.8485
    print(f"\n   {'🚀 Improvement' if improvement > 0 else '⚠️  Degradation'}: {improvement:+.4f}")

    # Train on full dataset
    print("\n⏳ Training ensemble on full dataset...")
    ensemble.fit(X, y)

    # Make predictions
    print("⏳ Making predictions on test set...")
    predictions = ensemble.predict(X_test)

    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })

    output_dir = ROOT / "competitions" / "active" / "titanic" / "submissions"
    output_path = output_dir / "ensemble_submission.csv"
    submission.to_csv(output_path, index=False)

    survival_rate = predictions.mean() * 100
    print(f"\n✅ Submission saved: {output_path}")
    print(f"   Predicted survival rate: {survival_rate:.2f}%")

    # Log to MLflow
    print("\n📝 Logging to MLflow...")
    with mlflow_run("titanic", model_type="ensemble") as run:
        # Log parameters
        run.log_param("model_type", "ensemble")
        run.log_param("voting", "soft")
        run.log_param("weights", "0.40,0.35,0.25")
        run.log_param("xgb_weight", 0.40)
        run.log_param("lgb_weight", 0.35)
        run.log_param("rf_weight", 0.25)

        # Log metrics
        run.log_metric("cv_score", cv_mean)
        run.log_metric("cv_std", cv_std)
        run.log_metric("improvement_vs_xgb", improvement)
        run.log_metric("survival_rate", survival_rate)

        # Log artifact
        run.log_artifact(str(output_path))

    print("\n" + "="*80)
    print(f"ENSEMBLE COMPLETE — CV: {cv_mean:.4f} ({improvement:+.4f} vs XGBoost)")
    print("="*80)

    return cv_mean, improvement

if __name__ == "__main__":
    main()
