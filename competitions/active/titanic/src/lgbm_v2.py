"""LightGBM model with features V2.

Gradient boosting may handle the advanced features better than RandomForest.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Setup paths
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.experiment_tracker import run as mlflow_run
from features_v2 import engineer_features_v2

def main():
    print("=" * 80)
    print("TITANIC LIGHTGBM WITH FEATURES V2")
    print("=" * 80)

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    print(f"\n✓ Loaded {len(train)} train samples, {len(test)} test samples")

    # Engineer features
    print("\n🔧 Engineering features (V2)...")
    X = engineer_features_v2(train, is_train=True)
    y = train['Survived']
    X_test = engineer_features_v2(test, is_train=False)

    print(f"  Features: {len(X.columns)}")
    print(f"  Shape: {X.shape}")

    # Train LightGBM
    print("\n💡 Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )

    # Cross-validation
    print("\n📊 Running 5-fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    print(f"\n  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Check improvement
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
        print(f"  {row['feature']:20s}: {row['importance']:.0f}")

    # Generate predictions
    print("\n🔮 Generating predictions...")
    predictions = model.predict(X_test)

    # Save submission
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })

    submission_path = data_dir.parent / "submissions" / "lgbm_v2_submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\n✅ Submission saved to {submission_path.relative_to(ROOT)}")
    print(f"   Predicted survival rate: {predictions.mean():.2%}")

    # Log to MLflow
    print("\n📝 Logging to MLflow...")
    config = {
        'model': 'LightGBM',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'features': len(X.columns),
        'train_acc': model.score(X, y),
        'improvement_over_baseline': improvement
    }

    with mlflow_run(
        competition_slug='titanic',
        model_type='LightGBM',
        feature_set='v2_advanced',
        config=config,
        notes='LightGBM with advanced features from v2'
    ) as (run, captured):
        captured['cv_mean'] = cv_scores.mean()
        captured['cv_std'] = cv_scores.std()
        print(f"   Logged as run: {run.info.run_id}")

    print("\n" + "=" * 80)
    print(f"LIGHTGBM V2 COMPLETE — CV: {cv_scores.mean():.4f} (improvement: {improvement:+.4f})")
    print("=" * 80)

    return cv_scores.mean()

if __name__ == '__main__':
    score = main()
