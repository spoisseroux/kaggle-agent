"""Hyperparameter tuning with Optuna for Titanic.

Test multiple models: RandomForest, LightGBM, XGBoost
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Setup paths
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.experiment_tracker import run as mlflow_run
from features_v3 import engineer_features_v3

# Load data globally for Optuna trials
data_dir = ROOT / "competitions" / "active" / "titanic" / "data"
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")

X = engineer_features_v3(train, is_train=True)
y = train['Survived']
X_test = engineer_features_v3(test, is_train=False)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective_rf(trial):
    """RandomForest hyperparameter search."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

def objective_lgbm(trial):
    """LightGBM hyperparameter search."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': 42,
        'verbose': -1
    }

    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

def objective_xgb(trial):
    """XGBoost hyperparameter search."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': 42,
        'tree_method': 'hist',
        'verbosity': 0
    }

    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

def tune_model(model_name, objective, n_trials=50):
    """Run Optuna optimization for a model."""
    print(f"\n{'='*80}")
    print(f"TUNING {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Running {n_trials} trials...")

    study = optuna.create_study(direction='maximize', study_name=f'titanic_{model_name}')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n✅ Best CV Score: {study.best_value:.4f}")
    print(f"\n📋 Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key:20s}: {value}")

    return study

def main():
    print("=" * 80)
    print("TITANIC HYPERPARAMETER TUNING WITH OPTUNA")
    print("=" * 80)
    print(f"\n✓ Loaded {len(X)} train samples with {len(X.columns)} features")
    print(f"  Baseline CV: 0.8339")
    print(f"  Features V3 CV: 0.8350")

    results = {}

    # Tune RandomForest
    study_rf = tune_model('RandomForest', objective_rf, n_trials=50)
    results['RandomForest'] = {'score': study_rf.best_value, 'params': study_rf.best_params}

    # Tune LightGBM
    study_lgbm = tune_model('LightGBM', objective_lgbm, n_trials=50)
    results['LightGBM'] = {'score': study_lgbm.best_value, 'params': study_lgbm.best_params}

    # Tune XGBoost
    study_xgb = tune_model('XGBoost', objective_xgb, n_trials=50)
    results['XGBoost'] = {'score': study_xgb.best_value, 'params': study_xgb.best_params}

    # Summary
    print(f"\n{'='*80}")
    print("TUNING SUMMARY")
    print(f"{'='*80}")

    sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)

    for rank, (model_name, result) in enumerate(sorted_results, 1):
        improvement = result['score'] - 0.8339
        print(f"\n#{rank}. {model_name}: {result['score']:.4f} (improvement: {improvement:+.4f})")

    # Train best model and generate submission
    best_model_name = sorted_results[0][0]
    best_params = sorted_results[0][1]['params']
    best_score = sorted_results[0][1]['score']

    print(f"\n{'='*80}")
    print(f"TRAINING BEST MODEL: {best_model_name}")
    print(f"{'='*80}")

    if best_model_name == 'RandomForest':
        model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    elif best_model_name == 'LightGBM':
        model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
    else:  # XGBoost
        model = xgb.XGBClassifier(**best_params, random_state=42, verbosity=0)

    model.fit(X, y)

    # Generate predictions
    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })

    submission_path = data_dir.parent / "submissions" / "optuna_best_submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\n✅ Submission saved: {submission_path.relative_to(ROOT)}")
    print(f"   Predicted survival rate: {predictions.mean():.2%}")

    # Log to MLflow
    print("\n📝 Logging to MLflow...")
    config = best_params.copy()
    config['features'] = len(X.columns)
    config['train_acc'] = model.score(X, y)
    config['tuning_method'] = 'optuna'

    with mlflow_run(
        competition_slug='titanic',
        model_type=best_model_name,
        feature_set='v3_selective',
        config=config,
        notes=f'Optuna-tuned {best_model_name}, 50 trials per model'
    ) as (run, captured):
        captured['cv_mean'] = best_score
        print(f"   Logged as run: {run.info.run_id}")

    print(f"\n{'='*80}")
    print(f"TUNING COMPLETE — Best: {best_model_name} @ {best_score:.4f}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
