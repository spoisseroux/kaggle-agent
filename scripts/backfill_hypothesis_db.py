#!/usr/bin/env python3
"""Backfill hypothesis database with existing experiments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.hypothesis_db import HypothesisDatabase


def main():
    db = HypothesisDatabase()

    # Competition baseline
    baseline_cv = 0.3572  # v19

    experiments = [
        {
            'experiment_id': 'v19',
            'hypothesis': 'LightGBM with v1 features is optimal',
            'rationale': 'Model architecture matters more than hyperparameters',
            'category': 'model_architecture',
            'cv_score': 0.3572,
            'lb_score': 0.498,
            'succeeded': True,
            'features': ['Lag_3', 'Lag_7', 'Roll_mean_7', 'Roll_mean_14', 'Roll_mean_30',
                        'Roll_mean_60', 'Roll_mean_90', 'Roll_std_7', 'day_of_week',
                        'is_weekend', 'is_holiday', 'onpromotion'],
            'params': {'learning_rate': 0.05, 'num_leaves': 64, 'max_depth': 6}
        },
        {
            'experiment_id': 'v38',
            'hypothesis': 'Reduce regularization to improve generalization',
            'rationale': 'High CV-LB gap may indicate overfitting that regularization can fix',
            'category': 'hyperparameter_tuning',
            'cv_score': 0.366,
            'lb_score': 0.533,
            'succeeded': False,
            'failure_reason': 'Regularization not the issue - model already well-tuned',
            'features': ['Lag_3', 'Lag_7', 'Roll_mean_7', 'Roll_mean_14', 'Roll_mean_30'],
            'params': {'reg_alpha': 0.1, 'reg_lambda': 0.1}
        },
        {
            'experiment_id': 'v39',
            'hypothesis': 'Grid search L1 vs L2 regularization separately',
            'rationale': 'Test whether L1-only or L2-only performs better than combined',
            'category': 'hyperparameter_tuning',
            'cv_score': 0.4526,
            'lb_score': None,
            'succeeded': False,
            'failure_reason': 'All 48 configurations worse than baseline - regularization hurts',
            'features': None,
            'params': {'grid_size': 48}
        },
        {
            'experiment_id': 'v40',
            'hypothesis': 'XGBoost with simple parameters reduces overfitting',
            'rationale': 'XGBoost has 9.3% CV-LB gap vs LightGBM 39.5%',
            'category': 'model_architecture',
            'cv_score': 0.5220,
            'lb_score': None,
            'succeeded': False,
            'failure_reason': 'XGBoost significantly worse than LightGBM for this problem',
            'features': None,
            'params': {'max_depth': 4, 'learning_rate': 0.05}
        },
        {
            'experiment_id': 'v41',
            'hypothesis': 'XGBoost with historical proven parameters',
            'rationale': 'Use parameters from v1/v3 that achieved good CV historically',
            'category': 'hyperparameter_tuning',
            'cv_score': 0.4554,
            'lb_score': None,
            'succeeded': False,
            'failure_reason': 'Better than v40 but still worse than LightGBM',
            'features': None,
            'params': {'max_depth': 8, 'reg_alpha': 0.1, 'reg_lambda': 1.0}
        },
        {
            'experiment_id': 'v42',
            'hypothesis': 'Longer validation window (60 days) reduces bias',
            'rationale': 'More representative validation set with varied sales patterns',
            'category': 'validation_strategy',
            'cv_score': 0.4595,
            'lb_score': None,
            'succeeded': False,
            'failure_reason': 'Validation bias persists - all 2017 summer periods have elevated sales',
            'features': None,
            'params': {'validation_days': 60}
        },
        {
            'experiment_id': 'v43',
            'hypothesis': 'Walk-forward CV reduces single-window bias',
            'rationale': 'Multiple temporal folds span different market conditions',
            'category': 'validation_strategy',
            'cv_score': 0.4490,
            'lb_score': None,
            'succeeded': False,
            'failure_reason': 'Better than single XGBoost holdout but worse than LightGBM',
            'features': None,
            'params': {'n_folds': 3, 'model': 'xgboost'}
        },
        {
            'experiment_id': 'v44',
            'hypothesis': 'Walk-forward CV with LightGBM (best model)',
            'rationale': 'Apply walk-forward to best model architecture',
            'category': 'validation_strategy',
            'cv_score': 0.4665,
            'lb_score': None,
            'succeeded': False,
            'failure_reason': 'Worse than v19 single holdout - potential feature leakage',
            'features': None,
            'params': {'n_folds': 3, 'model': 'lightgbm'}
        },
        {
            'experiment_id': 'v46',
            'hypothesis': 'Recent data only (last 1.5 years) improves generalization',
            'rationale': 'Top Kaggle solutions show recent data generalizes better',
            'category': 'data_preprocessing',
            'cv_score': 0.5445,
            'lb_score': None,
            'succeeded': False,
            'failure_reason': 'Less training data hurt performance despite better temporal alignment',
            'features': None,
            'params': {'data_years': 1.5}
        },
        {
            'experiment_id': 'v47',
            'hypothesis': 'Day-of-week specific features capture weekly patterns',
            'rationale': 'Store/family/dayofweek averages are highly predictive',
            'category': 'feature_engineering',
            'cv_score': 0.4324,
            'lb_score': None,
            'succeeded': False,
            'failure_reason': 'DOW features have signal but not enough to beat v1 simple features',
            'features': ['store_family_dow_mean', 'store_family_dow_std'],
            'params': None
        },
        {
            'experiment_id': 'v48',
            'hypothesis': 'Seed ensemble reduces prediction variance',
            'rationale': 'Train multiple models with different seeds, average predictions',
            'category': 'ensemble_methods',
            'cv_score': 0.4411,
            'lb_score': None,
            'succeeded': False,
            'failure_reason': 'Implementation differs from v19 - unexpected worse performance',
            'features': None,
            'params': {'ensemble_size': 5, 'seeds': [42, 123, 456, 789, 1337]}
        },
    ]

    print("Backfilling hypothesis database...")
    print()

    for exp in experiments:
        print(f"Adding {exp['experiment_id']}: {exp['hypothesis']}")

        hyp_id = db.add_hypothesis(
            competition="store-sales-time-series-forecasting",
            experiment_id=exp['experiment_id'],
            hypothesis=exp['hypothesis'],
            rationale=exp['rationale'],
            category=exp['category']
        )

        db.record_result(
            hypothesis_id=hyp_id,
            cv_score=exp['cv_score'],
            lb_score=exp.get('lb_score'),
            baseline_cv=baseline_cv,
            succeeded=exp['succeeded'],
            failure_reason=exp.get('failure_reason'),
            features=exp.get('features'),
            params=exp.get('params')
        )

    print()
    print("="*70)
    print("ANALYZING PATTERNS")
    print("="*70)

    patterns = db.analyze_patterns("store-sales-time-series-forecasting")

    for pattern in patterns:
        print(f"\n{pattern['category'].upper()}: {pattern['insight']}")
        print(f"Confidence: {pattern['confidence']:.2f}")
        print(f"Evidence: {pattern['evidence']}")

        # Store as insight
        db.store_insight(
            competition="store-sales-time-series-forecasting",
            insight=pattern['insight'],
            evidence=pattern['evidence'],
            confidence=pattern['confidence'],
            category=pattern['category']
        )

    print()
    print("="*70)
    print("EXPERIMENT SUGGESTIONS")
    print("="*70)

    suggestions = db.suggest_next_experiments("store-sales-time-series-forecasting", n=5)

    for i, sugg in enumerate(suggestions, 1):
        print(f"\n{i}. {sugg['suggestion']} ({sugg['priority']} priority)")
        print(f"   Category: {sugg['category']}")
        print(f"   Rationale: {sugg['rationale']}")

    print()
    print("Backfill complete!")


if __name__ == "__main__":
    main()
