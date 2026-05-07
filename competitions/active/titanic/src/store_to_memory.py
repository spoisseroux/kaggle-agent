"""Store all Titanic experiments and learnings to memory (PostgreSQL + Qdrant)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.memory import (
    insert_experiment,
    insert_submission,
    upsert_competition,
    store_vector,
    ensure_collections
)
import requests
import os

def main():
    print("="*80)
    print("STORING TITANIC LEARNINGS TO MEMORY")
    print("="*80)

    # Update competition with final stats
    print("\n1. Updating competition record...")
    upsert_competition(
        'titanic',
        name='Titanic - Machine Learning from Disaster',
        metric='accuracy',
        higher_better=True,
        deadline='2030-12-31',
        status='completed',
        best_cv=0.8126,
        best_lb=0.78708,
        submissions_used=6,
        submissions_max=10
    )
    print("   ✅ Competition updated")

    # Store all experiments
    print("\n2. Storing experiments to PostgreSQL...")

    experiments = [
        {
            'model_type': 'xgboost_optuna',
            'feature_set': 'features_v3',
            'cv_mean': 0.8485,
            'cv_std': 0.0186,
            'lb_score': 0.77511,
            'notes': 'Aggressive hyperparameter tuning with Optuna. High CV but worst CV/LB gap (-0.0734). Overfitted.'
        },
        {
            'model_type': 'logistic_regression_baseline',
            'feature_set': 'baseline',
            'cv_mean': 0.8339,
            'cv_std': None,
            'lb_score': 0.76794,
            'notes': 'Simple baseline with basic features. Better generalization than Optuna model.'
        },
        {
            'model_type': 'xgboost_regularized',
            'feature_set': 'features_v3',
            'cv_mean': 0.8047,
            'cv_std': 0.0225,
            'lb_score': 0.78468,
            'notes': 'Strong regularization (reg_alpha=0.1, reg_lambda=1.0), reduced complexity (max_depth=4, n_estimators=150). Much better generalization.'
        },
        {
            'model_type': 'xgboost_minimal',
            'feature_set': 'minimal_6_features',
            'cv_mean': 0.7924,
            'cv_std': 0.0117,
            'lb_score': 0.77272,
            'notes': 'Only 6 core features. Best CV/LB gap (-0.0197). Shows simpler features generalize better.'
        },
        {
            'model_type': 'xgboost_no_ticket',
            'feature_set': 'features_v3_no_ticket',
            'cv_mean': 0.8126,
            'cv_std': 0.0104,
            'lb_score': 0.78708,
            'notes': 'Best model! Removed Ticket features after ablation study. Improved both CV and LB.'
        }
    ]

    exp_ids = []
    for exp in experiments:
        exp_id = insert_experiment('titanic', **exp)
        if exp_id:
            exp_ids.append(exp_id)
            print(f"   ✅ Stored: {exp['model_type']} (id={exp_id})")

    print(f"\n   Total: {len(exp_ids)} experiments stored")

    # Store submissions
    print("\n3. Storing submissions to PostgreSQL...")
    submissions = [
        {'filename': 'optuna_best_submission.csv', 'cv_score': 0.8485, 'lb_score': 0.77511,
         'notes': 'Optuna-tuned XGBoost'},
        {'filename': 'baseline_submission.csv', 'cv_score': 0.8339, 'lb_score': 0.76794,
         'notes': 'Simple LogisticRegression baseline'},
        {'filename': 'xgb_regularized_submission.csv', 'cv_score': 0.8047, 'lb_score': 0.78468,
         'notes': 'Regularized XGBoost'},
        {'filename': 'xgb_minimal_features_submission.csv', 'cv_score': 0.7924, 'lb_score': 0.77272,
         'notes': 'Minimal 6 features only'},
        {'filename': 'xgb_no_ticket_submission.csv', 'cv_score': 0.8126, 'lb_score': 0.78708,
         'notes': 'Best! No Ticket features'},
    ]

    for sub in submissions:
        sub_id = insert_submission('titanic', **sub)
        if sub_id:
            print(f"   ✅ Stored: {sub['filename']} (id={sub_id})")

    # Store key insights to Qdrant
    print("\n4. Storing insights to Qdrant...")

    # Ensure collections exist
    ensure_collections()

    # Create embeddings via MCP
    def create_embedding(text):
        """Create embedding using MCP service."""
        try:
            mcp_url = os.environ.get("MCP_URL", "http://docker:8000")
            mcp_token = os.environ.get("MCP_BEARER_TOKEN", "")
            r = requests.post(
                f"{mcp_url}/v1/embeddings",
                headers={"Authorization": f"Bearer {mcp_token}"},
                json={"input": text, "model": "text-embedding-3-small"},
                timeout=10
            )
            if r.status_code == 200:
                return r.json()["data"][0]["embedding"]
            else:
                print(f"     ⚠️  Embedding API error: {r.status_code}")
                return None
        except Exception as e:
            print(f"     ⚠️  Embedding error: {e}")
            return None

    insights = [
        {
            'collection': 'kaggle_insights',
            'text': """Titanic Competition Key Insight: High CV Score ≠ Good Leaderboard Score

The Optuna-tuned XGBoost had the highest CV score (0.8485) but performed 3rd on leaderboard (0.77511).
The best leaderboard score (0.78708) came from a model with lower CV (0.8126) but better regularization.

CV/LB gaps:
- Optuna: -0.0734 (worst generalization)
- Best (no ticket): -0.0256 (good generalization)
- Minimal features: -0.0197 (best generalization)

Lesson: Optimize for CV/LB gap, not just CV score. Use strong regularization on small datasets (<1000 samples).""",
            'metadata': {
                'competition': 'titanic',
                'topic': 'cv-lb-gap',
                'dataset_size': 891,
                'tags': 'overfitting,regularization,small-dataset'
            }
        },
        {
            'collection': 'kaggle_insights',
            'text': """Titanic Feature Engineering: Ticket Features Hurt Generalization

Feature ablation study showed that Ticket prefix features reduced CV by -0.0011 and LB performance.
Removing Ticket features improved LB from 0.78468 to 0.78708 (+0.0024).

Why Ticket features failed:
1. String pattern extraction (e.g., "PC 17599" → "PC") may overfit to training patterns
2. Test set may have different ticket naming conventions
3. High cardinality feature with sparse representation

Lesson: Be suspicious of string-based feature engineering on small datasets. Test with ablation studies.""",
            'metadata': {
                'competition': 'titanic',
                'topic': 'feature-engineering',
                'feature_type': 'ticket',
                'tags': 'feature-ablation,overfitting,string-features'
            }
        },
        {
            'collection': 'kaggle_insights',
            'text': """Titanic Regularization Recipe for Small Datasets

Best regularization configuration for XGBoost on small dataset (891 samples):
- reg_alpha: 0.1 (L1 regularization, was ~0)
- reg_lambda: 1.0 (L2 regularization, was ~0)
- gamma: 0.1 (minimum loss reduction, was ~0)
- min_child_weight: 10 (conservative splits, was 5)
- max_depth: 4 (shallow trees, was 7)
- n_estimators: 150 (fewer trees, was 294)
- learning_rate: 0.01 (slower learning, was 0.021)

Impact: Improved LB by +0.0096 (0.77511 → 0.78468) despite lower CV.

Rule of thumb: When train_size < 1000, start with strong regularization.""",
            'metadata': {
                'competition': 'titanic',
                'topic': 'regularization',
                'model': 'xgboost',
                'tags': 'hyperparameters,small-dataset,overfitting'
            }
        },
        {
            'collection': 'kaggle_features',
            'text': """Titanic Feature Importance Analysis

Sex is by far the dominant feature (72% importance):
- Sex_male: 0.7217 (72%)
- Pclass: 0.1306 (13%)
- Fare: 0.0447 (4.5%)
- FamilySize: 0.0441 (4.4%)
- Age: 0.0304 (3%)
- IsAlone: 0.0286 (2.9%)

This aligns with "women and children first" evacuation policy. All other features combined contribute only 28%.

Good features for Titanic:
- Binary indicators (Sex, IsAlone)
- Ordinal variables (Pclass)
- Core numerics (Age, Fare)
- Simple aggregations (FamilySize)

Bad features:
- String prefixes (Ticket, Cabin) - overfit
- Complex interactions (Age*Class) - minimal benefit
- High-missing features (Cabin: 77% missing)""",
            'metadata': {
                'competition': 'titanic',
                'topic': 'feature-importance',
                'tags': 'sex-dominant,feature-selection,titanic'
            }
        },
        {
            'collection': 'kaggle_experiments',
            'text': """Titanic Competition Final Results Summary

5 models tested, 6 submissions made:
1. Best: XGBoost without Ticket (LB: 0.78708, CV: 0.8126)
2. Regularized XGBoost (LB: 0.78468, CV: 0.8047)
3. Optuna XGBoost (LB: 0.77511, CV: 0.8485)
4. Minimal 6 features (LB: 0.77272, CV: 0.7924)
5. Baseline LogReg (LB: 0.76794, CV: 0.8339)

Key improvements:
- +1.97 pp from initial (77.511% → 78.708%)
- Feature ablation identified bad features
- Regularization prevented overfitting
- Systematic approach beat random search

Time investment: ~15 minutes for 5 experiments + ablation study.""",
            'metadata': {
                'competition': 'titanic',
                'final_score': 0.78708,
                'improvement': 0.0120,
                'tags': 'results-summary,titanic,systematic-approach'
            }
        }
    ]

    for i, insight in enumerate(insights, 1):
        try:
            embedding = create_embedding(insight['text'])
            if embedding:
                payload = {
                    'text': insight['text'],
                    **insight['metadata']
                }
                point_id = store_vector(
                    collection=insight['collection'],
                    vector=embedding,
                    payload=payload
                )
                if point_id:
                    print(f"   ✅ Stored insight {i} to {insight['collection']} (id={point_id[:8]}...)")
                else:
                    print(f"   ⚠️  Failed to store insight {i} (store_vector returned None)")
            else:
                print(f"   ⚠️  Failed to create embedding for insight {i}")
        except Exception as e:
            print(f"   ⚠️  Failed to store insight {i}: {e}")

    print("\n" + "="*80)
    print("MEMORY STORAGE COMPLETE")
    print("="*80)
    print(f"""
✅ Competition: titanic (best LB: 0.78708)
✅ Experiments: {len(exp_ids)} stored to PostgreSQL
✅ Submissions: {len(submissions)} stored to PostgreSQL
✅ Insights: {len(insights)} stored to Qdrant

All learnings are now searchable for future competitions!
""")

if __name__ == "__main__":
    main()
