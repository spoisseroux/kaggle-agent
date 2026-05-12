#!/usr/bin/env python3
"""Ensemble v49 - Hill Climbing (Kaggle Grandmaster Technique)

TECHNIQUE: Systematic weighted ensembling from NVIDIA Kaggle Grandmaster Playbook

PROCESS:
1. Start with strongest single model (v19 LightGBM)
2. Add alternative models with varying weights
3. Keep combinations that improve validation scores
4. Repeat until no gains

MODELS TO COMBINE:
- v19 (LightGBM, CV 0.357) - base model
- v40 (XGBoost simple, CV 0.522) - different architecture
- v47 (LightGBM + DOW features, CV 0.432) - different features

STRATEGY:
- Test weight combinations for each model pair
- Evaluate on validation set
- Keep best performing combination
- Add third model if improves further

Expected: Capture complementary strengths of different approaches
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import itertools
import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from core.langfuse_logger import log_kaggle_model
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False

SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions():
    """Load predictions from existing models."""
    # Load v19 (LightGBM baseline)
    v19_file = list(SUBMISSION_DIR.glob("lgbm_v19_*.csv"))[0]
    v19 = pd.read_csv(v19_file)

    # Load validation predictions (need to regenerate these)
    # For now, we'll use submission files and assume we can reconstruct validation
    # In practice, we'd save validation predictions during training

    models = {
        'v19_lgbm': {
            'file': v19_file,
            'cv_score': 0.3572,
            'predictions': v19['sales'].values
        }
    }

    # Try to load other model predictions
    for model_name in ['v40', 'v47']:
        files = list(SUBMISSION_DIR.glob(f"*{model_name}*.csv"))
        if files:
            df = pd.read_csv(files[0])
            models[model_name] = {
                'file': files[0],
                'predictions': df['sales'].values
            }

    return models


def hill_climbing_ensemble(
    predictions_dict: dict,
    val_true: np.ndarray,
    test_ids: np.ndarray
) -> dict:
    """
    Perform hill climbing to find optimal model weights.

    Args:
        predictions_dict: Dict of {model_name: {'val_preds': array, 'test_preds': array, 'cv': float}}
        val_true: True validation labels
        test_ids: Test set IDs for final submission

    Returns:
        Dict with optimal weights and predictions
    """
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)

    print(f"Hill Climbing Ensemble with {n_models} models:")
    for name in model_names:
        print(f"  - {name}: CV {predictions_dict[name].get('cv', 'N/A')}")
    print()

    # Start with best single model
    best_model = min(
        predictions_dict.items(),
        key=lambda x: x[1].get('cv', float('inf'))
    )[0]

    current_ensemble = {best_model: 1.0}
    current_val_preds = predictions_dict[best_model]['val_preds']
    current_cv = np.sqrt(mean_squared_log_error(val_true, current_val_preds))

    print(f"Starting with: {best_model} (CV: {current_cv:.4f})")
    print()

    # Iteratively add models
    remaining_models = [m for m in model_names if m != best_model]

    iteration = 1
    improved = True

    while improved and remaining_models:
        improved = False
        best_addition = None
        best_new_cv = current_cv
        best_new_weights = None
        best_new_preds = None

        print(f"Iteration {iteration}:")

        # Try adding each remaining model
        for candidate in remaining_models:
            # Test different weight combinations
            # Try weights: 0.1, 0.2, 0.3, 0.4, 0.5 for the new model
            for new_weight in [0.1, 0.2, 0.3, 0.4, 0.5]:
                # Adjust existing weights proportionally
                test_weights = {m: w * (1 - new_weight) for m, w in current_ensemble.items()}
                test_weights[candidate] = new_weight

                # Compute weighted prediction
                ensemble_preds = sum(
                    predictions_dict[m]['val_preds'] * w
                    for m, w in test_weights.items()
                )

                # Evaluate
                cv = np.sqrt(mean_squared_log_error(val_true, ensemble_preds))

                if cv < best_new_cv:
                    best_new_cv = cv
                    best_addition = candidate
                    best_new_weights = test_weights
                    best_new_preds = ensemble_preds

                    print(f"  {candidate} @ {new_weight:.1f}: CV {cv:.4f} ✓")

        if best_addition and best_new_cv < current_cv:
            improvement = (current_cv - best_new_cv) / current_cv * 100
            print(f"  → Adding {best_addition}: {current_cv:.4f} → {best_new_cv:.4f} ({improvement:.2f}% better)")
            print(f"  → New weights: {best_new_weights}")
            print()

            current_ensemble = best_new_weights
            current_cv = best_new_cv
            current_val_preds = best_new_preds
            remaining_models.remove(best_addition)
            improved = True
            iteration += 1
        else:
            print(f"  No improvement found. Stopping.")
            print()

    # Final ensemble on test set
    test_preds = sum(
        predictions_dict[m]['test_preds'] * w
        for m, w in current_ensemble.items()
    )

    return {
        'weights': current_ensemble,
        'val_preds': current_val_preds,
        'test_preds': test_preds,
        'cv_score': current_cv
    }


def main():
    print("="*70)
    print("Ensemble v49 - Hill Climbing")
    print("="*70)
    print("Systematic weighted model combination")
    print()

    # NOTE: This is a simplified version that uses submission files
    # In production, we'd save validation predictions during training

    print("⚠️  NOTE: This implementation requires validation predictions")
    print("For a complete hill climbing ensemble, we need to:")
    print("1. Re-run models and save validation predictions")
    print("2. Load those predictions here")
    print("3. Optimize weights on validation set")
    print()
    print("Creating framework infrastructure instead...")

    # Create the hill climbing framework
    framework_code = """
# Hill Climbing Ensemble Framework

## Usage Pattern:

1. Train base models and save validation + test predictions:
   ```python
   # In each model script
   np.save(f'predictions/{model_name}_val.npy', val_preds)
   np.save(f'predictions/{model_name}_test.npy', test_preds)
   ```

2. Run hill climbing:
   ```python
   from ensemble_v49_hill_climbing import hill_climbing_ensemble

   predictions = {
       'v19_lgbm': {
           'val_preds': np.load('predictions/v19_val.npy'),
           'test_preds': np.load('predictions/v19_test.npy'),
           'cv': 0.3572
       },
       'xgb_v40': {
           'val_preds': np.load('predictions/v40_val.npy'),
           'test_preds': np.load('predictions/v40_test.npy'),
           'cv': 0.5220
       }
   }

   result = hill_climbing_ensemble(
       predictions,
       val_true=y_val,
       test_ids=test_df['id'].values
   )

   print(f"Optimal weights: {result['weights']}")
   print(f"Ensemble CV: {result['cv_score']:.4f}")
   ```

3. Submit ensemble predictions:
   ```python
   submission = pd.DataFrame({
       'id': test_ids,
       'sales': result['test_preds']
   })
   submission.to_csv('ensemble_submission.csv', index=False)
   ```

## Next Steps:

To use this framework:
1. Modify model scripts to save predictions
2. Re-run v19, v40, v47 with prediction saving
3. Run hill climbing optimization
4. Evaluate on leaderboard
"""

    print(framework_code)

    # Save framework documentation
    framework_path = SUBMISSION_DIR.parent / "docs" / "hill_climbing_ensemble.md"
    framework_path.parent.mkdir(exist_ok=True)
    with open(framework_path, 'w') as f:
        f.write(framework_code)

    print(f"Framework documentation saved: {framework_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
