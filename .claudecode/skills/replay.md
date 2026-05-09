---
name: replay
description: Re-execute past experiment with modifications
---

# /replay - Experiment Replay

Re-run a past experiment with modifications to test hypotheses quickly.
Useful for A/B testing hyperparameters, feature sets, or validation strategies.

## Usage

```
/replay experiment_id --lr 0.1
/replay xgb_v3 --features "Lag_1,Lag_7,Roll_mean_7"
/replay last --validation TimeSeriesSplit
/replay lgbm_baseline --max-depth 10 --lr 0.05
```

## What It Does

1. **Loads experiment config** from MLflow or memory
2. **Applies modifications** (hyperparameters, features, validation)
3. **Re-runs training** with new config
4. **Compares results** to original run
5. **Logs to MLflow** as new experiment

## Modification Types

**Hyperparameters:**
- `--lr <float>` - Learning rate
- `--max-depth <int>` - Tree depth
- `--subsample <float>` - Subsample ratio
- `--colsample <float>` - Column sample ratio
- `--n-estimators <int>` - Number of trees

**Features:**
- `--features "feat1,feat2,..."` - Use specific features
- `--add-features "feat1,feat2"` - Add to existing
- `--remove-features "feat1,feat2"` - Remove from existing

**Validation:**
- `--validation <strategy>` - Change CV strategy
- `--n-splits <int>` - Number of CV folds

**Model:**
- `--model <type>` - Switch model (xgboost, lightgbm, catboost)

## Examples

**Test different learning rate:**
```python
from core.experiment_replay import replay_experiment

# Original: lr=0.05, CV=0.350
# Test: lr=0.1
result = replay_experiment(
    experiment_id="xgb_v3",
    modifications={"learning_rate": 0.1}
)

print(f"Original CV: {result['original_cv']}")
print(f"New CV: {result['new_cv']}")
print(f"Delta: {result['delta']}")
```

**Try different feature set:**
```python
result = replay_experiment(
    experiment_id="lgbm_baseline",
    modifications={
        "features": ["Lag_1", "Lag_7", "Lag_14", "Roll_mean_7"]
    }
)
```

**Switch validation strategy:**
```python
result = replay_experiment(
    experiment_id="xgb_v2",
    modifications={
        "validation_strategy": "TimeSeriesSplit",
        "n_splits": 5
    }
)
```

## Implementation

The replay system:
1. Queries MLflow for experiment metadata
2. Loads training code and config
3. Merges modifications with original config
4. Re-executes training
5. Compares metrics (CV, LB if available)
6. Logs as new experiment with `replayed_from` tag

## Use Cases

- **Hyperparameter sweep**: Test ranges quickly
- **Feature ablation**: Remove features to test importance
- **Validation testing**: Compare CV strategies
- **Model comparison**: Same features, different models
- **Quick iteration**: Avoid re-writing experiment code

## Performance

- Reuses existing data pipeline
- Skips EDA/preprocessing
- Just re-trains model
- ~5-10 min for typical experiment
