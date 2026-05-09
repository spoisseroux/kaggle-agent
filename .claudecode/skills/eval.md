---
name: eval
description: Evaluate experiment quality using DeepEval (LLM-based validation)
---

# /eval - Experiment Quality Evaluation

Evaluates a Kaggle experiment configuration using LLM-based analysis (Ollama + RTX 5070).

## Usage

```
/eval --cv 0.350 --baseline 0.367 --validation TimeSeriesSplit
/eval --experiment experiment_name
/eval --last  # Evaluate most recent experiment
```

## What It Does

1. **Extracts experiment details** from MLflow or provided parameters
2. **Runs LLM evaluation** using qwen3:14b on RTX 5070
3. **Checks for issues**:
   - Wrong validation strategy (e.g., train_test_split for time-series)
   - Data leakage in features
   - Performance regressions
   - Hyperparameter sanity
4. **Returns verdict**: Pass/Fail with detailed analysis

## Implementation

Run this command:

```python
python -m core.experiment_evaluator --cv {cv_score} --baseline {baseline} --validation {strategy}
```

Or use the evaluator directly in Python:

```python
from core.experiment_evaluator import ExperimentEvaluator

evaluator = ExperimentEvaluator()
result = evaluator.evaluate_experiment(
    experiment_config={
        "validation_strategy": "TimeSeriesSplit",
        "features": ["Lag_1", "Lag_7", "Roll_mean_7"],
        "hyperparameters": {"learning_rate": 0.05, "max_depth": 8},
        "model_type": "xgboost"
    },
    cv_score=0.350,
    baseline_score=0.367,
    competition_type="time-series"
)

if result["passed"]:
    print(f"✅ {result['recommendation']}")
else:
    print(f"❌ {result['recommendation']}")
    for issue in result["issues"]:
        print(f"  - {issue}")
```

## Integration Points

- **Pre-submission**: Automatically runs before asking for submission approval
- **Experiment tracking**: Results logged to experiments table
- **Frontend**: View eval history at `/kaggle/deepeval`

## Performance

- **Time**: 8-18 seconds per evaluation
- **Model**: qwen3:14b (14.8B params) on RTX 5070
- **Cost**: $0 (100% local)
