# Experiment Queue Usage Guide

## Quick Start

```python
from core.experiment_queue import ExperimentQueue

# Initialize queue
queue = ExperimentQueue(max_parallel=2)

# Submit experiments
exp_ids = []
for config in experiment_configs:
    exp_id = queue.submit(
        competition_slug="titanic",
        config=config,  # Dict with model, params, features
        priority=1,  # 1=highest, 10=lowest
        estimated_duration_min=10,
    )
    exp_ids.append(exp_id)

# Poll for next experiment to run
while True:
    next_exp = queue.get_next()
    if next_exp:
        # Run experiment...
        result = run_experiment(next_exp['config'])
        
        # Mark complete
        queue.complete(
            next_exp['id'],
            mlflow_run_id=result['run_id'],
            results={"cv_score": result['cv'], "train_score": result['train']},
        )
    else:
        break  # Queue empty or at capacity

# Get results summary
results = queue.get_results_summary("titanic")
best = results[0]  # Sorted by CV score DESC
```

## Integration with Multi-Agent System

```python
from core.orchestrator import MultiAgentOrchestrator
from core.experiment_queue import ExperimentQueue

# Planner generates experiment configs
research_report = {...}
experiment_configs = generate_experiment_batch(research_report)

# Submit to queue
queue = ExperimentQueue()
for config in experiment_configs:
    queue.submit("titanic", config, priority=config['priority'])

# Worker loop (runs in background)
while queue.list_active():
    next_exp = queue.get_next()
    if next_exp:
        # Multi-agent orchestrator executes
        orchestrator = MultiAgentOrchestrator(next_exp['competition_slug'])
        result = orchestrator.run_experiment(next_exp['config'])
        
        # Report results
        queue.complete(next_exp['id'], result['mlflow_run_id'], result['metrics'])
```

## Priority Strategy

**Priority 1-3: High (research-recommended approaches)**
- Research agent marks top approaches from literature
- Run these first to establish baseline

**Priority 4-6: Medium (variations and ensembles)**
- Hyperparameter tuning
- Feature combinations
- Ensemble experiments

**Priority 7-10: Low (exploratory)**
- New architectures to test
- Ablation studies
- Debugging runs

## Dependencies

```python
# Experiment B depends on Experiment A
exp_a = queue.submit("titanic", config_baseline, priority=1)
exp_b = queue.submit(
    "titanic",
    config_ensemble,
    priority=2,
    depends_on=[exp_a],  # Won't run until exp_a completes
)
```

## Monitoring

```sql
-- Active experiments
SELECT * FROM active_experiments;

-- Completed experiments ranked by CV
SELECT
    id,
    config->>'model' as model,
    results->>'cv_score' as cv,
    EXTRACT(EPOCH FROM (completed_at - started_at))/60 as duration_min
FROM experiment_queue
WHERE competition_slug = 'titanic' AND status = 'completed'
ORDER BY (results->>'cv_score')::float DESC;

-- Queue stats
SELECT
    status,
    COUNT(*) as count,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))/60) as avg_duration_min
FROM experiment_queue
WHERE competition_slug = 'titanic'
GROUP BY status;
```

## VRAM-Aware Scheduling

```python
# Queue checks max_parallel before dispatching
queue = ExperimentQueue(max_parallel=2)  # Only 2 experiments run simultaneously

# Larger models can override
queue.submit(
    "store-sales",
    {
        "model": "catboost",
        "params": {"iterations": 10000},  # Heavy model
    },
    gpu_required=True,
    estimated_duration_min=30,
)

# Light models can run in parallel
queue.submit(
    "store-sales",
    {"model": "logistic_regression"},
    gpu_required=False,  # Can run while GPU experiment running
)
```

## Error Handling

```python
try:
    result = run_experiment(config)
    queue.complete(exp_id, run_id, results, success=True)
except Exception as e:
    queue.complete(
        exp_id,
        run_id=None,
        results={},
        success=False,
        error_message=str(e),
    )
```

## Cancellation

```python
# Cancel queued experiment
queue.cancel(exp_id)

# View cancelled experiments
SELECT * FROM experiment_queue WHERE status = 'cancelled';
```

## Best Practices

1. **Estimate duration accurately** - helps with scheduling
2. **Use dependencies** - don't waste compute on ensembles before baselines
3. **Set priorities wisely** - research-recommended > exploratory
4. **Monitor max_parallel** - adjust based on VRAM availability
5. **Clean up old experiments** - archive completed runs after competition ends
