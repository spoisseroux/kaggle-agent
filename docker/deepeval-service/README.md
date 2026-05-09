# DeepEval Service Deployment

## Overview
FastAPI service that wraps DeepEval for experiment quality validation.
Accessible via Tailscale IP on port 8001.

## Quick Start

### 1. Build and Deploy with Docker Compose

```bash
cd /path/to/kaggle-agent/docker
docker-compose -f docker-compose.ai-stack.yml up -d deepeval
```

### 2. Verify Health

```bash
curl http://localhost:8001/health
# or via Tailscale:
curl http://<tailscale-ip>:8001/health
```

Should return:
```json
{"status": "healthy", "service": "deepeval"}
```

### 3. Configure Agent Connection

Add to your `.env`:
```bash
DEEPEVAL_MODE=service
DEEPEVAL_URL=http://<tailscale-ip>:8001  # or http://localhost:8001
```

## API Endpoints

### POST /evaluate
Evaluate an experiment for quality and correctness.

**Request:**
```json
{
  "experiment_config": {
    "validation_strategy": "TimeSeriesSplit",
    "features": ["Lag_1", "Lag_7", "Roll_mean_7"],
    "hyperparameters": {"learning_rate": 0.05, "max_depth": 8},
    "model_type": "xgboost"
  },
  "cv_score": 0.350,
  "baseline_score": 0.367,
  "competition_type": "time-series"
}
```

**Response:**
```json
{
  "passed": true,
  "issues": [],
  "warnings": [],
  "score": 1.0,
  "recommendation": "✅ EXCELLENT: Experiment looks good. Safe to submit."
}
```

### POST /evaluate/features
Evaluate feature engineering quality.

**Request:**
```json
{
  "features": ["Lag_1", "Roll_mean_7", "day_of_week"],
  "competition_type": "time-series"
}
```

### GET /health
Health check endpoint.

## What It Checks

1. **Validation Strategy**
   - TimeSeriesSplit for time-series competitions
   - Catches wrong CV approaches

2. **Performance Regressions**
   - Flags experiments worse than baseline
   - Severity levels: >15% = critical, 5-15% = warning

3. **Suspicious Improvements**
   - Warns on >25% improvement (possible data leakage)

4. **Feature Engineering**
   - Time-series: checks for lags, rolling features
   - Detects future-leaking features

5. **Hyperparameter Sanity**
   - Learning rate bounds
   - Tree depth warnings

## Integration with Kaggle Agent

Use via Python:
```python
from core.evals import ExperimentEval

evaluator = ExperimentEval()  # Auto-connects to service if DEEPEVAL_MODE=service
result = evaluator.evaluate_experiment(
    experiment_config={...},
    cv_score=0.350,
    baseline_score=0.367
)

if not result["passed"]:
    print(f"❌ Issues: {result['issues']}")
```

Or via `/eval` skill (coming next).

## Logs

View logs:
```bash
docker logs claude-deepeval -f
```

## Updating

Rebuild after code changes:
```bash
docker-compose -f docker-compose.ai-stack.yml build deepeval
docker-compose -f docker-compose.ai-stack.yml up -d deepeval
```
