# Learning Multi-Agent System

Dynamic, learning-oriented multi-agent system for autonomous Kaggle competition improvement.

## Architecture

**Learning Loop**: Analyze → Strategize → Implement → Evaluate → Curate → REPEAT

### Agents

1. **Analyst** (`analyst.py`)
   - Analyzes all past submissions and experiments
   - Calculates CV-LB gaps, model performance comparisons
   - Identifies patterns and bottlenecks
   - Uses Ollama to synthesize insights
   
2. **Strategist** (`strategist.py`)
   - Decides what experiments to try next
   - Creates prioritized experiment queue (not fixed phases!)
   - Estimates impact vs effort tradeoffs
   - Queries past learnings from Qdrant
   
3. **Engineer** (`engineer.py`)
   - Implements experiment code using Ollama
   - Retry logic with incremental fixes (max 3 attempts)
   - Validates syntax and imports before running
   - Maintains memory of past implementation errors
   
4. **Evaluator** (`evaluator.py`)
   - Assesses if experiment met success criteria
   - Performs root cause analysis on failures
   - Generates actionable next steps
   - Uses Ollama for evaluation reasoning
   
5. **Curator** (`curator.py`)
   - Stores learnings to Postgres (structured)
   - Records patterns to Qdrant (semantic search)
   - Updates competition CLAUDE.md with insights
   - Tags experiments in MLflow

### Orchestrator

**File**: `orchestrator.py`

Coordinates the 5 agents in an iterative learning loop with:
- Compute budget tracking
- Telegram progress notifications
- Langfuse observability tracing
- Graceful failure handling

## Usage

### Full Autonomous Run

```bash
python -m agents.orchestrator <competition_slug> [max_iterations] [compute_hours]
```

**Example**:
```bash
# Run 3 iterations with 1.5h budget on store-sales
python -m agents.orchestrator store-sales-time-series-forecasting 3 1.5
```

**What it does**:
1. Analyzes current state (past submissions, models, gaps)
2. Proposes top 3 experiments based on findings
3. Implements highest priority experiment
4. Evaluates results and performs RCA
5. Stores learnings and updates knowledge base
6. Repeats until budget exhausted or stopped

### Individual Agent Testing

Each agent can be run standalone for testing:

```bash
# Analyst - analyze competition state
python -m agents.analyst store-sales-time-series-forecasting

# Strategist - plan experiments (requires analysis JSON)
python -m agents.strategist store-sales-time-series-forecasting

# Engineer - implement experiment (requires experiment JSON)
python -m agents.engineer store-sales-time-series-forecasting '{"id": "exp_001", ...}'

# Evaluator - assess results (requires experiment + implementation JSON)
python -m agents.evaluator store-sales-time-series-forecasting '{"id": "exp_001", ...}' '{"success": true, ...}'

# Curator - store learnings (requires experiment + evaluation JSON)
python -m agents.curator store-sales-time-series-forecasting '{"id": "exp_001", ...}' '{"success": true, ...}'
```

## Key Differences from Fixed Pipeline

| Old Approach | Learning Multi-Agent |
|--------------|---------------------|
| Fixed 9 phases (0-8) | Dynamic experiment queue |
| No memory between runs | Every agent reads past work |
| Single pass | Iterative improvement loop |
| No failure analysis | Evaluator does RCA |
| No knowledge capture | Curator builds memory |
| Phase-based | Impact-based prioritization |

## Example Output

### Analyst Output
```
Best CV: 0.315 (xgb_v3_external_data_0.3154.csv)
Best LB: 0.464 (xgb_optuna_v1_0.3453.csv)

Insights:
  1. XGBoost models dominate but LightGBM untapped
  2. Large CV-LB gaps (0.69) indicate overfitting
  3. Optuna tuning achieved best balance (0.345 CV, 0.464 LB)
  4. Higher CV often underperforms on LB (validation mismatch)

Bottlenecks:
  1. Systemic overfitting (gaps up to 0.69)
  2. Missing feature importance analysis (no MLflow integration)

Patterns:
  1. External data causes extreme overfitting (LB 1.00)
  2. LightGBM's single submission has highest LB
```

### Strategist Output
```
exp_001: Improve test set lag features
  Priority: 1
  Hypothesis: Using 30-day rolling stats instead of single lag will reduce CV-LB gap
  Expected Impact: +0.02 to +0.05 LB
  Effort: 30 minutes

exp_002: Remove low-impact features
  Priority: 2
  Hypothesis: Dropping 15 weak features will reduce overfitting
  Expected Impact: +0.01 to +0.03 LB
  Effort: 15 minutes
```

## Dependencies

- **Ollama** (qwen3:14b): All agent reasoning
- **Postgres**: Structured memory (submissions, experiments, learnings)
- **Qdrant**: Semantic pattern storage (optional)
- **MLflow**: Experiment tracking (optional)
- **Langfuse**: Observability tracing (optional)

No LangChain required - uses existing primitives.

## Observability

All agent decisions tracked in Langfuse:
- Each agent call = span
- Orchestrator iteration = trace
- Experiments tagged with competition and outcome

View in Langfuse dashboard: `http://localhost:3000`

## Design Document

Full architecture details: `docs/learning_multiagent_design.md`
