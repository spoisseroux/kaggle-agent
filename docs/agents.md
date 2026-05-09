# Multi-Agent System Documentation

AutoKaggle-inspired multi-agent architecture for Kaggle competitions.

## Overview

The multi-agent system provides a structured, phase-based workflow with specialized agents that handle different aspects of competition participation. This approach:

- **Reduces API costs by 67%** through aggressive Ollama offloading
- **Provides structured workflow** with phase-based task decomposition
- **Generates comprehensive documentation** automatically
- **Enables iterative debugging** with up to 5 retry attempts per task
- **Integrates existing tools** (DeepEval, semantic search, MLflow, Telegram)

## Architecture

```
┌─────────────────────────────────────────────┐
│    Claude Code (Top-Level Orchestrator)     │
│  - Human-in-the-loop via Telegram          │
│  - Strategic decision making                │
│  - Agent escalation handling                │
└─────────────────────────────────────────────┘
                     │
      ┌──────────────┼──────────────┬─────────────┐
      ▼              ▼              ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Reader  │  │ Planner  │  │Developer │  │ Reviewer │
│  (Ollama)│  │ (Ollama  │  │ (Ollama  │  │ (Ollama) │
│   100%   │  │   80%)   │  │   60%)   │  │   100%   │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │  Summarizer  │
                                         │   (Ollama)   │
                                         │     100%     │
                                         └──────────────┘
```

## Agents

### Reader Agent

**File:** `core/agents/reader.py`  
**Purpose:** Parse competition documentation into structured JSON  
**Cost:** 100% Ollama (zero API cost)

**Input:**
- `competitions/active/{slug}/overview.txt`
- Sample data from `data/{slug}/train.csv` and `test.csv`
- `.claude/notebook_insights_{slug}.md` (from `scripts/ingest_notebooks.py`)

**Output:** `competition_info.json` containing:
```json
{
  "problem_type": "classification|regression|time_series",
  "eval_metric": "accuracy|rmse|auc|etc",
  "data_shape": {"train_rows": X, "train_cols": Y, "test_rows": Z},
  "features": [{"name": "...", "type": "...", "missing_pct": 0.0}],
  "target_info": {"name": "...", "type": "...", "distribution": "..."},
  "known_issues": ["data leaks", "train/test differences", "..."],
  "winning_approaches": ["feature patterns from top notebooks"],
  "summary": "one paragraph competition overview"
}
```

**Usage:**
```python
from core.agents.reader import read_competition, save_competition_info

comp_info = read_competition("titanic")
save_competition_info(comp_info, Path(".claude/competition_info_titanic.json"))
```

### Planner Agent

**File:** `core/agents/planner.py`  
**Purpose:** Decompose phases into executable tasks  
**Cost:** 80% Ollama, 20% Claude (escalates for ambiguous cases)

**Input:**
- Phase name (e.g., "Feature Engineering")
- Competition info from Reader
- Current state (previous phase outputs)

**Process:**
1. Search semantic index for similar past approaches
2. Generate ≤4 tasks with detailed methodologies using Ollama
3. Check for contradictions/ambiguities
4. Escalate to Claude Code if issues found

**Output:** Plan dict:
```json
{
  "phase": "Feature Engineering",
  "approach": "overall strategy description",
  "tasks": [
    {
      "name": "Create lag features",
      "methodology": "1) Load data 2) Create lags 1,7,14 3) Save to CSV",
      "expected_output": "features.csv with lag columns",
      "compute_estimate": "5 minutes",
      "dependencies": []
    }
  ],
  "search_results": [...],
  "llm_used": "ollama|claude"
}
```

**Escalation triggers:**
- Plan contains contradictory tasks (e.g., "daily lags" + "weekly aggregation" for weekly-level eval)
- Vague methodologies (<50 chars or contains "maybe", "unclear", "tbd")
- Too many or too few tasks (0 or >6)

**Usage:**
```python
from core.agents.planner import plan_phase

plan = plan_phase("Feature Engineering", comp_info, state)
print(f"{len(plan['tasks'])} tasks planned")
```

### Developer Agent

**File:** `core/agents/developer.py`  
**Purpose:** Implement code with iterative debugging  
**Cost:** 60% Ollama, 40% Claude (escalates after 3+ debug retries)

**Input:**
- Task specification from Planner
- ML tools library (`core/tools/`)
- Current state (previous code, data paths)

**Process:**
1. Generate code using Ollama (uses tools library where applicable)
2. Execute code in temporary sandbox
3. If error: debug with Ollama (attempts 1-3) or Claude (attempts 4-5)
4. Validate with DeepEval (if available)
5. Return working code or raise exception after 5 attempts

**Output:** Result dict:
```json
{
  "code": "import pandas as pd\n...",
  "output": "execution stdout",
  "eval": {"pass": true, "issues": [], "warnings": []},
  "attempts": 2,
  "llm_used": "ollama|both",
  "success": true
}
```

**Escalation triggers:**
- ≥3 debugging attempts with Ollama failed
- Error message contains "unclear root cause"
- Explicit `REQUEST_CLAUDE_REASONING` in context

**Usage:**
```python
from core.agents.developer import develop_task
from core.tools import TOOLS_LIBRARY

result = develop_task(task, TOOLS_LIBRARY, state)
print(f"Code generated in {result['attempts']} attempts")
```

### Reviewer Agent

**File:** `core/agents/reviewer.py`  
**Purpose:** Validate code and provide critical feedback  
**Cost:** 100% Ollama (zero API cost)

**Input:**
- Generated code
- Task specification
- Execution results

**Checks:**
- **Logic errors:** Incorrect implementations, off-by-one errors
- **Data leakage:** Using future data, target in features, train/test contamination
- **Efficiency:** Unnecessary loops, inefficient pandas operations
- **Edge cases:** Missing value handling, empty data, single row scenarios
- **Consistency:** Matches expected output, consistent with previous phases

**Output:** Review dict:
```json
{
  "pass": true,
  "issues": [
    {
      "severity": "critical|warning|info",
      "category": "logic|leakage|efficiency|edge_case|consistency",
      "description": "Specific issue found"
    }
  ],
  "suggestions": ["Actionable improvement 1", "..."],
  "score": 85,
  "summary": "One sentence assessment"
}
```

**Usage:**
```python
from core.agents.reviewer import review_code

review = review_code(code, task, exec_result, eval_result)
if not review["pass"]:
    print(f"Critical issues: {[i for i in review['issues'] if i['severity'] == 'critical']}")
```

### Summarizer Agent

**File:** `core/agents/summarizer.py`  
**Purpose:** Document phase execution  
**Cost:** 100% Ollama (zero API cost)

**Input:**
- Phase name
- All task results (code, outputs, reviews)
- Current state

**Process:**
1. Generate structured summary using Ollama
2. Save JSON and Markdown reports to `.claude/summaries/{slug}/`
3. Log to MLflow (if active run)
4. Send Telegram notification

**Output:** Summary dict:
```json
{
  "phase": "Feature Engineering",
  "key_findings": ["Created 10 lag features", "CV improved 5%"],
  "decisions_made": [
    {"decision": "Used target encoding", "reasoning": "Better than one-hot for high cardinality"}
  ],
  "metrics": {"cv_score": 0.85, "feature_count": 25, "train_time_seconds": 120},
  "artifacts": ["features.csv", "encoders.pkl"],
  "next_phase_recommendations": ["Try polynomial features", "Tune model hyperparameters"],
  "telegram_summary": "Feature Engineering complete. Created 25 features, CV: 0.85",
  "full_report": "# Feature Engineering\n\n..."
}
```

**Usage:**
```python
from core.agents.summarizer import summarize_phase

summary = summarize_phase("Feature Engineering", tasks, state, notify_telegram=True)
```

## Orchestrator

**File:** `core/orchestrator.py`  
**Class:** `MultiAgentOrchestrator`

Coordinates all 5 agents through a 6-phase workflow:

1. **Background Understanding** (Reader)
2. **Preliminary EDA**
3. **Data Cleaning**
4. **In-Depth EDA**
5. **Feature Engineering**
6. **Model Building**

Each phase follows: Planner → Developer (×N tasks) → Reviewer (per task) → Summarizer

**State Management:**
- Maintains cumulative state across phases
- Stores competition info, code, artifacts, metrics
- Exports full history to `.claude/multi_agent_history_{slug}.json`

**Usage:**
```python
from core.orchestrator import run_multi_agent

# Full workflow
result = run_multi_agent("titanic", start_phase=0, end_phase=6)

# Specific phases
result = run_multi_agent("store-sales", start_phase=4, end_phase=5)  # Just Feature Engineering

# Custom data directory
result = run_multi_agent("competition-slug", data_dir=Path("/custom/path"))
```

**Result:**
```json
{
  "competition_slug": "titanic",
  "state": {...},
  "history": [{phase, plan, results, summary}, ...],
  "phases_completed": 3
}
```

## ML Tools Library

**Location:** `core/tools/`

21 validated functions available to agents:

### Data Cleaning (7 tools)
- `handle_missing_values`: Various strategies (mean, median, mode, drop)
- `detect_outliers`: IQR or Z-score methods
- `remove_duplicates`: Drop duplicate rows
- `validate_dtypes`: Validate and coerce column types
- `fix_inconsistent_values`: Standardize categorical values
- `parse_dates`: Parse datetime columns
- `standardize_strings`: Lowercase, strip, remove special chars

### Feature Engineering (11 tools)
- `one_hot_encode`: One-hot encoding
- `target_encode`: Target encoding with smoothing
- `frequency_encode`: Count/frequency encoding
- `standard_scale`: StandardScaler (mean=0, std=1)
- `minmax_scale`: MinMaxScaler to [0,1]
- `robust_scale`: RobustScaler (median, IQR)
- `correlation_analysis`: Find redundant features
- `create_lag_features`: Time series lags
- `create_rolling_features`: Rolling aggregations
- `create_polynomial_features`: Polynomial and interactions
- `create_bins`: Discretize continuous variables

### Modeling (3 tools)
- `select_model`: Choose LightGBM, XGBoost, or CatBoost
- `train_with_cv`: K-Fold or TimeSeriesSplit training
- `build_ensemble`: Weighted average, rank average, or stacking

**Usage in agents:**
```python
from core.tools import TOOLS_LIBRARY

# Developer agent receives this and uses in prompts
tools = TOOLS_LIBRARY["feature_engineering"]
for tool in tools:
    print(f"{tool['name']}: {tool['description']}")
```

## LLM Interface

**File:** `core/llm_interface.py`  
**Purpose:** Abstract LLM calls for easy backend switching

**Backends:**
- `claude_code` (default): Escalates to Claude Code via `ask_human.py`
- `claude_api` (future): Direct Anthropic API calls via SDK

**Configuration:**
```bash
export CLAUDE_BACKEND="claude_code"  # or "claude_api"
export ANTHROPIC_API_KEY="sk-..."    # for claude_api backend
export CLAUDE_MODEL="claude-sonnet-4-5"
```

**Usage:**
```python
from core.llm_interface import ask_ollama, ask_claude, should_escalate

# Local Ollama
response = ask_ollama("Generate code for...", think=True)

# Escalate to Claude
if should_escalate("Developer", error, attempt=3)[0]:
    response = ask_claude("Debug this complex error...", reason="3+ attempts failed")
```

**Escalation logic:**
```python
def should_escalate(agent, context, attempt, error):
    # Developer: after 3 retries
    if agent == "Developer" and attempt >= 3:
        return True, f"Debugging failed after {attempt} attempts"
    
    # Planner: contradictions
    if agent == "Planner" and "contradiction" in context:
        return True, "Plan has contradictions"
    
    # Explicit request
    if "REQUEST_CLAUDE_REASONING" in context:
        return True, "Explicit escalation requested"
    
    return False, None
```

## Cost Analysis

**Baseline (single-agent):**
- All work via Claude Code API
- ~100K tokens/competition at $0.015/1K input, $0.075/1K output
- **Est. cost:** ~$3-5/competition

**Multi-agent (Ollama + selective escalation):**
- Reader: 100% Ollama = $0
- Planner: 80% Ollama, 20% Claude = ~$0.20
- Developer: 60% Ollama, 40% Claude = ~$1.20
- Reviewer: 100% Ollama = $0
- Summarizer: 100% Ollama = $0
- **Est. cost:** ~$1.40/competition

**Savings: 67% reduction**

## Troubleshooting

### Import Errors
Agents are imported via `core/agents/__init__.py`. Check that all agent files exist and have correct function signatures.

### Ollama Connection Failed
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart if needed
# (Ollama auto-starts on WSL)
```

### Escalation Not Working
Check `CLAUDE_BACKEND` env var:
```bash
echo $CLAUDE_BACKEND  # Should be "claude_code"
```

If using `claude_api`, set `ANTHROPIC_API_KEY`.

### Tasks Failing After Max Retries
- Check Developer logs for specific errors
- Increase `max_attempts` parameter (default: 5)
- Review tools library - may need to add specific utility

### State Not Persisting
State is stored in orchestrator instance. Export history after workflow:
```python
orch = MultiAgentOrchestrator("slug")
orch.run()
orch.export_history(Path(".claude/history.json"))
```

## Examples

### Full Competition Run
```python
from core.orchestrator import run_multi_agent

result = run_multi_agent(
    "store-sales-time-series-forecasting",
    start_phase=0,  # Reader
    end_phase=6,    # Model Building
)

print(f"Completed {result['phases_completed']} phases")
print(f"Final CV: {result['state'].get('metrics', {}).get('cv_score')}")
```

### Phase-by-Phase with Inspection
```python
from core.orchestrator import MultiAgentOrchestrator

orch = MultiAgentOrchestrator("titanic")

# Reader
orch.run(start_phase=0, end_phase=1)
comp_info = orch.state["competition_info"]
print(f"Problem: {comp_info['problem_type']}, Metric: {comp_info['eval_metric']}")

# Feature Engineering only
orch.run(start_phase=4, end_phase=5)
fe_summary = orch.get_phase_summary("Feature Engineering")
print(fe_summary["key_findings"])
```

### Custom Tools Library
```python
from core.tools import TOOLS_LIBRARY

# Add custom tool
TOOLS_LIBRARY["feature_engineering"].append({
    "name": "my_custom_feature",
    "func": my_function,
    "description": "Does custom transformation"
})

# Use in orchestrator
orch = MultiAgentOrchestrator("slug")
orch.tools_library = TOOLS_LIBRARY
orch.run()
```

## Future Enhancements

- **LoRA fine-tuning**: After 50+ competition runs, fine-tune Qwen3 on agent trajectories
- **Full DeepEval integration**: Connect Developer agent to full experiment evaluator
- **Stacking meta-learner**: Ensemble agent with trained meta-model instead of weighted average
- **Parallel task execution**: Run independent Developer tasks concurrently
- **Interactive plan refinement**: Human-in-the-loop after Planner generates initial plan
