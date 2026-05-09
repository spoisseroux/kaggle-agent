# AutoKaggle Multi-Agent Integration Design

**Date:** 2026-05-08  
**Status:** Design Document  
**Goal:** Integrate AutoKaggle-style multi-agent architecture into Kaggle Agent while preserving existing functionality and minimizing Claude Code API costs

## Executive Summary

This document outlines how to evolve the current Kaggle Agent into an AutoKaggle-inspired multi-agent system that:
- Maintains the existing top-level orchestrator for Telegram/routing
- Integrates all recent work (DeepEval, semantic search, /replay skill)
- Minimizes Claude Code API usage via aggressive local GPU offloading
- Uses dynamic model switching (qwen3:14b for boilerplate, Claude for reasoning)
- Preserves all existing functionality without breaking changes

## Current Architecture vs AutoKaggle

### Current Kaggle Agent Architecture

```
┌─────────────────────────────────────────────────────┐
│         Claude Code (Top-Level Orchestrator)        │
│  - Reads CLAUDE.md workflow (Phase 0-8)             │
│  - Routes tasks to Ollama for boilerplate           │
│  - Handles Telegram via message_bus.py              │
│  - Manages memory (Postgres + Qdrant)               │
│  - MLflow experiment tracking                       │
└─────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Ollama Client│  │  Skill System│  │ Data Pipeline│
│  qwen3:14b   │  │ /eval /replay│  │ refine→enrich│
│  (local GPU) │  │ /search-similar│ │ →synthetic   │
└──────────────┘  └──────────────┘  └──────────────┘
```

**Strengths:**
- Working end-to-end (research → submission)
- Local GPU offloading already in place (qwen3:14b)
- Rich tooling (DeepEval, semantic search, MLflow, Qdrant)
- Human-in-the-loop via Telegram (ask_human.py)
- Strong cost optimization (Ollama for 80%+ of work)

**Weaknesses:**
- Monolithic orchestrator (Claude Code does ALL reasoning)
- No task decomposition (phases hardcoded in CLAUDE.md)
- Limited specialization (same model reasons about EDA, feature eng, modeling)
- No structured code review or feedback loops
- Context bloat (all history in one conversation)

### AutoKaggle Architecture

```
┌─────────────────────────────────────────────────────┐
│              Main Orchestrator (gpt-4o)              │
│  - Phase sequencing (6 phases)                      │
│  - State management (propagates between agents)     │
└─────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────┬─────────────┐
        ▼                  ▼              ▼             ▼
┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐
│   Reader     │  │   Planner    │  │Developer │  │ Reviewer │
│ (gpt-4o-mini)│  │   (gpt-4o)   │  │ (gpt-4o) │  │(gpt-4o-  │
│              │  │              │  │          │  │  mini)   │
└──────────────┘  └──────────────┘  └──────────┘  └──────────┘
                                                        │
                                                        ▼
                                                ┌──────────────┐
                                                │  Summarizer  │
                                                │(gpt-4o-mini) │
                                                └──────────────┘
```

**Workflow per phase:**
1. **Reader**: Parse competition docs → structured summary
2. **Planner**: Break phase into ≤4 tasks with methodologies
3. **Developer**: Implement → Execute → Debug (max 5 retries)
4. **Reviewer**: Validate outputs, provide critical feedback
5. **Summarizer**: Document decisions/findings

**Strengths:**
- Clear separation of concerns (reading ≠ planning ≠ coding)
- Iterative debugging with error localization
- Structured feedback loops (Reviewer catches issues early)
- Cost tiering (gpt-4o only for Planner/Developer)
- Phase isolation (errors don't propagate)

**Weaknesses:**
- All OpenAI API (expensive for extended runs)
- No local GPU utilization
- Limited human-in-the-loop (only after Planner)
- No long-term memory (Qdrant, semantic search)
- No skill system or experiment replay

## Proposed Hybrid Architecture

### Design Principles

1. **Preserve Top-Level Orchestrator**: Claude Code remains the entry point for Telegram routing, human interactions, and overall workflow
2. **Introduce Specialized Agents**: Implement AutoKaggle-style agents as Python modules (NOT separate Claude Code instances)
3. **Maximize Local GPU**: Use qwen3:14b for 90%+ of agent work; reserve Claude only for critical reasoning
4. **Integrate Existing Tools**: Agents leverage DeepEval, semantic search, /replay, MLflow
5. **Zero Breaking Changes**: Existing CLAUDE.md workflow still works; agents are opt-in enhancement

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│         Claude Code Top-Level Orchestrator                          │
│  - Telegram routing (message_bus.py)                                │
│  - Human-in-the-loop (ask_human.py)                                 │
│  - Phase 0 research + submission approval                           │
│  - Multi-agent coordination (NEW)                                   │
└─────────────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼───────────────────┬─────────────┐
        ▼                  ▼                   ▼             ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Reader Agent │  │ Planner Agent│  │Developer Agent│  │ Reviewer Agent│
│ (qwen3:14b)  │  │ (qwen3:14b   │  │ (qwen3:14b   │  │ (qwen3:14b)  │
│              │  │  + Claude*)  │  │  + Claude*)  │  │              │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
       │                  │                 │                  │
       └──────────────────┼─────────────────┴──────────────────┘
                          ▼
                 ┌──────────────┐
                 │  Summarizer  │
                 │  (qwen3:14b) │
                 └──────────────┘
                          │
        ┌─────────────────┼─────────────────┬─────────────────┐
        ▼                 ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  DeepEval    │  │Semantic Search│  │  /replay     │  │   MLflow     │
│ (RTX 5070)   │  │  (Qdrant +   │  │              │  │  Tracking    │
│              │  │   RTX 5070)  │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

**\*Claude invoked only when:**
- Planner encounters ambiguous phase goals → strategic decision needed
- Developer debugging exceeds 3 retries → requires root cause analysis
- Any agent explicitly requests "human reasoning" mode

### Agent Specifications

#### 1. Reader Agent (`core/agents/reader.py`)

**Purpose:** Parse competition docs, data samples, past notebook insights

**Input:**
- `overview.txt` (Kaggle competition description)
- Sample data (train.csv head, test.csv head)
- `.claude/notebook_insights_{slug}.md` (from ingest_notebooks.py)

**Output:**
- `competition_info.txt`: Structured summary
  - Problem type (classification, regression, time series)
  - Evaluation metric
  - Data shape, features, target
  - Known issues/leaks from discussions
  - Top notebook approaches

**Implementation:**
```python
from core.ollama_client import generate

def read_competition(slug: str) -> dict:
    """Read and structure competition information."""
    overview = Path(f"competitions/active/{slug}/overview.txt").read_text()
    train_head = pd.read_csv(f"data/{slug}/train.csv", nrows=5)
    insights = Path(f".claude/notebook_insights_{slug}.md").read_text()
    
    prompt = f"""
    Parse this Kaggle competition:
    
    Overview: {overview}
    
    Data sample: {train_head.to_string()}
    
    Notebook insights: {insights}
    
    Generate JSON with:
    - problem_type
    - eval_metric
    - data_shape
    - features (list with types)
    - target_info
    - known_issues
    - winning_approaches
    """
    
    result = generate(prompt, model="qwen3:14b", system="You are a data science competition analyst.")
    return json.loads(result)
```

**Cost:** 100% Ollama (zero Claude API)

#### 2. Planner Agent (`core/agents/planner.py`)

**Purpose:** Decompose phases into ≤4 executable tasks with methodologies

**Input:**
- Phase name (e.g., "Feature Engineering")
- Competition info (from Reader)
- Previous phase outputs (state dict)
- Semantic search results (similar features from past competitions)

**Output:**
- Task list with detailed methodologies
- Estimated compute time per task
- Dependencies between tasks

**Implementation:**
```python
from core.ollama_client import generate
from core.semantic_search import search_code

def plan_phase(phase: str, comp_info: dict, state: dict) -> dict:
    """Break phase into executable tasks."""
    
    # Search for similar past work
    similar = search_code(
        f"{phase} {comp_info['problem_type']} {comp_info['eval_metric']}",
        limit=3,
        category=_phase_to_category(phase)
    )
    
    prompt = f"""
    Plan the {phase} phase for this competition:
    
    Problem: {comp_info['problem_type']}, metric: {comp_info['eval_metric']}
    Current state: {state}
    
    Similar past approaches:
    {_format_similar(similar)}
    
    Generate ≤4 tasks with:
    - Task name
    - Methodology (detailed steps)
    - Expected output
    - Compute estimate
    - Dependencies
    
    Prefer approaches that worked in similar competitions.
    """
    
    # Try Ollama first (80% of cases)
    result = generate(prompt, model="qwen3:14b", think=True, max_tokens=3000)
    
    # If plan is vague or contradictory, escalate to Claude
    if _is_plan_ambiguous(result):
        # THIS is where Claude Code gets invoked via ask_human.py
        # User sees: "Planner needs strategic input: [explain ambiguity]. Approve qwen plan or provide guidance?"
        result = _escalate_to_claude(prompt, comp_info)
    
    return json.loads(result)
```

**Cost:** 80% Ollama, 20% Claude (only for ambiguous cases)

#### 3. Developer Agent (`core/agents/developer.py`)

**Purpose:** Implement tasks, execute, debug (max 5 retries)

**Input:**
- Task specification (from Planner)
- ML tools library (validated functions)
- Previous code (for context)

**Output:**
- Working code
- Execution results
- Unit test results
- Artifact paths

**Implementation:**
```python
from core.ollama_client import generate
from core.evals import run_deepeval

def develop_task(task: dict, tools_library: dict, state: dict) -> dict:
    """Implement and debug task code."""
    
    # Generate initial code
    prompt = f"""
    Implement this task:
    {task['name']}: {task['methodology']}
    
    Use these validated tools:
    {_format_tools(tools_library)}
    
    Previous code context:
    {state.get('code', '')}
    
    Write complete, executable Python code.
    """
    
    code = generate(prompt, model="qwen3:14b", think=True, max_tokens=4000)
    
    # Iterative debugging (max 5 attempts)
    for attempt in range(5):
        result = _execute_code(code)
        
        if result['success']:
            # Validate with DeepEval
            eval_result = run_deepeval(code, task, state)
            if eval_result['pass']:
                return {'code': code, 'output': result['output'], 'eval': eval_result}
        
        # Debug
        error = result.get('error', '')
        
        if attempt < 3:
            # Ollama debugging
            code = _debug_with_ollama(code, error, task)
        else:
            # Escalate to Claude for complex debugging
            code = _escalate_debug_to_claude(code, error, task, state)
    
    raise Exception(f"Failed to implement {task['name']} after 5 retries")
```

**Cost:** 60% Ollama, 40% Claude (debugging is harder)

#### 4. Reviewer Agent (`core/agents/reviewer.py`)

**Purpose:** Validate outputs, provide critical feedback

**Input:**
- Generated code
- Task specification
- Execution results
- DeepEval results

**Output:**
- Pass/fail decision
- Critical feedback (logic errors, data leaks, efficiency issues)
- Suggestions for improvement

**Implementation:**
```python
from core.ollama_client import generate

def review_code(code: str, task: dict, results: dict, eval_results: dict) -> dict:
    """Provide critical code review."""
    
    prompt = f"""
    Review this code for the task: {task['name']}
    
    Code:
    ```python
    {code}
    ```
    
    Execution results: {results}
    DeepEval metrics: {eval_results}
    
    Check for:
    - Logic errors
    - Data leakage
    - Efficiency issues
    - Edge case handling
    - Consistency with previous phases
    
    Provide:
    - pass: true/false
    - issues: list of critical problems
    - suggestions: list of improvements
    """
    
    review = generate(prompt, model="qwen3:14b", system="You are a critical code reviewer.")
    return json.loads(review)
```

**Cost:** 100% Ollama

#### 5. Summarizer Agent (`core/agents/summarizer.py`)

**Purpose:** Document phase execution (decisions, findings, reasoning)

**Input:**
- Phase name
- All task results
- Code generated
- Review feedback

**Output:**
- Phase summary report
- Key findings
- Decisions made and why
- Next phase recommendations

**Implementation:**
```python
from core.ollama_client import generate
from core.format_telegram import format_phase_summary

def summarize_phase(phase: str, tasks: list, state: dict) -> dict:
    """Generate comprehensive phase summary."""
    
    prompt = f"""
    Summarize the {phase} phase execution:
    
    Tasks completed:
    {_format_tasks(tasks)}
    
    Code changes:
    {state.get('code_summary', '')}
    
    Results:
    {state.get('results', '')}
    
    Generate:
    - key_findings: bullet list
    - decisions_made: with reasoning
    - metrics: CV score, feature count, etc.
    - next_phase_recommendations
    - telegram_summary: concise version for notification
    """
    
    summary = generate(prompt, model="qwen3:14b", max_tokens=2000)
    result = json.loads(summary)
    
    # Log to MLflow
    _log_phase_summary(phase, result)
    
    # Notify user
    telegram_msg = format_phase_summary(result['telegram_summary'])
    subprocess.run(["python", "core/notify.py", telegram_msg])
    
    return result
```

**Cost:** 100% Ollama

### Orchestrator Integration

**New file:** `core/orchestrator.py`

```python
"""Multi-agent orchestrator for AutoKaggle-style workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import logging

from core.agents.reader import read_competition
from core.agents.planner import plan_phase
from core.agents.developer import develop_task
from core.agents.reviewer import review_code
from core.agents.summarizer import summarize_phase
from core.experiment_tracker import log_experiment
from core.notify import notify

log = logging.getLogger(__name__)

PHASES = [
    "Background Understanding",
    "Preliminary EDA",
    "Data Cleaning",
    "In-Depth EDA",
    "Feature Engineering",
    "Model Building",
]


class MultiAgentOrchestrator:
    """Coordinate multi-agent workflow."""
    
    def __init__(self, competition_slug: str):
        self.slug = competition_slug
        self.state = {}
        self.history = []
    
    def run(self, start_phase: int = 0, end_phase: int | None = None) -> dict:
        """Execute multi-agent workflow."""
        end_phase = end_phase or len(PHASES)
        
        # Phase 0: Reader
        if start_phase == 0:
            notify(f"📖 Reader agent analyzing {self.slug}")
            comp_info = read_competition(self.slug)
            self.state['competition_info'] = comp_info
            notify(f"✓ Competition parsed: {comp_info['problem_type']}, metric: {comp_info['eval_metric']}")
        
        # Phases 1-6
        for i in range(max(start_phase, 1), end_phase):
            phase = PHASES[i - 1]
            notify(f"🔧 Starting {phase}")
            
            # 1. Plan
            plan = plan_phase(phase, self.state['competition_info'], self.state)
            log.info(f"{phase} plan: {len(plan['tasks'])} tasks")
            
            # 2. Develop each task
            phase_results = []
            for task in plan['tasks']:
                notify(f"  ⚙️  {task['name']} (~{task['compute_estimate']})")
                
                dev_result = develop_task(task, self._load_tools(), self.state)
                
                # 3. Review
                review = review_code(
                    dev_result['code'],
                    task,
                    dev_result['output'],
                    dev_result['eval']
                )
                
                if not review['pass']:
                    # Retry with feedback
                    dev_result = self._retry_with_feedback(task, review)
                
                phase_results.append({'task': task, 'result': dev_result, 'review': review})
            
            # 4. Summarize
            summary = summarize_phase(phase, phase_results, self.state)
            self.state[f'{phase}_summary'] = summary
            self.history.append({'phase': phase, 'summary': summary})
            
            notify(f"✓ {phase} complete — {summary['key_findings'][0]}")
        
        return {'state': self.state, 'history': self.history}
    
    def _load_tools(self) -> dict:
        """Load ML tools library."""
        # TODO: implement validated function library
        return {}
    
    def _retry_with_feedback(self, task: dict, review: dict) -> dict:
        """Retry task with reviewer feedback."""
        # TODO: implement feedback loop
        raise NotImplementedError


def run_multi_agent(slug: str, start_phase: int = 0) -> dict:
    """Quick helper to run multi-agent workflow."""
    orch = MultiAgentOrchestrator(slug)
    return orch.run(start_phase=start_phase)
```

### Backward Compatibility

**CLAUDE.md remains unchanged.** Claude Code can still operate in single-agent mode:

```python
# Option 1: Traditional single-agent mode (existing behavior)
# Just follow CLAUDE.md phases 0-8 as before

# Option 2: Multi-agent mode (opt-in)
from core.orchestrator import run_multi_agent

result = run_multi_agent("store-sales-time-series-forecasting", start_phase=1)
```

**When to use each mode:**
- **Single-agent:** Quick experiments, debugging, one-off tasks
- **Multi-agent:** Full competition runs, complex feature engineering, production submissions

## Cost Optimization Strategy

### Token Usage Breakdown (AutoKaggle vs Proposed)

**AutoKaggle (all OpenAI):**
- Reader: gpt-4o-mini (~500 tokens/phase) × 6 = 3K tokens
- Planner: gpt-4o (~2000 tokens/phase) × 6 = 12K tokens
- Developer: gpt-4o (~5000 tokens/phase × 3 retries avg) × 6 = 90K tokens
- Reviewer: gpt-4o-mini (~1000 tokens/phase) × 6 = 6K tokens
- Summarizer: gpt-4o-mini (~800 tokens/phase) × 6 = 5K tokens
- **Total:** ~116K tokens/competition (~$2.32 at GPT-4o pricing)

**Proposed (Ollama + Claude Code):**
- Reader: qwen3:14b (local, $0)
- Planner: qwen3:14b 80%, Claude 20% (~2.4K Claude tokens)
- Developer: qwen3:14b 60%, Claude 40% (~36K Claude tokens)
- Reviewer: qwen3:14b (local, $0)
- Summarizer: qwen3:14b (local, $0)
- **Total:** ~38.4K Claude tokens/competition (~$0.77)

**Savings:** 67% reduction in API costs

### Dynamic Model Switching

**When to use qwen3:14b:**
- All boilerplate code generation
- Straightforward debugging (syntax errors, import errors)
- Data loading, preprocessing scaffolds
- Simple feature implementations (lag, rolling windows)
- Summaries and documentation
- Code reviews (no complex reasoning needed)

**When to escalate to Claude Code:**
- Ambiguous phase goals (Planner can't decide between approaches)
- Complex debugging (3+ retries, unclear root cause)
- Strategic decisions (trade-off analysis, approach selection)
- Human-in-the-loop needed (ask_human.py)

**Escalation triggers:**
```python
def should_escalate(context: str, agent: str, attempt: int) -> bool:
    """Decide if task needs Claude reasoning."""
    
    # Debugging: escalate after 3 Ollama attempts
    if agent == "Developer" and attempt >= 3:
        return True
    
    # Planning: escalate if qwen plan has contradictions
    if agent == "Planner" and _has_contradictions(context):
        return True
    
    # Explicit request from any agent
    if "REQUEST_CLAUDE_REASONING" in context:
        return True
    
    return False
```

### LoRA Considerations

**Not recommended for current setup because:**
1. qwen3:14b already domain-adapted (Qwen3 trained on code)
2. Fine-tuning requires 100+ quality examples (we don't have labeled agent outputs yet)
3. LoRA inference overhead negates speed gains on small batches
4. Better ROI: focus on prompt engineering + tool libraries

**Future consideration:**
- After 50+ competition runs, collect high-quality agent trajectories
- Fine-tune Qwen3 on:
  - Reader: competition_info.txt extraction
  - Developer: feature engineering patterns
- Expect 10-20% quality improvement + 5-10% speedup

## Integration Checklist

### Phase 1: Agent Infrastructure (4-5 hours)

- [ ] Create `core/agents/` directory
- [ ] Implement `core/agents/reader.py`
- [ ] Implement `core/agents/planner.py`
- [ ] Implement `core/agents/developer.py`
- [ ] Implement `core/agents/reviewer.py`
- [ ] Implement `core/agents/summarizer.py`
- [ ] Create `core/orchestrator.py`
- [ ] Add escalation logic (Ollama → Claude)
- [ ] Write unit tests for each agent

### Phase 2: Tools Library (3-4 hours)

- [ ] Create `core/tools/` directory
- [ ] Implement data cleaning tools (7 functions)
  - Missing value handlers
  - Outlier detection
  - Duplicate removal
  - Type validation
- [ ] Implement feature engineering tools (11 functions)
  - Encoding (one-hot, target, frequency)
  - Scaling (standard, minmax, robust)
  - Correlation analysis
  - Lag features
  - Rolling aggregations
- [ ] Implement model building tools (3 functions)
  - Model selection helper
  - Training with CV
  - Ensemble builder
- [ ] Validate all tools on past competitions
- [ ] Generate tool documentation for agents

### Phase 3: Integration with Existing Systems (2-3 hours)

- [ ] Connect Reader to `scripts/ingest_notebooks.py` output
- [ ] Connect Planner to semantic search (`core/semantic_search.py`)
- [ ] Connect Developer to DeepEval (`core/evals.py`)
- [ ] Connect Summarizer to MLflow (`core/experiment_tracker.py`)
- [ ] Connect all agents to Telegram (`core/notify.py`)
- [ ] Add orchestrator to main workflow (opt-in mode)

### Phase 4: Testing & Validation (3-4 hours)

- [ ] Run multi-agent mode on `titanic` (known baseline)
- [ ] Run multi-agent mode on `store-sales-time-series-forecasting`
- [ ] Compare CV scores: single-agent vs multi-agent
- [ ] Measure token usage: verify 60%+ Ollama usage
- [ ] Validate escalation triggers (should be <20% of tasks)
- [ ] Test backward compatibility (single-agent mode still works)

### Phase 5: Documentation & Rollout (1-2 hours)

- [ ] Update `CLAUDE.md` with multi-agent usage
- [ ] Document agent responsibilities in `docs/agents.md`
- [ ] Create `docs/tools_library.md`
- [ ] Add troubleshooting guide for agent failures
- [ ] Update `TODO.md` to mark AutoKaggle study complete
- [ ] Write blog post for Kaggle writeup (optional)

**Total estimated time:** 13-18 hours

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Agents produce worse code than monolithic | Medium | High | Thorough testing on 3+ competitions; keep single-agent fallback |
| Token costs increase (over-escalation) | Low | Medium | Monitor escalation rate; tune triggers conservatively |
| Integration breaks existing workflows | Low | High | Maintain backward compatibility; gradual rollout |
| Tools library has bugs | Medium | Medium | Validate on past data; unit test all functions |
| Context bloat from agent communication | Medium | Low | Compress summaries; prune non-essential state |

## Success Metrics

**After 5 competitions using multi-agent mode:**

1. **Quality:** Average CV score ≥ single-agent baseline
2. **Cost:** ≥60% of work handled by Ollama (zero API cost)
3. **Speed:** Phase completion time ≤1.2× single-agent
4. **Reliability:** ≥80% of phases complete without human intervention
5. **Leaderboard:** Average rank ≤50th percentile (matches AutoKaggle)

## Conclusion

The proposed hybrid architecture combines the best of AutoKaggle (specialized agents, structured workflow) with Kaggle Agent's strengths (local GPU, existing tools, human-in-the-loop). By routing 60-80% of work to Ollama and reserving Claude for strategic reasoning, we achieve:

- **Better separation of concerns** (reading ≠ planning ≠ coding)
- **Structured feedback loops** (Reviewer catches issues early)
- **Cost efficiency** (67% API cost reduction)
- **Backward compatibility** (single-agent mode preserved)
- **Integration of all recent work** (DeepEval, semantic search, /replay)

Next step: **Begin Phase 1 implementation** (agent infrastructure).

---

## Appendix: Dynamic Model Switching Examples

### Example 1: Planner Escalation

```python
# Planner receives ambiguous competition info
comp_info = {
    'problem_type': 'time series',
    'eval_metric': 'RMSLE',
    'known_issues': ['hierarchical sales data', 'store closures', 'holidays']
}

# Qwen generates plan
qwen_plan = plan_with_ollama(comp_info)
# Result: contradictory tasks (wants both daily and weekly aggregations)

# Escalate to Claude via ask_human
choice = ask_human(f"""
Planner needs guidance on {comp_info['problem_type']}:

Qwen's plan has contradictions:
- Task 1: Daily lag features
- Task 2: Weekly rolling means
- But evaluation is weekly-level

Should we:
1) Focus on weekly features (align with eval)
2) Try both and ablate later
3) Hierarchical features (daily→weekly)

Reply with 1, 2, or 3.
""")

final_plan = refine_plan_with_choice(qwen_plan, choice)
```

### Example 2: Developer Debugging Escalation

```python
# Developer attempts feature engineering
code = generate_feature_code(task, ollama)

# Attempt 1: syntax error → fix with Ollama
# Attempt 2: import error → fix with Ollama
# Attempt 3: shape mismatch → fix with Ollama
# Attempt 4: still fails with cryptic pandas error

# Escalate to Claude Code's main reasoning
claude_fix = ask_claude_to_debug(code, error_history, task, state)
# Claude analyzes: "The issue is a broadcast failure because you're merging on 
# different date granularities. Train is daily, aggregation is weekly."

code = claude_fix
# Success!
```

### Example 3: Strategic Decision (No Escalation)

```python
# Planner encounters clear, unambiguous phase
comp_info = {
    'problem_type': 'binary classification',
    'eval_metric': 'AUC',
    'data_shape': (10000, 20),
    'known_issues': []
}

# Qwen generates solid plan
plan = plan_with_ollama(comp_info)
# Tasks: 1) EDA (distributions, correlations)
#        2) Missing value imputation
#        3) Encoding categoricals
#        4) Train LightGBM baseline

# No contradictions, no ambiguity → proceed with Ollama
# (Zero Claude tokens used)
```

## References

- [AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competitions](https://arxiv.org/abs/2410.20424)
- [AutoKaggle GitHub Repository](https://github.com/multimodal-art-projection/AutoKaggle)
- [AutoKaggle Project Site](https://m-a-p.ai/AutoKaggle.github.io/)
