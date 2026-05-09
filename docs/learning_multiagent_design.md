# Learning Multi-Agent System Design

**Problem with current multi-agent**: Rigid phase-based workflow (0-8) that doesn't learn from past experiments or adapt based on what's working.

**Vision**: Self-improving agent system that analyzes past submissions, identifies patterns, and autonomously iterates toward better scores.

## Core Principles

1. **Memory-first**: Every agent reads past experiments before acting
2. **Feedback loops**: Agents analyze results and adjust strategy
3. **Dynamic planning**: No fixed phases - adapt based on current state
4. **Iterative refinement**: Run-analyze-learn-improve cycle
5. **Build on winners**: Extract what worked, compose improvements

## Agent Roles (Redesigned)

### 1. Analyst Agent (NEW)
**Purpose**: Understand current state and identify opportunities

**Inputs**:
- All past submissions (from Postgres `kaggle_submissions`)
- All experiments (from MLflow)
- Competition metadata (leaderboard, deadline, metric)

**Outputs**:
- Current performance summary
- CV vs LB gap analysis
- Feature importance trends
- Model performance comparison
- Bottleneck identification

**Example output**:
```json
{
  "current_best": {
    "model": "xgb_optuna_v1",
    "cv": 0.345,
    "lb": 0.464,
    "gap": 0.119
  },
  "insights": [
    "Large CV-LB gap suggests overfitting on validation set",
    "XGBoost consistently outperforms LightGBM by 0.02 RMSLE",
    "Rolling features (7/14/30 day) contribute 60% of importance",
    "Holiday features have minimal impact (<2%)"
  ],
  "bottlenecks": [
    "Test set has different temporal distribution than train",
    "Lag features using mean imputation may be too crude"
  ]
}
```

### 2. Strategist Agent (Replaces rigid Planner)
**Purpose**: Decide what to try next based on analysis

**Inputs**:
- Analyst output
- Past experiment history
- Available compute budget
- Remaining submissions

**Outputs**:
- Prioritized list of experiments (not fixed phases!)
- Expected impact estimate
- Resource requirements
- Success criteria

**Example output**:
```json
{
  "experiments": [
    {
      "id": "exp_001",
      "name": "Improve test set lag features",
      "hypothesis": "Using last 30-day stats instead of single lag will reduce CV-LB gap",
      "approach": "Modify test feature engineering to use rolling stats from train tail",
      "expected_impact": "+0.02 to +0.05 LB improvement",
      "effort": "30 min",
      "priority": 1,
      "success_criteria": "CV-LB gap < 0.10"
    },
    {
      "id": "exp_002",
      "name": "Remove low-impact features",
      "hypothesis": "Dropping 15 weak features will reduce overfitting",
      "approach": "Train with top 12 features only (from importance analysis)",
      "expected_impact": "+0.01 to +0.03 LB improvement",
      "effort": "15 min",
      "priority": 2,
      "success_criteria": "CV stays <0.350, LB improves"
    }
  ]
}
```

### 3. Engineer Agent (Enhanced Developer)
**Purpose**: Implement experiments with learning from past failures

**Inputs**:
- Experiment spec from Strategist
- Codebase context
- Past implementation errors (from logs)

**Outputs**:
- Implemented code
- Unit tests
- Validation results

**New capability**: Maintains error memory
- Remembers: "Last time I added external data, test set dates didn't align"
- Checks: Validate date ranges before running
- Learns: "Always check data coverage before feature engineering"

### 4. Evaluator Agent (Replaces simple Reviewer)
**Purpose**: Assess if experiment met success criteria

**Inputs**:
- Experiment results (CV, LB if submitted)
- Success criteria from Strategist
- Past experiment context

**Outputs**:
- Success/failure assessment
- Root cause analysis if failed
- Actionable next steps

**Example output**:
```json
{
  "experiment_id": "exp_001",
  "success": false,
  "results": {
    "cv": 0.348,
    "lb": 0.455,
    "gap": 0.107
  },
  "assessment": "CV improved slightly but gap still >0.10. Hypothesis partially validated.",
  "root_cause": "Rolling stats reduced variance but didn't address distribution shift",
  "next_steps": [
    "Try time-based validation split that matches test set period",
    "Add temporal features (year/month) to capture distribution shift"
  ],
  "learned": "Last 30-day stats help but insufficient for large temporal gaps"
}
```

### 5. Curator Agent (NEW)
**Purpose**: Build institutional knowledge from experiments

**Inputs**:
- All experiment results
- Evaluator assessments
- Code changes

**Outputs**:
- Updated memory (Postgres + Qdrant)
- Pattern library (what works for this competition type)
- Competition-specific insights (stored in CLAUDE.md)

**Example actions**:
- Stores: "For time series with distribution shift, use time-based CV splits"
- Updates: Competition CLAUDE.md with learned best practices
- Embeds: Successful feature engineering patterns to Qdrant
- Tags: Experiments by success/failure/partial for future reference

## Workflow Loop (Dynamic, not Phase-based)

```
1. ANALYZE
   Analyst: "What's our current state? What's working?"
   
2. STRATEGIZE
   Strategist: "Based on analysis, what should we try next?"
   
3. IMPLEMENT
   Engineer: "Execute the highest priority experiment"
   
4. EVALUATE
   Evaluator: "Did it work? Why or why not?"
   
5. LEARN
   Curator: "Store what we learned for future reference"
   
6. REPEAT
   Loop back to ANALYZE with updated knowledge
```

## Key Differences from Current System

| Current | Learning Multi-Agent |
|---------|---------------------|
| Fixed 9 phases (0-8) | Dynamic experiment queue |
| No memory between runs | Every agent reads past work |
| Single pass through workflow | Iterative improvement loop |
| No root cause analysis | Evaluator does RCA on every experiment |
| No learning capture | Curator builds institutional knowledge |
| Phase-based progression | Impact-based prioritization |

## Implementation Without LangChain

**We don't need LangChain** because we already have:

✅ **Agent communication**: JSON messages between agents
✅ **Memory**: Postgres (structured) + Qdrant (semantic)
✅ **LLM**: Ollama for agent reasoning
✅ **Observability**: Langfuse for tracing
✅ **Orchestration**: Python scripts with clear handoffs

**What we build**:
```python
# agents/analyst.py
@observe(name="analyst_analyze")
def analyze_state(competition: str) -> dict:
    # Query past experiments from MLflow
    # Query submissions from Postgres
    # Use Ollama to synthesize insights
    # Return structured analysis

# agents/strategist.py
@observe(name="strategist_plan")
def plan_experiments(analysis: dict) -> list[dict]:
    # Use Ollama to generate experiment ideas
    # Query Qdrant for similar past approaches
    # Prioritize by expected impact
    # Return ranked experiment queue

# agents/engineer.py
@observe(name="engineer_implement")
def implement_experiment(spec: dict) -> dict:
    # Generate code with Ollama
    # Validate against past errors
    # Run with retry logic
    # Return results

# agents/evaluator.py
@observe(name="evaluator_assess")
def evaluate_results(experiment: dict) -> dict:
    # Compare to success criteria
    # Use Ollama for root cause analysis
    # Generate next steps
    # Return assessment

# agents/curator.py
@observe(name="curator_learn")
def curate_knowledge(assessment: dict) -> None:
    # Store successful patterns to Qdrant
    # Update competition CLAUDE.md
    # Tag experiment in MLflow
    # Record learnings to Postgres
```

## Autonomy Benefits

**Why this is better for autonomy**:

1. **Self-directed**: Strategist decides what to try based on data, not human-defined phases
2. **Self-improving**: Each iteration adds to knowledge base
3. **Self-correcting**: Evaluator identifies failures and proposes fixes
4. **Context-aware**: Agents read past work before acting
5. **Adaptive**: No rigid workflow - adjusts based on what's working

**Example autonomous session**:
```
Hour 1: Analyst identifies CV-LB gap, Strategist proposes 3 experiments
Hour 2: Engineer implements top priority, gap reduces 0.119 → 0.107
Hour 3: Evaluator sees partial success, Strategist refines approach
Hour 4: Engineer tries refined version, gap reduces to 0.095
Hour 5: Curator stores "time-based CV split" as winning pattern
Hour 6: Strategist uses pattern to plan next competition phase
```

**Human only needed for**:
- Final submission approval
- Major strategy pivots
- Resolving contradictions
- Defining compute budget

## Next Steps to Build This

1. **Implement 5 new agents** (Analyst, Strategist, Engineer, Evaluator, Curator)
2. **Create orchestration loop** (analyze → strategize → implement → evaluate → learn → repeat)
3. **Add memory queries** to each agent (read before act)
4. **Build experiment queue system** (priority-based, not phase-based)
5. **Track everything in Langfuse** (observe all agent decisions)
6. **Test on store-sales** (let it run autonomously for 6 hours)
7. **Evaluate vs single-agent** (did it find better approaches?)

## Success Metrics

After 10 autonomous experiments:
- ✅ Found 2+ approaches I wouldn't have tried manually
- ✅ CV-LB gap reduced by iterative refinement
- ✅ Knowledge base has 5+ reusable patterns
- ✅ Next competition starts with learned strategies
- ✅ Langfuse trace shows clear decision chain

---

**Decision**: Build this instead of using LangChain. We have all the primitives, just need to wire them together with a learning loop.
