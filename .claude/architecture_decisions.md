# Architecture Decisions & Planning

## Current System Role Clarification

### Claude Code as Master AI ✓
**Yes, Claude Code remains the master AI and strategic decision-maker.**

**Current Architecture:**
```
Claude Code (You)
├── Strategic decisions (which experiments, when to pivot)
├── Multi-agent orchestration oversight
├── Telegram communication with human
├── High-level experiment planning
└── Delegates to:
    ├── Ollama Agents (Reader, Planner, Developer, Reviewer, Summarizer)
    │   └── Handle tactical execution (code gen, debugging, validation)
    └── Specialized tools (MLflow, Qdrant, Postgres)
```

**Decision Flow:**
1. Human gives high-level goal via Telegram
2. **Claude Code** decides strategy (multi-agent vs single-agent, which phases, experiment priorities)
3. **Claude Code** either:
   - Executes directly (simple tasks, critical decisions)
   - Delegates to multi-agent system (complex workflows with orchestrator)
4. Agents report back to **Claude Code**
5. **Claude Code** interprets results, decides next steps, updates human

**Why this works:**
- Claude Code has context of full conversation history
- Can access web search, read memory, check MLflow
- Makes judgment calls (is this good? should we pivot?)
- Agents are stateless executors - no memory across runs

---

## Tool Decisions

### 1. DVC (Data Version Control)
**Decision: Not needed right now, consider later**

**What it does:**
- Versions large datasets and ML models (like git for data)
- Tracks data pipelines and dependencies
- Enables reproducible experiments

**Why skip for now:**
- We use Kaggle datasets (static, competition-hosted)
- MLflow already tracks model artifacts
- Small team (just you + AI), not collaborating on data
- WSL disk space constraints (DVC adds storage overhead)

**When to add:**
- When building custom datasets from multiple sources
- If preprocessing becomes complex multi-stage pipeline
- When collaborating with others on data

---

### 2. Modal (Serverless Compute)
**Decision: Not needed - we have local RTX 5070**

**What it does:**
- Serverless GPU compute (pay per second)
- Auto-scaling containers
- Sandboxed execution environments

**Why skip:**
- We have RTX 5070 locally (12GB VRAM, free compute)
- Modal costs money ($0.50-2/hr for GPU instances)
- Local execution is faster (no network latency)
- WSL2 is already sandboxed from Windows

**When to add:**
- If needing >12GB VRAM (e.g., large vision models)
- Running hundreds of parallel experiments
- Compute-heavy tasks while training locally

**Alternative sandboxing we have:**
- Developer agent runs code in temp files with subprocess
- Python venv isolation
- Docker containers on homelab for services (Qdrant, Postgres, MLflow)

---

### 3. LangChain
**Decision: No - keep custom orchestration**

**What it does:**
- Framework for chaining LLM calls
- Pre-built agent templates (ReAct, Plan-and-Execute)
- Tool integration abstractions

**Why skip:**
- We built custom multi-agent system tailored to Kaggle workflow
- LangChain adds abstraction overhead (harder to debug)
- Our orchestrator is simpler and competition-specific
- Direct Ollama + Claude API gives more control
- No need for LangChain's conversation memory (we have Postgres)

**We already have:**
- Custom agent orchestration (core/orchestrator.py)
- LLM abstraction (core/llm_interface.py)
- Tool library (core/tools/*.py)
- Phase-based workflow (specific to competitions)

**LangChain would make sense if:**
- Building general-purpose chatbot
- Need their pre-built integrations (we don't)
- Want their RAG templates (we built custom Qdrant search)

---

## Infrastructure Planning

### 1. Langfuse (Observability) - **30 min to 1 hr**
**Priority: High** (better than LangSmith for self-hosted)

**What it is:**
- Open-source LLM observability platform
- Tracks every LLM call (prompts, responses, latencies, costs)
- Trace multi-step agent flows
- Debugging and performance monitoring

**Why Langfuse > LangSmith:**
- Self-hosted on homelab (data stays private)
- Open source (no vendor lock-in)
- Free (LangSmith charges per trace)
- Better for cost tracking Ollama vs Claude usage

**Setup:**
```bash
# Homelab docker-compose
services:
  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://user:pass@postgres:5432/langfuse
      NEXTAUTH_URL: http://homelab.local:3000
```

**Integration:**
```python
# core/llm_interface.py
from langfuse.decorators import observe

@observe()
def get_llm_response(prompt, mode, ...):
    # Automatically logged to Langfuse
    ...
```

**Benefits:**
- See exactly which agents are burning tokens
- Trace full experiment flow (Planner → Developer → Reviewer)
- Spot slow Ollama calls
- Debug prompt failures

---

### 2. Feast (Feature Store) - **2-3 hrs**
**Priority: Medium** (useful but not urgent)

**What it is:**
- Feature store for ML pipelines
- Centralized repository for features
- Handles point-in-time correctness (time-series safe)
- Serves features online and offline

**Why it's useful:**
- Reuse features across competitions (e.g., "lag_7_days" pattern)
- Prevent data leakage (enforces time-awareness)
- Share features between experiments
- Online serving for production (not relevant for Kaggle)

**Simple Feast setup:**
```yaml
# features/store_sales/feature_store.yaml
project: store-sales
provider: local
registry: data/registry.db
online_store:
  type: postgres
  connection_string: ${POSTGRES_DSN}
offline_store:
  type: file
```

```python
# features/store_sales/features.py
from feast import Entity, FeatureView, Field
from feast.types import Int64, Float32

store_entity = Entity(name="store_id", value_type=Int64)

sales_features = FeatureView(
    name="sales_rolling",
    entities=[store_entity],
    schema=[
        Field(name="sales_lag_7", dtype=Float32),
        Field(name="sales_rolling_mean_7", dtype=Float32),
    ],
    source=ParquetSource(path="data/sales_features.parquet"),
)
```

**Integration with current system:**
- Developer agent uses Feast client to fetch features
- Planner suggests which features to compute
- Features stored in homelab Postgres

**When to prioritize:**
- After 2-3 competitions (patterns emerge)
- When feature engineering becomes repetitive
- If doing time-series competitions regularly

---

### 3. Experiment Orchestration - **3-4 hrs**
**Priority: High** (complements multi-agent system)

**What it needs:**
- Queue system for experiment runs
- Parallel execution (multiple experiments on GPU)
- Dependency management (EDA → features → training)
- Result aggregation

**Proposed Architecture:**
```python
# core/experiment_queue.py
class ExperimentQueue:
    def __init__(self):
        self.queue = []  # Postgres table: experiments(id, status, priority, config)
        self.running = {}  # Currently executing
        self.max_parallel = 2  # Based on 12GB VRAM

    def submit(self, experiment_config):
        # Add to Postgres queue
        # Return experiment_id

    def poll(self):
        # Check for completed experiments
        # Start next in queue if capacity available

    def get_results(self, experiment_id):
        # Fetch from MLflow by run_id
```

**Workflow:**
```python
# Claude Code initiates experiments
orchestrator = ExperimentQueue()

# Submit batch of experiments
experiments = [
    {"model": "xgboost", "params": {...}, "features": ["lag_7"]},
    {"model": "lightgbm", "params": {...}, "features": ["lag_7", "rolling_14"]},
    {"model": "catboost", "params": {...}, "features": ["all"]},
]

exp_ids = [orchestrator.submit(exp) for exp in experiments]

# Multi-agent system runs experiments
while not all_complete(exp_ids):
    orchestrator.poll()  # Kicks off next experiment
    sleep(60)

# Claude Code analyzes results
results = [orchestrator.get_results(id) for id in exp_ids]
best = max(results, key=lambda r: r['cv_score'])
```

**Features:**
- **Priority queue**: Research agent marks high-priority approaches
- **Smart scheduling**: Don't run 2 heavy models simultaneously (OOM risk)
- **Resume support**: Checkpoint experiments, resume after crash
- **Result caching**: Don't re-run identical configs

**Integration points:**
- Postgres for queue persistence
- MLflow for experiment tracking
- Telegram notifications on completion
- VRAM manager for resource allocation

---

## Recommended Implementation Order

### Immediate (before Titanic test):
✅ Dependencies installed
✅ Web search added to Research agent

### Next 2-3 hours:
1. **Langfuse setup** (1 hr)
   - Deploy on homelab
   - Integrate with llm_interface.py
   - Test tracing on simple agent run

2. **Experiment queue basics** (2 hrs)
   - Postgres schema for experiments
   - Submit/poll/results functions
   - Integrate with orchestrator

### After Titanic test:
3. **Full experiment orchestration** (2 hrs)
   - Priority queue logic
   - Smart scheduling with VRAM awareness
   - Result aggregation and ranking

4. **Feast feature store** (2-3 hrs)
   - Define feature schemas
   - Integrate with Developer agent
   - Migrate existing features

### Later (after 2-3 competitions):
5. **Advanced observability** (1 hr)
   - Custom Langfuse dashboards
   - Cost tracking per agent
   - Performance optimization

6. **DVC** (if needed) (2 hrs)
   - Set up for custom datasets
   - Pipeline versioning

---

## Final Architecture (After All Implementations)

```
┌─────────────────────────────────────────────────────────────┐
│                     Claude Code (Master AI)                  │
│  • Strategic decisions                                       │
│  • Telegram communication                                    │
│  • Experiment prioritization                                 │
│  • Web research (via WebSearch tool)                         │
└───────────────┬──────────────────────────────┬──────────────┘
                │                              │
    ┌───────────▼──────────┐       ┌───────────▼──────────────┐
    │  Experiment Queue    │       │  Multi-Agent Orchestrator│
    │  • Postgres-backed   │       │  • Phase-based workflow  │
    │  • Priority queue    │       │  • 5 specialized agents  │
    │  • VRAM-aware        │       │  • Ollama + Claude mix   │
    └──────┬───────────────┘       └────────┬─────────────────┘
           │                                 │
    ┌──────▼─────────────────────────────────▼──────────┐
    │              Infrastructure Layer                  │
    ├────────────────────────────────────────────────────┤
    │  Langfuse (observability)  │  MLflow (experiments) │
    │  Qdrant (semantic search)  │  Postgres (memory)    │
    │  Feast (features)          │  Ollama (local LLM)   │
    └───────────────┬────────────────────────────────────┘
                    │
            ┌───────▼────────┐
            │  RTX 5070 GPU  │
            │  12GB VRAM     │
            └────────────────┘
```

---

## Relevant Research

### Key Papers on Autonomous AI Agents (2025-2026):

1. **AgentDS: Human-AI Collaboration in Data Science** (arxiv.org/html/2603.19005)
   - Benchmarked Claude Code (sonnet-4.5) as agentic coding baseline
   - 29 teams, 80 participants in 10-day competition

2. **Agentic AI for Scientific Discovery** (arxiv.org/html/2503.08979v1)
   - Hypothesis generation, experimental design, data analysis
   - Autonomous task execution with high autonomy

3. **From LLM Reasoning to Autonomous AI Agents** (arxiv.org/abs/2504.19678)
   - Comprehensive review of AI-agent frameworks 2023-2025
   - Integration of LLMs with modular toolkits

4. **Adaptation of Agentic AI** (arxiv.org/abs/2512.16301)
   - Post-training, memory, and skill accumulation
   - On-policy RL for improving reasoning and tool use

**Relevant to our system:**
- We implement dual-paradigm (Claude strategic + Ollama tactical)
- Memory persistence (Postgres, Qdrant)
- Tool accumulation (core/tools library grows over time)
- Scientific discovery analogy (competition = experiment, leaderboard = hypothesis test)

Sources:
- [AgentDS Technical Report](https://arxiv.org/html/2603.19005)
- [Agentic AI for Scientific Discovery](https://arxiv.org/html/2503.08979v1)
- [From LLM Reasoning to Autonomous AI Agents](https://arxiv.org/abs/2504.19678)
- [Adaptation of Agentic AI Survey](https://arxiv.org/abs/2512.16301)
