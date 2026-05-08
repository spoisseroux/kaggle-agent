# Autonomous 24/7 Operation Design

## Goal
Run continuous experiment loops without:
- Melting the PC (VRAM/CPU management)
- Burning Claude API credits (aggressive Ollama usage)
- User losing visibility (progress tracking)
- Runaway costs (hard limits + alerts)

## Resource Guardrails

### 1. API Usage Limits (Claude Code)
**Problem:** Claude Opus 4.6 is expensive. 24/7 usage could drain credits fast.

**Solution: Ollama-First Architecture + Smart Token Budget**
```python
# 95% of code generation → Ollama (qwen3:14b, FREE, local)
from core.ollama_client import generate

code = generate(
    "Write a LightGBM training loop with 5-fold CV",
    system="You are a Kaggle ML engineer. Return only code, no explanation."
)

# 5% strategic decisions → Claude (this conversation)
# But with smart budgeting:
from core.api_budget import check_and_record

allowed, reason = check_and_record(5000, operation_type="strategic")
if not allowed:
    # Fall back to Ollama or wait
    use_ollama_instead()
```

**Smart Token Budget (not simple rate limits!):**
- Hourly: 200K tokens (~40 strategic calls or 20 code reviews)
- Daily: 3M tokens (~$18/day at current pricing)
- **Priority system**: Strategic = 0.5x cost, Docs/Routine = 2x cost
- Soft warnings at 70% usage, hard limits at 90%
- Auto-switches to Ollama-only mode when budget tight
- Tracks actual token usage, not just call counts
- Implemented in: `core/api_budget.py` ✅

### 2. GPU/VRAM Management
**Problem:** RTX 5070 has limited VRAM. Back-to-back training can overheat.

**Solution: Already Built!**
```python
from core.vram_manager import request_training_vram, release_training_vram

# Before training
request_training_vram()  # Clears cache, waits for availability
train_model()
release_training_vram()  # Explicit cleanup

# Between experiments: 5 min cooldown
time.sleep(300)
```

**Monitoring:**
- `core/gpu_monitor.py` tracks temp/usage
- Auto-pause if GPU temp > 85°C
- Alert via Telegram if sustained high temps

### 3. Experiment Loop Rate Limiting
**Problem:** Running too many experiments wastes compute on bad ideas.

**Solution: Flexible, Adaptive Pacing**

Full config in: `config/experiment_limits.yaml` ✅

**Base Limits (by competition type):**
- **Playground**: 100 experiments/day (for practice)
- **Featured**: 40/day (high-stakes competitions)
- **Research**: 60/day (balanced)
- **Default**: 50/day (was 20, increased!)

**Adaptive Scaling:**
- Near deadline: +50% (7 days), +100% (2 days), +200% (12 hours)
- Hot streak (3+ improvements in a row): +50% temporarily
- API budget critical: -70% (scale back to preserve budget)
- GPU overheating: -50% (prevent damage)

**Smart Stopping:**
- Stop after 8 experiments with no improvement (was 5)
- Min interval: 5 min (was 15, faster iteration)
- Emergency stop: 5 consecutive errors, >50% error rate, thermal throttling

This means you can do quick iteration when things work, but automatically pause when stuck!

## Progress Visibility

### Real-Time Notifications (Telegram)
**Send updates for:**
- Phase transitions: "🔧 Starting feature engineering phase"
- Experiment complete: "✓ XGBoost CV: 0.8234 (+0.0012 vs baseline)"
- Milestones: "🎯 New best CV: 0.8456!"
- Blockers: "⚠️ Stuck after 5 experiments, no CV gain. Pausing for input."

**Frequency:**
- Every experiment completion (~15-30 min)
- Hourly summary if long-running
- Immediate alerts on errors/limits

### Dashboard (Web UI - Future Enhancement)
```
Current Competition: house-prices-advanced
Status: Training XGBoost (8/20 daily experiments used)
Best CV: 0.8456 | Best LB: 0.8401 | Rank: ~top 15%

Last 5 Experiments:
  [10:23] XGBoost + FE_v3 → CV: 0.8456 ✓ NEW BEST
  [09:54] LightGBM + FE_v2 → CV: 0.8421
  [09:31] CatBoost baseline → CV: 0.8312

API Usage: 3/10 Claude calls this hour | GPU: 67°C, 8.2GB VRAM
```

### MLflow UI (Already Available)
- Run: `mlflow ui --port 5000`
- View all experiments, metrics, params
- Already integrated, just needs to be accessible

## Submission Strategy

### When to Submit
```python
def should_submit(cv_score, lb_score_history):
    # Only submit if:
    # 1. New CV best by >0.5%
    if cv_score < best_cv * 1.005:
        return False

    # 2. Haven't submitted in last 4 hours
    if time_since_last_submit < 4 * 3600:
        return False

    # 3. Have submissions remaining (save 3 for final day)
    days_left = (deadline - now).days
    if days_left < 1 and submissions_left < 3:
        return False

    return True
```

**Hard Rule:** Still require human approval for first submission per competition.
After that, auto-submit within guardrails IF user has set `auto_submit: true` in competition config.

## Competition Selection (Periodic Sweeps)

### Daily Research Cron Job
```bash
# At 6 AM daily
crontab -e
0 6 * * * cd /home/keehar/kaggle-agent && ./scripts/research_new_competitions.sh
```

**Script logic:**
1. Fetch all active competitions from Kaggle API
2. Filter by criteria:
   - Deadline 7-60 days out
   - Prize pool $5K-$50K (sweet spot)
   - Category: Playground, Research, Featured
   - Not "Getting Started" (too easy/gameable)
3. For top 3 matches:
   - Run research phase (leaderboard, discussions, notebooks)
   - Score by: learning value, time investment, likelihood of top 20%
4. Send Telegram summary: "📋 Found 2 new interesting competitions"
5. Wait for human approval before starting

## Example 24/7 Workflow

```
Day 1, 08:00: User says "start house-prices-advanced"
Day 1, 08:15: Phase 0 complete (research → green flag)
Day 1, 08:30: Phase 1 (EDA) → Phase 2 (data pipeline)
Day 1, 10:00: Phase 3 (baseline) → CV: 0.8012
Day 1, 11:30: Experiment 2 (FE_v1) → CV: 0.8234 ✓ +0.0222
Day 1, 13:00: Experiment 3 (FE_v2) → CV: 0.8256 ✓ +0.0022
Day 1, 13:05: Auto-submit (first submission, human approved in config)
Day 1, 15:30: Experiment 4 (XGBoost tuning) → CV: 0.8401 ✓ +0.0145
Day 1, 17:00: Experiment 5 (ensemble) → CV: 0.8389 ✗ -0.0012
...
[Continuous loop, ~20 experiments/day, notifications every completion]
...
Day 7, 14:00: No improvement in 5 experiments → pause, notify
Day 7, 14:15: User replies "try neural net approach"
Day 7, 14:20: Resume with new direction
```

## Cost Estimates

**With Ollama-First:**
- Code generation: 95% Ollama (FREE)
- Strategic decisions: ~10 Claude calls/hour × 24h = 240/day
- Assuming $0.01/call (rough estimate) = ~$2.40/day = $72/month

**Without Ollama (all Claude):**
- ~100 Claude calls/hour × 24h = 2400/day
- Cost: ~$24/day = $720/month

**Savings: 90% reduction by using Ollama aggressively**

## Implementation Status

### ✅ Already Built
- Ollama integration (`core/ollama_client.py`)
- VRAM manager (`core/vram_manager.py`)
- GPU monitor (`core/gpu_monitor.py`)
- Telegram notifications (`core/notify.py`)
- Memory system (Postgres + Qdrant)
- MLflow tracking

### 🔨 Need to Create
- ✅ `core/api_budget.py` - Smart token-based budget system (DONE)
- ✅ `config/experiment_limits.yaml` - Flexible pacing config (DONE)
- ⏳ `core/experiment_pacer.py` - Enforcement engine for limits
- ⏳ `scripts/research_new_competitions.sh` - Daily competition sweeps
- ⏳ `scripts/autonomous_loop.py` - Main 24/7 orchestrator
- ⏳ Competition config schema with `auto_submit`, type-specific overrides

### 🎯 Next Steps
1. Choose a Playground competition to test with
2. Create the missing components above
3. Run first 24-hour test with human monitoring
4. Tune the guardrails based on results
5. Full autonomous mode
