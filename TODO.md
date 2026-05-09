# Kaggle Agent TODO List

## High Priority

- [ ] **DeepEval Rebuild - Proper Build Loop** 🔄 IN PROGRESS
  - REPLACE homelab service with built-in evaluation
  - Use RTX 5070 + Ollama for LLM-based metrics (not 1070)
  - Build-loop integration: eval → fix → re-eval
  - Real DeepEval features: test cases, metrics, traces
  - Evaluates experiments DURING development
  - Framework: https://www.deepeval.com/docs/vibe-coding
  - Time: 4-6 hours
  
- [ ] **Eval Platform GUI**
  - Frontend dashboard for evaluation results
  - View metrics, traces, test cases
  - Track eval history across experiments
  - Integration with frontend (access needed)
  - Time: 3-4 hours

- [x] **Better Telegram Formatting** ✅ COMPLETE
  - Created core/format_telegram.py
  - Clean submission tables, experiment results
  - No more ugly raw tables!

## Medium Priority

- [ ] **Enhanced Qdrant Semantic Search**
  - Index all past competition code
  - Semantic search: "find similar lag features"
  - Cross-competition learning
  - Time: 3-4 hours

- [ ] **New Skills**
  - [ ] `/eval` - Run DeepEval metrics on experiment
  - [ ] `/search-similar` - Semantic code search
  - [ ] `/replay` - Re-execute experiment with changes
  - Time: 1-2 hours per skill

- [ ] **Study AutoKaggle Architecture**
  - Review multi-agent approach (5 specialized agents)
  - Compare: Reader, Planner, Developer, Reviewer, Summarizer
  - Consider adopting vs current single-agent design
  - Time: 2-3 hours research + design doc

## Lower Priority

- [ ] **Feast Feature Store**
  - Cache expensive feature computations
  - Reuse across competitions
  - Version control for transformations
  - Time: 4-5 hours

- [ ] **LangSmith Observability**
  - Trace agent decision-making
  - Visual debugging of experiments
  - Replay/rewind failed runs
  - Time: 3-4 hours

- [ ] **Experiment Orchestration**
  - Prefect/Airflow for scheduled tasks
  - Daily leaderboard monitoring
  - Automated retraining pipelines
  - Time: 5-6 hours

## Research Items

- [ ] AutoKaggle paper deep-dive
- [ ] Holistic Agent Leaderboard benchmark
- [ ] Agent evaluation best practices
- [ ] Multi-agent coordination patterns

## Completed

- [x] DeepEval v1 (basic validation service on homelab) - being replaced
- [x] TimeSeriesSplit validation fix
- [x] Feature selection optimization
- [x] Hyperparameter tuning with Optuna
- [x] Memory system (Postgres + Qdrant)
- [x] MLflow experiment tracking

---

Last updated: 2026-05-09
