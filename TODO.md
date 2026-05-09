# Kaggle Agent TODO List

## High Priority

- [x] **DeepEval Integration** ✅ COMPLETE
  - Built-in evaluation with RTX 5070 + Ollama
  - LLM-based experiment validation (8-18s per eval)
  - Pre-submission workflow integration
  - /eval skill created
  - Frontend dashboard at /kaggle/deepeval
  
- [x] **Eval Platform GUI** ✅ COMPLETE
  - Frontend dashboard for evaluation results
  - Stats cards, eval history table
  - Pass/fail indicators, issues/warnings
  - Deployed via Vercel

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
  - [x] `/eval` - Run DeepEval metrics on experiment ✅ COMPLETE
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
