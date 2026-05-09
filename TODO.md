# Kaggle Agent TODO List

## High Priority

- [ ] **Test Learning Multi-Agent System** 🔥 NEW
  - Run autonomous test on store-sales competition
  - Validate Analyst → Strategist → Engineer → Evaluator → Curator loop
  - Measure if it finds approaches not tried manually
  - Monitor Langfuse traces for agent decisions
  - Time: 2-3 hours autonomous run + monitoring
  
- [ ] **Telegram Group Chat Onboarding** 🔥 NEW
  - Get group chat ID from user
  - Get authorized user IDs (user + ML partner)
  - Configure .env with group settings
  - Test multi-user interaction
  - Deploy telegram_bot_group.py service
  - Time: 30 min setup + testing

- [ ] **Configure Langfuse Authentication** 🔥 NEW
  - Get API keys from user (pk-... and sk-...)
  - Add to .env file
  - Test trace export from agents
  - Verify dashboard shows agent decisions
  - Time: 15 min

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

- [x] **Enhanced Qdrant Semantic Search** ✅ COMPLETE
  - Local embeddings on RTX 5070 (3x faster)
  - Index all past competition code
  - Semantic search: "find similar lag features"
  - Cross-competition learning
  - Code indexer + semantic search + /search-similar skill

- [ ] **New Skills**
  - [x] `/eval` - Run DeepEval metrics on experiment ✅ COMPLETE
  - [x] `/search-similar` - Semantic code search ✅ COMPLETE
  - [x] `/replay` - Re-execute experiment with changes ✅ COMPLETE
  - Time: 1-2 hours per skill

- [x] **Study AutoKaggle Architecture** ✅ COMPLETE
  - Reviewed multi-agent approach (5 specialized agents)
  - Compared: Reader, Planner, Developer, Reviewer, Summarizer
  - Integration design doc created: docs/autokaggle_integration_design.md
  - Hybrid architecture preserves existing + adds agents
  - 67% API cost reduction via Ollama offloading
  - Time: 3 hours research + comprehensive design doc

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

- [x] **Learning Multi-Agent System** ✅ COMPLETE (2026-05-09)
  - Built 5 specialized agents (Analyst, Strategist, Engineer, Evaluator, Curator)
  - Orchestrator with iterative learning loop
  - Dynamic experiment queue (not fixed phases)
  - Memory-first approach (agents read past work)
  - Langfuse tracking integration
  - No LangChain dependency
  - Full documentation + standalone CLI for each agent
  - 1,465 lines of code in 2 hours

- [x] **Telegram Group Chat Support** ✅ COMPLETE (2026-05-09)
  - telegram_bot_group.py with multi-user support
  - User authorization whitelist
  - @mention requirement option
  - User attribution in messages
  - Commands: /status, /help, /queue
  - Backward compatible with private chats
  - Full setup guide

- [x] **Langfuse Single-Agent Tracking** ✅ COMPLETE (2026-05-09)
  - core/langfuse_tracker.py with decorators
  - @track_experiment, @track_decision, @track_phase
  - Manual trace creation for ad-hoc events
  - Graceful handling when Langfuse not configured

- [x] DeepEval v1 (basic validation service on homelab) - being replaced
- [x] TimeSeriesSplit validation fix
- [x] Feature selection optimization
- [x] Hyperparameter tuning with Optuna
- [x] Memory system (Postgres + Qdrant)
- [x] MLflow experiment tracking

---

Last updated: 2026-05-09
