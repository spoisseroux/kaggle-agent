---
name: Research competitions before optimizing
description: Always research competition meta-game first to avoid wasting time on gameable/tutorial competitions
type: feedback
---

ALWAYS do Phase 0 (RESEARCH) before any technical work on a Kaggle competition.

**Why:** Titanic incident - spent time optimizing ML models to 0.787 without realizing:
- Hundreds of perfect 1.0 scores on leaderboard
- Test set is based on real historical data (Encyclopedia Titanica)
- Can get 100% by matching test passenger names to historical survivor records
- Not a real ML problem, just a matching exercise

**How to apply:**
1. Run `./scripts/research_competition.sh <slug>` FIRST (15-30 min)
2. Check for red flags:
   - Many perfect scores (1.0) on leaderboard
   - Tiny test set (<500 samples)
   - "Getting started" or tutorial competition label
   - Historical/public data where answers exist
3. If red flags present: use the gameable approach or skip to real competitions
4. Only invest in feature engineering / model optimization if competition is legitimate

**Source:** User feedback after Titanic attempt, 2026-05-07
