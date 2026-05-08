# Store Sales - Phase 0 Research Complete

**Date:** 2026-05-08
**Competition:** store-sales-time-series-forecasting
**Status:** ✅ GREEN FLAG - Worth optimizing

## Assessment: LEGITIMATE COMPETITION ✅

### Red Flag Check
- ❌ Perfect scores? NO (leaderboard at 0.377-0.379 RMSLE)
- ❌ Tiny test set? NO (28.5K predictions needed)
- ❌ Gameable? NO (large training set, realistic scores)
- ❌ Tutorial/trivial? NO (time series, multi-dimensional problem)
- ✅ Good for learning? YES (new problem type, good data quality)

**Verdict:** Proceed with full autonomous workflow

## Competition Overview

**Prediction Task:**
- Forecast daily sales for 33 product families across 54 stores
- Train period: 2013-01-01 to 2017-08-15 (~4.7 years)
- Test period: 2017-08-16 to 2017-08-31 (16 days)
- Metric: RMSLE (lower is better, leaderboard ~0.377)

**Data Structure:**
- **train.csv**: 3,000,888 rows (store × family × date × sales)
- **test.csv**: 28,512 rows (predictions needed)
- **stores.csv**: 54 stores (type A-E, 17 clusters)
- **holidays_events.csv**: 350 special events/holidays
- **oil.csv**: 1,218 daily oil prices
- **transactions.csv**: 83,488 daily transaction counts

**Key Features:**
- `store_nbr`: 1-54 (store identifier)
- `family`: 33 product categories (AUTOMOTIVE, BEVERAGES, GROCERY, etc.)
- `sales`: Target variable (what we predict)
- `onpromotion`: Number of items on promotion
- `date`: Daily granularity
- Store metadata: city, state, type, cluster
- External: holidays, oil prices, transaction counts

## Competition Type & Difficulty

**Type:** Time series forecasting (regression)
**Complexity:** Medium
- Multiple stores (54)
- Multiple product families (33)
- Time series component (4.7 years history)
- External features (holidays, oil, promotions)

**Differs from Titanic:**
- Time series instead of classification
- Much larger dataset (3M vs 891 rows)
- Multi-dimensional (store × family × time)
- Requires temporal feature engineering

## Leaderboard Analysis

**Top scores:** 0.377-0.379 RMSLE
**Participants:** 936 teams
**Score distribution:** Normal, no clustering at perfect scores
**Recent activity:** Active submissions in April-May 2026

**Benchmark:** 0.377 RMSLE (current leader)
**Goal:** <0.40 RMSLE for top 50%, <0.38 for top 25%

## Common Winning Approaches (Expected)

Based on competition structure:

1. **Feature Engineering:**
   - Lag features (sales from prior days/weeks/months)
   - Moving averages (7-day, 30-day MA)
   - Holiday proximity (days to/from holiday)
   - Seasonality (day-of-week, month, year effects)
   - Store cluster indicators
   - Promotion impact features

2. **Models to Try:**
   - LightGBM/XGBoost (gradient boosting with temporal features)
   - Prophet (Facebook's time series library)
   - LSTM/GRU (if we want deep learning)
   - Ensemble of above

3. **Cross-validation Strategy:**
   - Time-based CV (not random k-fold!)
   - Walk-forward validation
   - Leave last N days for validation

4. **External Data:**
   - Oil price correlation
   - Holiday impact analysis
   - Transaction count trends

## Data Quality

**Pros:**
- No missing values in training sales
- Complete date coverage
- Clean store/family identifiers

**Cons:**
- Short test period (16 days only)
- One missing oil price value (2013-01-01)
- Sparse holiday coverage (~350 events over 4.7 years)

## Next Steps (Autonomous Workflow)

**Phase 1:** EDA
- Visualize sales trends by store/family
- Identify seasonality patterns
- Analyze promotion impact
- Correlation analysis (oil prices, holidays, transactions)

**Phase 2:** Data Pipeline
- Create lag features (1, 7, 14, 30 days)
- Moving averages
- Holiday proximity features
- Day-of-week, month encoding
- Store/family target encoding

**Phase 3:** Baseline
- Simple moving average baseline
- LightGBM with basic features
- Target: <0.45 RMSLE

**Phase 4-7:** Feature engineering, model selection, tuning, ensembling

**Phase 8:** Submit with human approval

## Time Estimate

- EDA: 30 min
- Data pipeline: 1 hour
- Baseline: 30 min
- Feature engineering: 2-3 hours
- Model tuning: 3-4 hours (Optuna overnight)
- **Total:** ~8-12 hours autonomous work

## Resource Requirements

- Training time per experiment: ~5-15 min (3M rows)
- VRAM: Low (tabular data, <2GB)
- Disk: 200MB (data + models)
- API budget: ~50K tokens/hour (moderate Ollama usage)

## Competition-Specific Notes

- Time series requires chronological validation (no shuffle!)
- Sales can be zero (closed days/no demand) - handle in log transform
- Multiple product families may need separate models or hierarchical approach
- Store clusters suggest geographic/demographic patterns to explore
- Oil prices may correlate with Ecuador's economy (petroleum exporter)

---

**✅ CLEARED FOR AUTONOMOUS OPERATION**
