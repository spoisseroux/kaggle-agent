# Path to 0.376 Leaderboard Score

**Current Position**: LB 0.455 (v50/v54)  
**Target**: LB 0.376 (top of leaderboard)  
**Gap**: 17.2% improvement needed

## Root Cause Analysis: Why Top Scorers Beat Us

### 1. Recursive Forecasting ⭐ HIGHEST IMPACT
**What top scorers do:**
- Predict day-by-day instead of all 16 test days at once
- Each prediction uses real lag features from prior days
- Errors accumulate but much better than constant mean

**What we do (v19/v50):**
- Fill ALL lag features with single constant (30-day mean)
- Only day_of_week, is_weekend, is_holiday, onpromotion vary in test set
- Loses all temporal dynamics

**Evidence:**
- v32 attempted this but failed with implementation bugs (LB 0.594)
- Negative correlation with v19 (-0.0246) indicates fundamental bug, not concept failure
- Top notebooks explicitly use recursive forecasting

**Expected Impact**: 15-20% LB improvement (0.455 → 0.37-0.38)

**Action**: Fix v32 implementation bugs and retest

---

### 2. Feature Engineering at Scale ⭐ HIGH IMPACT
**What top scorers do:**
- Generate 100s-1000s of features systematically
- Interaction features (store × family, store × dayofweek, etc.)
- Aggregation features (percentiles, quantiles, multi-window stats)
- Let model select useful features automatically

**What we do:**
- Manually design 12 features (lags 3/7, rolling means, day_of_week, etc.)
- Limited feature interactions
- No systematic generation

**From research:**
> "Generate Thousands of Features: Systematic generation of 10,000+ features (interactions, aggregations)"

**Expected Impact**: 5-10% LB improvement

**Action**: Implement automated feature generation pipeline

---

### 3. Store-Family-DayOfWeek Specific Features ⭐ MEDIUM IMPACT
**What top scorers do:**
- Average sales for THIS product/store on Mondays specifically
- "Recent Winning Strategies: Average for specific day: Store-family-day_of_week mean is highly predictive"

**What we do:**
- Global day_of_week feature (0-6)
- No store/family-specific dayofweek patterns

**Expected Impact**: 3-5% LB improvement

**Action**: Add store_family_dow_mean feature

---

### 4. Recent Data Focus ⭐ MEDIUM IMPACT
**What top scorers do:**
- Train only on last 1.5 years of data
- Recent patterns more relevant than 2013-2014 data

**What we do:**
- Use full 4+ year history (2013-2017)
- Older data may dilute recent patterns

**From research:**
> "Use less data: Last 1.5 years of training data performs better than full history"

**Expected Impact**: 2-5% LB improvement

**Action**: Train on data >= 2016-01-01

---

### 5. Seed Ensembling ⭐ MEDIUM IMPACT
**What top scorers do:**
- Train same model 5-10 times with different random seeds
- Average predictions to reduce variance

**What we do:**
- Single model per configuration
- No seed-based ensembling

**From research:**
> "Seed ensembling (multiple random seeds), Full-data retraining after CV optimization"

**Expected Impact**: 2-3% LB improvement

**Action**: Train LightGBM with 5 different random seeds and average

---

### 6. Pseudo-Labeling ⭐ LOWER IMPACT (Test Set Unknown)
**What it is:**
- Use model predictions on test set as "labels"
- Retrain including test set with pseudo-labels
- Iterate multiple times

**Challenge:**
- Works best when test labels can be inferred confidently
- May not help much for time series with future uncertainty

**Expected Impact**: 1-3% LB improvement (uncertain)

**Action**: Implement if other methods don't reach 0.376

---

### 7. Larger Ensembles ⭐ LOWER IMPACT (Diminishing Returns)
**What top scorers do:**
- Ensemble 5-10+ diverse models
- Different model families (LightGBM, XGBoost, CatBoost, Neural Nets)

**What we do:**
- 2-model ensemble (LightGBM + XGBoost)
- Ridge meta-learner

**Challenge:**
- v51-v53 showed adding more models HURT performance
- LightGBM captures all useful signal
- Need fundamentally different approaches (neural nets, recursive forecasting)

**Expected Impact**: 1-2% LB improvement

**Action**: Only pursue if recursive forecasting works

---

## Recommended Experiment Sequence

### Phase 1: Fix Recursive Forecasting (Highest Expected Impact)
**Goal**: Bridge 15-20% gap to 0.37-0.38

**v55 - Debug Recursive Forecasting:**
1. Fix v32 bugs:
   - Line 138: Use proper fallback instead of fillna(0)
   - Verify feature alignment in concat operations
   - Add extensive logging/validation per day
2. Test on single store-family first (faster iteration)
3. Validate predictions have positive correlation with v19
4. Full test set prediction

**Success Criteria**: LB 0.37-0.40 (matches top scorers)

**Estimated Time**: 2-4 hours

**If Success**: Problem largely solved, stop here or do Phase 2 for final 1-2%
**If Failure**: Move to Phase 2 (feature engineering)

---

### Phase 2: Feature Engineering at Scale
**Goal**: 5-10% improvement through better features

**v56 - Store-Family-DayOfWeek Features:**
1. Calculate mean sales for each (store, family, dayofweek) combination
2. Add to existing v1 features
3. Train LightGBM

**Success Criteria**: CV < 0.35, LB < 0.45

**v57 - Recent Data Only:**
1. Filter training data to >= 2016-01-01
2. Same v1 features
3. Train LightGBM

**Success Criteria**: CV < 0.36, LB < 0.46

**v58 - Automated Feature Generation:**
1. Interaction features (store × family, store × dow, family × dow)
2. Multi-window aggregations (percentiles, quantiles)
3. Polynomial features (squares, cubes of numeric)
4. Feature selection (remove low-importance)

**Success Criteria**: CV < 0.34, LB < 0.43

**Estimated Time**: 4-6 hours

---

### Phase 3: Seed Ensembling & Refinement
**Goal**: Final 2-3% improvement

**v59 - Seed Ensemble:**
1. Train best model from Phase 1/2 with seeds [42, 123, 456, 789, 2024]
2. Average predictions

**Success Criteria**: LB < 0.38

**v60 - Full Ensemble:**
1. Combine recursive forecasting + best features + seed ensemble
2. Ridge stacking if multiple approaches work

**Success Criteria**: LB < 0.376 🎯

**Estimated Time**: 2-3 hours

---

## LangGraph & Evals: Should We Use Them?

### LangGraph
**What it could help with:**
- ✅ Orchestrate complex multi-step workflows (recursive forecasting with validation)
- ✅ Manage state across feature generation → training → ensembling pipeline
- ✅ Coordinate parallel experiments (test multiple hypotheses simultaneously)
- ✅ Handle errors and retries in recursive loops

**What it WON'T help with:**
- ❌ Better features (still need domain knowledge)
- ❌ Better models (still need good algorithms)
- ❌ Understanding data (still need EDA)

**Recommendation**: **YES, but for Phase 1 (recursive forecasting)**
- Recursive forecasting is complex multi-step (16 days × feature creation × prediction)
- LangGraph state management could prevent bugs like v32
- Good investment if we're doing complex workflows

**Implementation**:
```python
# LangGraph workflow for recursive forecasting
from langgraph.graph import Graph

workflow = Graph()
workflow.add_node("load_history", load_train_data)
workflow.add_node("forecast_day", forecast_single_day)
workflow.add_node("update_history", append_to_history)
workflow.add_node("validate", check_predictions)

# Loop 16 times for test period
workflow.add_edge("load_history", "forecast_day")
workflow.add_edge("forecast_day", "validate")
workflow.add_edge("validate", "update_history")
workflow.add_conditional_edge("update_history", should_continue, "forecast_day", "end")
```

**Benefit**: Clearer logic, easier debugging, state persistence

---

### Evals
**What it could help with:**
- ✅ Validate feature quality before expensive training
- ✅ Test forecasting strategies on historical data
- ✅ Detect data leakage automatically
- ✅ Measure feature importance systematically

**What it WON'T help with:**
- ❌ Finding new features (still need generation strategy)
- ❌ Model selection (still need to train models)

**Recommendation**: **YES, but for Phase 2 (feature engineering)**
- Generate 1000s of features → eval quality → select top-N before training
- Prevents overfitting to validation set
- Saves compute time (don't train with bad features)

**Implementation**:
```python
from deepeval import evaluate
from deepeval.metrics import FreshnessMetric, DataLeakageMetric

# Eval each feature before training
for feature in generated_features:
    result = evaluate(
        [
            FreshnessMetric(train_dist, test_dist),  # Distribution shift
            DataLeakageMetric(feature, target),       # Correlation with future
        ]
    )
    if result.score > 0.8:
        selected_features.append(feature)
```

**Benefit**: Quality gating, faster iteration, leakage prevention

---

## Recommended Tooling Investments

**Priority 1: LangGraph for Recursive Forecasting**
- Highest impact experiment needs robust workflow
- Prevents bugs like v32
- Reusable for future multi-step experiments

**Priority 2: Evals for Feature Engineering**
- Scales to 1000s of features
- Prevents data leakage
- Faster iteration (quality gate before training)

**Priority 3: Automated Feature Generation**
- Polynomial, interaction, aggregation features
- Systematic generation at scale
- Let model do feature selection

**Don't Need:**
- AutoML (we understand the problem, manual is fine)
- Neural architecture search (tree models work well)
- Hyperparameter tuning at scale (v19 params already good)

---

## Success Probability Estimate

**High Confidence (>80%)**: Recursive forecasting alone gets us to 0.37-0.40
- Top scorers explicitly use this
- Our v32 concept was right, just buggy
- 15-20% improvement expected

**Medium Confidence (50-70%)**: Feature engineering adds 5-10%
- Research shows thousands of features help
- Store-family-dow proven effective
- Recent data focus validated

**Lower Confidence (30-50%)**: Seed ensemble adds final 2-3%
- Works but diminishing returns
- May not be necessary if Phase 1 succeeds

**Combined**: 70% chance of reaching 0.376 with Phases 1+2

---

## Time Investment

**Phase 1 (Recursive)**: 2-4 hours + LangGraph setup (4 hours) = **6-8 hours**
**Phase 2 (Features)**: 4-6 hours + Evals setup (3 hours) = **7-9 hours**
**Phase 3 (Ensemble)**: 2-3 hours

**Total**: 15-20 hours to 0.376 target

**Quick Path**: If Phase 1 succeeds → 6-8 hours to ~0.37 🎯

---

## Next Steps

1. **Decide on tooling**: LangGraph + Evals or manual implementation?
2. **Start Phase 1**: Debug recursive forecasting (v55)
3. **Measure results**: If LB < 0.40, continue to Phase 2
4. **Iterate**: Each phase builds on prior success

**Recommendation**: Start with **v55 recursive forecasting** using **LangGraph for robustness**. This single experiment has the highest probability of bridging the 17.2% gap to top leaderboard.
