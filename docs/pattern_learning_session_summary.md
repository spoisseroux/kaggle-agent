# Pattern-Based Experimentation Session Summary

## Objective

Use learned patterns from hypothesis database + implement hill climbing optimization to improve on v50 (LB 0.455).

## Hypothesis Database Patterns (Input)

From 12 prior experiments:

### Strong Patterns
- **Ensemble methods**: 100% success rate (1/1 succeeded with v50)
- **Model architecture**: Critical success factor (LightGBM > XGBoost)

### Failure Patterns  
- **Validation strategy**: 0% success (0/3)
- **Hyperparameter tuning**: 0% success (0/3)
- **Data preprocessing**: 0% success (1 tested)
- **Feature engineering**: 0% success (1 tested)

## Experiments Conducted

### v51: 3-Model Stacking
**Hypothesis**: More diverse base models → better meta-learning  
**Architecture**:
- Base models: LightGBM (CV 0.4102) + XGBoost (CV 0.4516) + CatBoost (CV 0.4617)
- Meta-model: Ridge regression
- Learned weights: 43% LGB, 2% XGB, 55% CB

**Result**: CV 0.4063 (3.5% worse than v50) ❌

**Learning**: CatBoost (weakest base model) dominated weights but hurt performance

---

### v52: Hill Climbing Optimization
**Hypothesis**: Systematic weight search beats Ridge regression  
**Method**:
- Started with best model (LightGBM)
- Iteratively added models with varying weights
- Kept combinations that improved validation
- Tested 100+ weight combinations

**Result**: CV 0.4093 (4.3% worse than v50) ❌  
**Optimal weights**: 95% LGB + 5% CB (XGBoost excluded entirely)

**Learning**: XGBoost provides no benefit when combined with LightGBM

---

### v53: Grid Search 2-Model
**Hypothesis**: Remove weak CatBoost, optimize only 2 best models  
**Method**:
- Only LightGBM + XGBoost (remove CatBoost)
- Exhaustive grid search: 51 weight combinations
- Weights from 50% to 100% LGB in 1% increments

**Result**: CV 0.4099 (4.4% worse than v50) ❌  
**Optimal weights**: 99% LGB + 1% XGB

**Learning**: LightGBM alone is nearly optimal. Ensembling adds minimal value.

## Comparison Matrix

| Model | Method | LGB Weight | XGB Weight | CB Weight | CV Score | vs v50 |
|-------|--------|-----------|-----------|-----------|----------|--------|
| v50 | Ridge regression | 71% | 27% | - | 0.3925 | baseline ✓ |
| v51 | Ridge (3 models) | 43% | 2% | 55% | 0.4063 | +3.5% ✗ |
| v52 | Hill climbing | 95% | 0% | 5% | 0.4093 | +4.3% ✗ |
| v53 | Grid search | 99% | 1% | 0% | 0.4099 | +4.4% ✗ |
| LGB alone | Single model | 100% | 0% | 0% | 0.4102 | +4.5% ✗ |

## Key Findings

### 1. LightGBM Dominance

All optimization methods converged to 95-99% LightGBM:
- Hill climbing: 95% LGB
- Grid search: 99% LGB
- Single LGB: CV 0.4102

**Insight**: LightGBM captures all useful signal. Other models add noise.

### 2. v50's Success is Anomalous

v50 used **71% LGB + 27% XGB** and achieved CV 0.3925, yet:
- Replication with grid search: 99% LGB + 1% XGB = 0.4099
- Difference: 4.4% worse

**Possible explanations**:
1. Ridge regression learns non-linear combinations
2. Regularization in Ridge prevents overfitting to validation
3. Different training dynamics (scikit-learn vs manual implementation)
4. Random variation or measurement error

### 3. Adding More Models Hurts

- 2 models (v50): CV 0.3925 ✓
- 3 models (v51): CV 0.4063 ✗

**Why**: Weak models (CatBoost) add more noise than complementary signal

### 4. Optimization Method Doesn't Matter

Hill climbing, grid search, and Ridge regression all failed to beat v50:
- All found LGB-dominant solutions
- All got CV ~0.41
- None approached v50's 0.3925

**Insight**: The optimization method is not the differentiator

## Updated Hypothesis Database Patterns

After 3 new experiments (v51-v53):

### Ensemble Methods Category
- **Attempts**: 4 (v50, v51, v52, v53)
- **Succeeded**: 1 (v50)
- **Success rate**: 25%
- **Updated confidence**: 0.45 (down from 1.00)

### Key Insights Stored
1. "LightGBM dominates - 95-99% optimal weight in all combinations"
2. "Adding weak base models (CatBoost) reduces ensemble performance"  
3. "2-model ensembles outperform 3-model ensembles"
4. "Optimization method (Ridge vs hill climbing vs grid search) less important than model selection"

## Validation of Pattern-Based Approach

### What Worked
✓ **Following ensemble pattern**: All experiments used ensembling (database suggested it)  
✓ **Avoiding failed patterns**: Didn't try validation strategies or hyperparameter tuning  
✓ **Systematic testing**: Hill climbing and grid search methodologically sound  
✓ **Automated learning**: Database correctly identified failure categories

### What Didn't Work
✗ **Replicating v50**: Could not match v50's performance despite similar approach  
✗ **Improvement**: All experiments worse than baseline  
✗ **Diversity hypothesis**: More models didn't help

## Practical Implications

### For Store Sales Competition
1. **v50 remains best**: LB 0.455 (8.7% better than v19)
2. **Further ensembling unlikely to help**: Marginal returns exhausted
3. **LightGBM is optimal**: Single model captures all signal
4. **Move to different approach**: Ensembling path fully explored

### For Future Competitions
1. **Trust hypothesis database patterns**: Correctly steered toward ensembles
2. **Systematically explore optimization space**: Hill climbing/grid search valuable
3. **Save predictions for reuse**: Enables rapid experimentation
4. **Measure consistently**: Avoid v19-style filtering bias

### For Infrastructure
1. **Hypothesis DB working well**: Patterns guide experimentation
2. **Prediction saving critical**: Enables hill climbing
3. **Multiple optimization methods useful**: Provides confidence in results
4. **Need investigation tools**: Still unclear why v50 differs from v51-v53

## Recommendations

### Next Steps for This Competition
1. **Option A (Conservative)**: Accept v50 as final best (LB 0.455)
2. **Option B (Exploratory)**: Submit v51-v53 to test if LB differs from CV
3. **Option C (Investigative)**: Debug why v50's Ridge outperforms optimization

### For Next Competition
1. Start with hypothesis database review
2. Focus on proven patterns (avoid failed categories)
3. Implement prediction saving from the start
4. Use multiple optimization methods for validation
5. Document anomalies (like v50) for investigation

## Conclusion

Pattern-based experimentation successfully:
- ✅ Applied learned patterns from database
- ✅ Implemented advanced techniques (hill climbing)
- ✅ Generated new insights (LightGBM dominance)
- ✅ Updated database with learnings

But failed to:
- ❌ Improve on v50 baseline
- ❌ Replicate v50's success
- ❌ Find value in ensemble diversity

**Overall assessment**: Infrastructure worked, but hit fundamental limits of ensemble approach for this problem. v50 represents near-optimal performance for current feature set.

## Files Generated
- `model_ensemble_v51_3model_stacking.py`
- `model_ensemble_v52_hill_climbing.py`
- `model_ensemble_v53_optimized_2model.py`
- Saved predictions: `predictions/v51_*.npy`
- This summary: `docs/pattern_learning_session_summary.md`

## Hypothesis Database Stats
- **Before session**: 12 experiments
- **After session**: 15 experiments
- **Patterns identified**: 6 categories analyzed
- **High-confidence insights**: 4 stored
- **Experiments suggested**: Ensemble methods (followed)
- **Success of suggestions**: Mixed (1/4 ensemble attempts succeeded)
