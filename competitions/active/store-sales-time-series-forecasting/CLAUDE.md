# Store Sales - Time Series Forecasting

- slug: `store-sales-time-series-forecasting`
- metric: `rmsle` (Root Mean Squared Logarithmic Error, lower is better)
- deadline: `2030-06-30`
- type: Getting Started (Knowledge reward)

See repo-root CLAUDE.md for the master workflow.

## Competition Overview
Predict store sales using time series data from Corporación Favorita, a large Ecuadorian-based grocery retailer.

## Notes
- Time series forecasting problem (different from classification)
- Multiple stores and product families
- External data available (oil prices, holidays, etc.)
- Good for testing autonomous system with different problem type

## Learnings (Updated May 9, 2026)

### Validation Strategy ⭐ CRITICAL
**PROBLEM**: TimeSeriesSplit CV scores don't predict leaderboard performance
- Best model CV (TSCV): 0.315
- Best model LB: 0.464  
- Gap: 0.149 (47% higher!)

**TESTED 3 STRATEGIES** (v13 experiment):
1. TimeSeriesSplit (5-fold): 1.2729 - TERRIBLE predictor
2. Last 16-day holdout: 0.4872 - Close to LB!
3. Last 30-day holdout: 0.4801 - Even closer to LB!

**ROOT CAUSE**: Test set is Aug 16-31, 2017 (16 days). TSCV uses much earlier validation folds that don't reflect the same temporal distribution.

**SOLUTION**: Use last-30-day holdout for all future validation
```python
cutoff_date = train_df['date'].max() - pd.Timedelta(days=30)
val_mask = train_df['date'] >= cutoff_date
train_mask = ~val_mask
X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
```

### Model Selection
**XGBoost is superior to LightGBM**
- XGBoost best CV: 0.315 (v3)
- LightGBM best CV: 0.4046 (v2)
- Focus optimization efforts on XGBoost

### Regularization
**Over-regularization causes underfitting** (v12 experiment)
- Strong L1/L2 + low learning rate: CV 0.8739 (terrible)
- Current params already well-balanced
- CV-LB gap not from overfitting, but from validation strategy

### Top Features (from v11 analysis)
Best 12 features (95% of predictive power):
1. Roll_mean_7, Roll_mean_14, Roll_mean_30
2. Roll_mean_60, Roll_mean_90
3. is_weekend, day_of_week
4. Lag_3, Lag_7
5. is_holiday, onpromotion
6. Roll_std_7

### External Data Warning
**Avoid transactions.csv** - causes severe data leakage
- xgb_v3_external_data: CV 0.315, LB 1.0 (worst possible)
- Oil prices and store metadata are safe
- Transaction data likely contains future information

## ⚠️ CRITICAL BUG DISCOVERED (May 9, 2026 Evening)

**v14, v15, v16 ALL INVALID** due to critical bugs:

1. **LabelEncoder Bug**: `store_family_enc` fit separately on train and test
   - Same store-family gets different encodings
   - Model completely confused on test set
   
2. **Oversimplified Test Features**: All lag/rolling features = single mean
   - No temporal patterns preserved
   - Explains catastrophic LB scores

3. **Result**: v16 ensemble CV 0.187 → LB 3.35 (1690% worse!)

**DO NOT USE v14, v15, or v16 for submission.**

## Current Best Models (Holdout Validation + LB Confirmed)

### v50 - Stacking Ensemble ⭐ CURRENT BEST (May 2026)
- **Holdout CV**: 0.3925 (unfiltered)
- **Actual LB**: 0.45493
- **Gap**: 15.9%
- **Architecture**: Two-stage stacking
  - Stage 1: LightGBM + XGBoost base models
  - Stage 2: Ridge meta-model (learned weights: 71% LightGBM, 27% XGBoost)
- **Features**: 12 v1 features (same as v19)
- **Result**: **8.7% better than v19 on LB** ✓
- **File**: `ensemble_v50_stacking_039245csv`
- **Rank**: Best validated submission
- **Why it works**: Meta-learning captures complementary strengths of different model architectures

### v54 - Ridge Optimized (Investigation-Validated) ⭐ INVESTIGATION COMPLETE (May 2026)
- **Holdout CV**: 0.3925 (exact match to v50)
- **Formula**: `predictions = 0.7145 × LightGBM + 0.2746 × XGBoost - 0.4937`
- **Discovery**: Ridge's **intercept term** is critical for ensemble performance
  - Ridge WITH intercept: CV 0.3925 ✓
  - Ridge WITHOUT intercept: CV 0.4165 ✗ (6.1% worse)
  - Grid search without intercept (v53): CV 0.4099 ✗ (4.4% worse)
- **Investigation**: Systematically tested 3 hypotheses (intercept, regularization, normalization)
- **Result**: Intercept term provides 5.8% CV improvement through bias correction
- **File**: `ensemble_v54_ridge_opt_039245csv`
- **Documentation**: `docs/ridge_intercept_discovery.md` (comprehensive investigation writeup)
- **Why it works**: Intercept corrects systematic overprediction in base models (-0.4937 shift)
- **Key insight**: Always use `fit_intercept=True` in Ridge/Lasso meta-learning

### Pattern Learning Experiments (v51-v53) - Led to Ridge Investigation

**v51 - 3-Model Stacking**
- **Hypothesis**: More diverse base models → better meta-learning
- **Architecture**: LightGBM + XGBoost + CatBoost → Ridge
- **Result**: CV 0.4063 (3.5% worse than v50) ❌
- **Learning**: Weak models (CatBoost CV 0.4617) add noise, not signal
- **Weights learned**: 43% LGB, 2% XGB, 55% CatBoost (worst model dominated!)

**v52 - Hill Climbing Optimization**
- **Hypothesis**: Systematic weight search beats Ridge regression
- **Method**: Start with best model, iteratively add others with varying weights
- **Result**: CV 0.4093 (4.3% worse than v50) ❌
- **Optimal**: 95% LGB + 5% CatBoost (XGBoost excluded entirely)
- **Learning**: XGBoost provides no benefit when combined with LightGBM

**v53 - Grid Search 2-Model**
- **Hypothesis**: Remove weak CatBoost, optimize only LGB + XGB
- **Method**: Exhaustive grid search (51 weight combinations, 50-100% LGB)
- **Result**: CV 0.4099 (4.4% worse than v50) ❌
- **Optimal**: 99% LGB + 1% XGB (essentially just LightGBM alone)
- **Learning**: All optimization converges to LGB-dominant solutions

**Key Pattern Discovered:**
All v51-v53 experiments worse than v50 despite:
- Different optimization methods (Ridge, hill climbing, grid search)
- Different model combinations (2 vs 3 models)
- Systematic search (100+ configurations tested)

**This anomaly triggered the investigation that discovered Ridge's intercept term is the differentiator!**

**Documentation**: `docs/pattern_learning_session_summary.md`

### v19 - LightGBM v1 Features (Previous Best)
- **Holdout CV**: 0.3572 (filtered, y>0 only) / 0.4523 (unfiltered)
- **Actual LB**: 0.49833
- **Gap**: 39.5% (filtered CV) / 10.3% (unfiltered CV)
- **Features**: 12 proven features (lags 3/7, rolling means 7/14/30/60/90, roll_std_7, day_of_week, is_weekend, is_holiday, onpromotion)
- **Model**: LightGBM with conservative parameters (depth=6, lr=0.05)
- **File**: `lgbm_v19_v1_features_03572.csv`
- **Important Note**: Reported CV 0.3572 excluded 14.69% of validation samples (sales=0), making it misleadingly low. True unfiltered CV is 0.4523.
- **Superseded by**: v50 stacking ensemble (8.7% better LB)

### v1_baseline_holdout (Historical)
- **Holdout CV**: 0.4842
- **Actual LB**: 0.5291
- **Gap**: 8.5%
- **Model**: XGBoost with manual hyperparameters
- **File**: `xgb_v1_baseline_holdout_04842.csv`

### v18 - Optuna on v1 Features
- **Holdout CV**: 0.4872
- **Actual LB**: 0.5334
- **Gap**: 9.5%
- **Result**: 0.62% worse than v1 in CV, 0.81% worse on LB ❌
- **Lesson**: v1 hyperparameters already optimal, Optuna didn't help
- **File**: `xgb_v18_v1_optuna_04872.csv`

### INVALID MODELS (Critical Bugs - DO NOT USE)

### v16 - Ensemble XGBoost + LightGBM ❌ CATASTROPHIC
- **Holdout CV**: 0.1873
- **Actual LB**: 3.353 (1690% worse!)
- **Bugs**: LabelEncoder fit separately on train/test, oversimplified test features
- **File**: `ensemble_v16_xgb_lgbm_01873.csv`

### v17 - Fixed Advanced Features ❌ STILL BROKEN
- **Holdout CV**: 0.4170
- **Actual LB**: 0.729 (75% worse!)
- **Bugs**: Test feature creation still fundamentally flawed
- **File**: `xgb_v17_fixed_04170.csv`

### v15 - Optuna Tuned Advanced Features ❌ BUGGY
- **Holdout CV**: 0.3648
- **Status**: Same bugs as v14/v16, not submitted
- **File**: `xgb_v15_optuna_adv_03648.csv`

### v14 - Advanced Features ❌ BUGGY
- **Holdout CV**: 0.4640
- **Status**: Critical bugs, not submitted
- **File**: `xgb_v14_advanced_04640.csv`

## Autonomous Progress (May 9, 2026 - Evening)

### Session 1: Bug Fixes and Validation ✅
- Fixed Telegram integration (142 stuck messages)
- Submitted v1_baseline to Kaggle
- Confirmed: Holdout CV 0.48 → LB 0.53 (9.3% gap)
- Discovered v14-v17 all had critical bugs

### Session 2: Hyperparameter Exploration ✅
- v18: XGBoost + Optuna → CV 0.487, LB 0.533
- Result: No improvement, v1 hyperparameters already optimal
- Learning: Hyperparameter tuning not the path forward

### Session 3: Model Architecture Breakthrough ✅
- v19: LightGBM + v1 features → CV 0.357, LB 0.498 ⭐ BEST!
- Improvement: 5.9% better on LB vs XGBoost
- Discovery: Model architecture matters more than hyperparameters
- Note: LightGBM has 39.5% CV-LB gap (vs XGBoost's 9.3%)

### Session 4: Overfitting Discovery ✅
- v20: LightGBM + Optuna → CV 0.352, LB 0.512 ❌
- Result: Better CV but WORSE LB (overfitting to validation)
- Learning: Aggressive optimization hurts generalization
- Conclusion: v19 simple hyperparameters beat over-optimized ones

## Key Learnings from Autonomous Work

### 1. Model Selection > Hyperparameters
LightGBM with simple params (v19) beats XGBoost with any tuning

### 2. Validation Gaps Are Model-Specific
- XGBoost: 9.3% gap (reliable)
- LightGBM: 39.5% gap (needs calibration)

### 3. Optuna Can Overfit
v20 showed better CV but worse LB - overfitting to validation set

### 4. Incremental Validation Works
Test one change at a time on LB to learn what actually helps

## Comprehensive Autonomous Session Results (v18-v25)

### All Experiments
1. **v18** XGBoost+Optuna: CV 0.487 → LB 0.533 ❌ No improvement
2. **v19** LightGBM: CV 0.357 → LB 0.498 ⭐ BEST!
3. **v20** LightGBM+Optuna: CV 0.352 → LB 0.512 ❌ Overfit
4. **v21** CatBoost: CV 0.446 ✅ Middle ranking
5. **v22** LightGBM+month: CV 0.358 ❌ No help
6. **v23** LightGBM+day_of_month: CV 0.377 ❌ Hurts (despite rank 1 importance!)
7. **v24** Ensemble (LightGBM+CatBoost): CV 0.357 ✅ Optimal weights: 100% LightGBM
8. **v25** LightGBM depth=8: CV 0.366 ❌ Worse than v19

### Final Validated Insights

**Model Architecture Ranking:**
1. LightGBM (v19): LB 0.498 ⭐
2. CatBoost (v21): Expected ~0.51
3. XGBoost (v1): LB 0.529

**What Makes v19 Optimal:**
- Simple v1 feature set (12 features)
- LightGBM architecture
- Conservative hyperparameters (depth=6, lr=0.05)
- No extra temporal features
- No aggressive tuning

**Why Everything Else Failed:**
- Optuna: Overfits to validation (v20)
- Temporal features: Add noise (v22, v23)
- Ensembling: No complementary value (v24)
- Deeper trees: Overfits (v25)
- XGBoost: Wrong model architecture (v18)

### Tomorrow's Queue (5 submissions available)
1. v21 CatBoost (CV 0.446)
2. v22 month feature (CV 0.358)
3. v23 day_of_month (CV 0.377)
4. v24 ensemble (CV 0.357)
5. v25 deeper (CV 0.366)

### Next Steps
1. Submit queued models to validate on LB
2. If v19 remains best, focus on:
   - Different feature types (not temporal)
   - External data (oil prices, properly encoded stores)
   - Alternative model architectures (Neural networks?)
3. Stop hyperparameter tuning - v19 params are optimal

## Advanced Experimentation Session (May 10, 2026)

### Battle Plan Execution
Attempted multiple advanced approaches to beat v19:

1. **v26 N-BEATS** (Neural Basis Expansion Analysis for Time Series)
   - Status: FAILED - NaN errors during validation
   - Architecture: 2 stacks (trend + seasonality), 30-day lookback, 16-day forecast
   - Conclusion: Numerical instability, not suitable for this problem

2. **v28 BiLSTM** (Bidirectional LSTM)
   - CV: 0.6096 (70% worse than v19)
   - Architecture: 2 layers, 128 units, dropout 0.2, bidirectional
   - Conclusion: RNN architectures don't capture sales patterns well

3. **v29 Fourier Transform** (FFT features)
   - CV: 0.3601 (0.8% worse than v19)
   - Features: FFT magnitudes from 7, 14, 30-day rolling windows (2 components each)
   - Conclusion: Frequency domain features add noise, not signal

4. **v30 External Data** (Oil prices + Store metadata)
   - CV: 0.3770 (5.5% worse than v19)
   - Features: Oil lag 1/7/30, rolling stats, pct_change + store city/state/type/cluster
   - Conclusion: External data not predictive for short-term sales patterns

### Key Findings

**Neural Networks Don't Work:**
- Both N-BEATS and BiLSTM failed catastrophically
- Tree-based models (LightGBM) are superior for this tabular time series

**Feature Engineering Hurts:**
- Every feature addition made CV worse
- v1 simple features (lags + rolling means) capture all useful signal
- Complex features (Fourier, external data) add noise

**v19 is Definitively Optimal:**
- 8 experiments tried to beat it (v22-v25, v26, v28-v30)
- All failed
- Simple v1 features + LightGBM architecture is the winning combination

### Remaining Work
1. Submit queued models (v21-v25) tomorrow to validate on LB
2. If nothing beats v19 on LB, accept it as final solution
3. Document learnings and move to next competition

### Tooling Improvements
- Fixed conversation monitor service (stdout buffering issue)
- All assistant messages now reliably forwarded to Telegram
- Autonomous experimentation workflow validated end-to-end


## Major Research Breakthrough (May 10, 2026 - Evening)

### Question: How do top scorers (0.376-0.379) beat my 0.498?

**Investigation Process:**
1. Downloaded top public notebooks (2895 votes comprehensive guide)
2. Analyzed leaderboard leaders' techniques
3. Found critical insight: **Recursive Forecasting**

### The Discovery

**v19's Test Set Handling (LB 0.498):**
For ALL lag and rolling features, fills with single constant (30-day mean):
```python
df["Lag_3"] = df["lag_mean"].fillna(0)  # Same value!
df["Lag_7"] = df["lag_mean"].fillna(0)  # Same value!
df[f"Roll_mean_{window}"] = df["lag_mean"].fillna(0)  # Same value!
```

Result: Only day_of_week, is_weekend, is_holiday, onpromotion actually vary in test set!

**Top Scorers' Approach: Recursive Forecasting**
Instead of predicting all 16 test days at once, predict day-by-day:
1. **Day 1**: Predict using real lag 1-7 from training
2. **Day 2**: Predict using lag 1 from Day 1 prediction + real lag 2-7 from training
3. **Day 3**: Predict using lag 1-2 from predictions + real lag 3-7 from training
4. Continue for all 16 days

**Why This Works:**
- Short lags (1-7 days) have high predictive power
- Each prediction builds real lag features for next day
- Errors accumulate but much better than constant mean
- Enables use of powerful recent patterns

### v32 Implementation

**Model:** LightGBM with v1 features (same as v19)  
**CV:** 0.3572 (same as v19 - validation uses batch)  
**Test Prediction:** Recursive (36 seconds for 16 days)  
**Expected LB:** 0.37-0.38 (top leaderboard range)

**Key Code:**
```python
def recursive_forecast(model, train_df, test_df, holidays, feature_cols):
    forecast_df = train_df.copy()
    
    for forecast_date in test_dates:
        # Combine history + current day
        combined = pd.concat([forecast_df, current_day])
        
        # Create features (lags from real data + predictions)
        combined = create_features(combined)
        
        # Predict
        predictions = model.predict(X_forecast)
        
        # Add to history for next day
        forecast_df = pd.concat([forecast_df, predictions])
```

### Hypothesis vs Reality

**Initial Hypothesis (WRONG):**
- Test is 16 days after training ends
- Lag 3/7 are NaN in test
- Need lag 16+ to be valid

**Actual Reality:**
- Test is 1 day after training (Aug 16 vs Aug 15)
- Test period is 16 days LONG (Aug 16-31)
- v19 fills lags with constant, not NaN
- Top scorers use recursive forecasting with short lags

### Impact

This explains the entire 24% gap:
- **v19 LB 0.498**: Crude approximation (all lags = same constant)
- **Top LB 0.376**: Sophisticated recursive with real lag features
- **v32 Expected**: Bridge the gap with proper technique

### Sources

- [Recursive MultiStep Time Series Forecasting](https://www.kaggle.com/code/ahmedabdulhamid/recursive-multistep-time-series-forecasting)
- [Store Sales Comprehensive Guide](https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide)
- Top leaderboard analysis (0.376-0.379 range)

### Next Steps

1. Submit v32 when daily limit resets (8:00 PM EDT / midnight UTC)
2. If LB confirms 0.37-0.38, recursive forecasting is validated
3. If not, investigate other top scorer techniques (ensembling, feature engineering on recursive predictions)

**Status:** v32 ready, awaiting submission slot

## v32 Submission Results (May 10, 2026 - Evening)

### Outcome: CATASTROPHIC FAILURE ❌

**Expectations vs Reality:**
- Expected LB: 0.37-0.38 (based on top scorer analysis)
- Actual LB: 0.5936 (19% WORSE than v19!)
- v19 LB: 0.4983 (baseline)

**Analysis:**
- v32 predictions have ZERO correlation with v19 (-0.0246)
- Not just error accumulation - fundamental implementation bug
- Recursive forecasting concept is sound, but implementation is broken

**Root Cause (Suspected):**
- Line 138: Fills NaN lag features with 0 instead of proper fallback
- Line 146: Potential index alignment issues in concat
- Feature engineering in recursive loop may be creating invalid features

**Lesson:** Recursive forecasting requires extremely careful implementation. The day-by-day loop is fragile and errors cascade quickly.

**Status:** v32 FAILED, not recommended for further use

## v33 Hierarchical Models (May 10, 2026 - Evening)

### Alternative Approach After v32 Failure

**Strategy:** Train separate LightGBM model for each of 33 product families

**Rationale:**
- Different families have different seasonality (AUTOMOTIVE ≠ GROCERY)
- Research shows top scorers use per-family or hierarchical models
- v19 uses single global model that averages across all patterns

**Results:**
- **CV: 0.3550** (0.62% better than v19's 0.3572)
- **File:** lgbm_v33_hierarchical_03550.csv
- **LB:** Pending submission approval

**Per-Family CV Variance:**
- Best: BOOKS (0.0885)
- Worst: LINGERIE (0.6078)
- Range: 6.9x difference shows importance of family-specific models

**Implementation:**
- 33 separate LightGBM models (one per family)
- Same v1 features for each model
- Same hyperparameters as v19 (depth=6, lr=0.05)
- Batch test prediction (not recursive)

**Expected LB Impact:** 3-5% improvement over v19 → ~0.47

**Status:** Submitted and FAILED

## v33 Submission Results (May 10/11, 2026)

### Outcome: CATASTROPHIC FAILURE ❌ (SAME AS v32!)

**Expectations vs Reality:**
- Expected LB: ~0.47 (3-5% improvement over v19)
- Actual LB: 0.5902 (18.5% WORSE than v19!)
- v19 LB: 0.4983 (baseline)

**Critical Finding:**
v33 predictions have NEGATIVE correlation with v19: **-0.0327**
(v32 had -0.0246)

This indicates **systematic bug** affecting both v32 and v33, not random error.

**What Makes This Mysterious:**
1. v33 used CORRECT feature creation (v19 approach with store-family means)
2. v32 and v33 use completely different approaches (recursive vs hierarchical)
3. Both fail with nearly identical LB scores (~0.59)
4. Both have negative correlation with working v19

**Comparison:**
- v19 mean predictions: 460.89
- v33 mean predictions: 368.07 (20% lower)
- v33 generally predicts LOWER than v19 across the board

**Hypothesis:**
There's a common bug in v32/v33 that:
- Reverses or inverts predictions somehow
- Occurs in training pipeline, not test feature creation
- Doesn't show up in CV but breaks LB predictions
- Not related to fillna approach (v33 used correct approach)

**Status:** v33 FAILED, systematic bug unidentified

## v34 Prepared (Recursive Forecasting - Fixed)

Created v34 with fix for v32's fillna(0) bug:
- Fills NaN lag features with store-family mean (like v19)
- Only date features use fillna(0)
- Expected to prevent v32's death spiral

**File:** model_lgbm_v34_recursive_fixed.py
**Status:** Ready to run, BUT likely will also fail if bug is elsewhere

## Critical Analysis Needed

**Pattern:**
- v19 (simple global model): LB 0.498 ✅
- v32 (recursive): LB 0.594 ❌
- v33 (hierarchical): LB 0.590 ❌
- Correlation v19-v32: -0.0246
- Correlation v19-v33: -0.0327

**Next Steps:**
1. **Line-by-line audit** - Compare v19 vs v33 training pipeline
2. **Check for:**
   - Data leakage in opposite direction
   - Reversed target variable
   - Train/test split issues
   - Feature preprocessing that inverts signal
3. **Test v34** - May also fail if bug is in common code
4. **Consider** - Return to v19 baseline, try simpler improvements first

**Current Best:** v19 at LB 0.498



## CV-LB Gap Investigation (May 11, 2026)

### Problem Statement
v19 has 39.5% CV-LB gap (CV 0.357 → LB 0.498). Investigated root causes and potential solutions through validation strategy experiments.

### Root Cause Analysis

**Validation Period Bias:**
- Training data: 2013-2017, mean sales 356
- Validation period (Jul 16-Aug 15, 2017): mean sales 472 (33% higher)
- Sales trend: 2013 (216) → 2014 (323) → 2015 (371) → 2016 (444) → 2017 (480)
- Validation uses PEAK sales period, causing optimization bias toward high sales

**Why the Gap Exists:**
- Model optimizes for unrepresentative high-sales validation period
- Test set (Aug 16-31) likely has similar elevated sales
- Gap is inherent to temporal split, not model overfitting
- Proper temporal validation requires this trade-off (no random splits in time series)

### Experiments to Address Gap

**v38 - Anti-Overfitting (LightGBM)**
- Strategy: Reduce regularization (L1=L2=0.1)
- CV: 0.366, LB: 0.533 (46% gap)
- Result: WORSE than v19 - regularization wasn't the issue ❌

**v39 - Grid Search L1 vs L2 (LightGBM)**
- Strategy: Test L1-only vs L2-only regularization
- 48 configurations tested
- Best CV: 0.4526 (26.7% worse than v19)
- Result: All configurations worse - regularization hurts ❌

**v40 - XGBoost Simple Parameters**
- Strategy: Switch to XGBoost (9.3% historical gap vs LightGBM's 39.5%)
- Parameters: max_depth=4, lr=0.05 (conservative)
- CV: 0.5220 (46% worse than v19)
- Result: XGBoost significantly worse than LightGBM ❌

**v41 - XGBoost Historical Parameters**
- Strategy: Use proven historical XGBoost params (max_depth=8, reg_alpha=0.1, reg_lambda=1.0)
- CV: 0.4554 (27.5% worse than v19)
- Result: Better than v40 but still worse than LightGBM ❌

**v42 - XGBoost 60-Day Validation**
- Strategy: Longer validation window to reduce bias (60 vs 30 days)
- Validation period: Jun 16-Aug 15 (still 34% higher sales than training)
- CV: 0.4595
- Result: Slightly worse than 30-day validation, bias persists ❌

**v43 - XGBoost Walk-Forward CV**
- Strategy: 3 temporal folds to reduce single-window bias
- Fold 1 (May 17-Jun 16): CV 0.4165
- Fold 2 (Jun 16-Jul 16): CV 0.4745
- Fold 3 (Jul 16-Aug 15): CV 0.4559
- Mean CV: 0.4490 ± 0.024
- Result: Better than single XGBoost holdout but worse than LightGBM ❌

**v44 - LightGBM Walk-Forward CV**
- Strategy: Apply walk-forward to best model (LightGBM)
- Mean CV: 0.4665 ± 0.016
- Fold 3 (same window as v19): CV 0.4557 vs v19's 0.3572
- Result: WORSE than v19 single holdout - unexpected! ❌
- Issue: Potential feature leakage or validation implementation difference

### Key Findings

**Model Architecture is Critical:**
- LightGBM (v19): CV 0.357 ⭐
- XGBoost (best): CV 0.449
- LightGBM consistently 25%+ better than XGBoost across all configurations

**Validation Strategies Don't Help:**
- 30-day holdout (v19): CV 0.357 ⭐ BEST
- 60-day holdout (v42): CV 0.460 (worse)
- Walk-forward XGBoost (v43): CV 0.449 (worse)
- Walk-forward LightGBM (v44): CV 0.467 (worse)

**Validation Bias is Inherent:**
- All 2017 summer validation periods have elevated sales (33-37% above mean)
- Bias exists in any temporal validation approach
- Cannot use random splits (causes data leakage)
- Trade-off between temporal validity and representativeness

**v19 is Optimal:**
- Best CV across 26+ experiments
- Simple approach (30-day holdout, v1 features, LightGBM)
- Additional complexity (walk-forward CV, longer windows, different models) doesn't improve results
- 39.5% CV-LB gap is acceptable given validation constraints

### Conclusion

**ACCEPTED: v19 as final best model**
- CV: 0.3572
- LB: 0.498
- Leaderboard gap is inherent to temporal validation bias, not fixable through:
  - Regularization tuning
  - Model architecture changes (XGBoost)
  - Validation strategy changes (longer windows, walk-forward CV)
  
**Recommendation:** Accept v19 and move to next competition. Further optimization unlikely to improve LB performance.

