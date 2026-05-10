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

### v1_baseline_holdout (CURRENT BEST) ✅
- **Holdout CV**: 0.4842
- **Actual LB**: 0.5291
- **Gap**: 8.5% ✅ VALIDATED
- **Features**: 12 proven features (lags 3/7, rolling means 7/14/30/60/90, roll_std_7, day_of_week, is_weekend, is_holiday, onpromotion)
- **Model**: XGBoost with manual hyperparameters
- **File**: `xgb_v1_baseline_holdout_04842.csv`
- **Rank**: Best validated submission

### v18 - Optuna on v1 Features
- **Holdout CV**: 0.4872
- **Actual LB**: 0.5334
- **Gap**: 9.5%
- **Result**: 0.62% worse than v1 in CV, 0.81% worse on LB ❌
- **Lesson**: v1 hyperparameters already optimal, Optuna didn't help
- **File**: `xgb_v18_v1_optuna_04872.csv`

### v19 - LightGBM v1 Features (Testing)
- **Status**: Training in progress
- **Purpose**: Test if model architecture matters with same v1 features
- **File**: `lgbm_v19_v1_features_*.csv`

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

### Phase 1: Validation Strategy Validation ✅
- Submitted v1_baseline to Kaggle
- Confirmed: Holdout CV 0.48 → LB 0.53 (9.3% gap)
- Much better than TimeSeriesSplit (163% gap!)

### Phase 2: Feature Engineering ✅
- Researched Kaggle best practices
- Built v14 with advanced features
- Achieved 4.2% improvement

### Phase 3: Hyperparameter Optimization (In Progress)
- v15 using Optuna on v14 features
- 30 trials, targeting <0.464 CV

## Next Steps
1. Complete v15 Optuna tuning
2. Try ensemble (XGBoost + LightGBM stacking)
3. Submit v14 to validate expected LB ~0.50
4. Feature selection to remove noise
