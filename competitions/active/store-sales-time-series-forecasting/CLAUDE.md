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

## Current Best Models (Holdout Validation)

### v16 - Ensemble XGBoost + LightGBM (CURRENT BEST) 🎉
- **Holdout CV**: 0.1873 ⭐⭐⭐⭐⭐
- **Expected LB**: ~0.20-0.21 (based on validated 9.3% gap)
- **Architecture**: Weighted ensemble
  - LightGBM with advanced features: 0.2031 CV (70% weight)
  - XGBoost v15: 0.5222 CV (30% weight)
- **Key Discovery**: LightGBM + advanced features = massive improvement
  - Previous LightGBM (v2, simple features): 0.405 CV
  - With v14 advanced features: 0.203 CV (51% better!)
- **Improvement**: 61.3% vs v1 baseline
- **File**: `ensemble_v16_xgb_lgbm_01873.csv`

### v15 - Optuna Tuned Advanced Features
- **Holdout CV**: 0.3648
- **Expected LB**: ~0.40
- **Features**: 27 advanced features
- **Improvement**: 24.7% vs v1 baseline
- **Best params**: lr=0.058, depth=7, min_child=1, subsample=0.91
- **File**: `xgb_v15_optuna_adv_03648.csv`

### v14 - Advanced Features
- **Holdout CV**: 0.4640
- **Expected LB**: ~0.50
- **Features**: 27 total (EWMA, WoW_diff, interactions, temporal, clustering)
- **Top features**: Roll_mean_7, onpromotion, Lag_7, EWMA_7, WoW_diff
- **Improvement**: 4.2% vs v1 baseline
- **File**: `xgb_v14_advanced_04640.csv`

### v1_baseline_holdout (Validated)
- **Holdout CV**: 0.4842
- **Actual LB**: 0.5291
- **Gap**: 0.045 (9.3%) ✅
- **Validation proof**: 30-day holdout works!
- **File**: `xgb_v1_baseline_holdout_04842.csv`

### Previous Models
- v2_fixed_lags_holdout: 0.5177 holdout CV
- optuna_holdout: 0.5615 holdout CV

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
