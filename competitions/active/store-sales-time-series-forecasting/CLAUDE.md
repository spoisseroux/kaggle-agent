# Store Sales - Time Series Forecasting

- **Slug**: `store-sales-time-series-forecasting`
- **Metric**: `rmsle` (Root Mean Squared Logarithmic Error, lower is better)
- **Deadline**: 2030-06-30
- **Type**: Getting Started (Knowledge reward)

## Current Best Model ⭐

**v67 - Ridge 90/10 Ensemble**: LB 0.45416, CV 0.3908
- Architecture: LightGBM (90%) + XGBoost (10%) + Ridge intercept
- Features: 12 v1 features (lags 3/7, rolling means 7/14/30/60/90, roll_std_7, day_of_week, is_weekend, is_holiday, onpromotion)
- Params: depth=6, num_leaves=64, lr=0.05, n_estimators=600
- File: `submissions/ensemble_v67_ridge_90_10_fixed_039078.csv`

**Alternative**: v50 (LB 0.45493, essentially tied with v67)

## Critical Rules ⚠️

1. **Conservative hyperparameters are optimal**
   - DO NOT tune depth/leaves (causes validation overfitting)
   - Validation period has 33% higher sales → tuning optimizes for anomalies
   - Query research DB for evidence: `python -c "from core.hypothesis_db import HypothesisDatabase; db=HypothesisDatabase(); db.get_competition_insights('store-sales-time-series-forecasting')"`

2. **Alignment bug pattern**
   - ALWAYS sort test_df BEFORE creating predictions
   - v63 and v73 failed (LB 3.5+ / 5.0+) due to predictions-IDs mismatch
   - Pattern: `test_df.sort_values(["store_nbr", "family", "date"])` BEFORE `model.predict()`

3. **Validation strategy**
   - Use last 30-day holdout (NOT TimeSeriesSplit)
   - Test set is Aug 16-31, 2017 (16 days)

4. **Invalid approaches** (stored in research DB)
   - Recursive forecasting (v32/v33/v34/v66: all failed)
   - EWM features (v68/v69: degraded performance)
   - Hyperparameter tuning (v20/v73: better CV, worse LB)
   - Feature engineering on 30-day validation (v81/v86: +49% CV, but LB 2.09/4.69 - features capture validation anomalies)
   - External data with transactions.csv (causes leakage)

## Research DB Usage

**At session start:**
```bash
python scripts/hooks/session_start.py
```

**Query insights:**
```python
from core.hypothesis_db import HypothesisDatabase
db = HypothesisDatabase()
insights = db.get_competition_insights("store-sales-time-series-forecasting", min_confidence=0.7)
history = db.get_experiment_history("store-sales-time-series-forecasting", limit=10)
```

**Log new experiments:**
```python
hyp_id = db.add_hypothesis(competition="...", experiment_id="v99", hypothesis="...", rationale="...", category="...")
db.record_result(hypothesis_id=hyp_id, cv_score=0.XX, lb_score=0.YY, baseline_cv=0.39, succeeded=True/False)
```

## Competition Status

- **21% gap to top scorers** (0.377) - not achievable with public techniques
- **Considered "solved"** at current performance
- All reasonable approaches tested and documented in research DB

## See Also

- Master instructions: `/home/keehar/kaggle-agent/CLAUDE.md`
- Research DB: Query for full experiment history
- Memory: `.claude/projects/.../memory/` for session context
