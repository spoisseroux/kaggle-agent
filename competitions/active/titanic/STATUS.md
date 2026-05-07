# Titanic Competition - Current Status
**Last updated:** 2026-05-07

## Overview
All development work complete. Ready for submission pending Kaggle API authentication fix.

## Workflow Progress
- [x] EDA — completed (eda_summary.json)
- [x] Data pipeline — completed (data_pipeline.py)
- [x] Baseline — completed (baseline.py, 0.8339 CV)
- [x] Feature engineering — completed (v2, v3)
- [x] Model selection — completed (tested LightGBM, XGBoost, RandomForest)
- [x] Hyperparameter tuning — completed (Optuna, 50 trials per model)
- [x] Ensembling — completed (weighted voting tested, not better than single model)
- [ ] **Submit** — **BLOCKED: Kaggle API 401 error**

## Best Model: XGBoost
- **CV Score:** 0.8485 (5-fold stratified cross-validation)
- **Submission file:** `submissions/optuna_best_submission.csv`
- **Predicted survival rate:** 35.89%

### Hyperparameters
```python
{
    'n_estimators': 294,
    'learning_rate': 0.020796927448298978,
    'max_depth': 7,
    'min_child_weight': 5,
    'subsample': 0.9197121567728955,
    'colsample_bytree': 0.8979777408556597,
    'gamma': 3.081587442535289e-07,
    'reg_alpha': 0.015105023850298839,
    'reg_lambda': 1.1626943463555593e-08,
    'random_state': 42
}
```

## All Models Evaluated

| Model | CV Score | Improvement vs Baseline | Notes |
|-------|----------|------------------------|-------|
| Baseline (LogReg) | 0.8339 | - | Simple features only |
| Features v2 (LightGBM) | 0.8315 | -0.0024 | Added interaction features |
| Features v3 (LightGBM) | 0.8406 | +0.0067 | Improved feature engineering |
| LightGBM (Optuna) | 0.8451 | +0.0112 | Hyperparameter tuning |
| RandomForest (Optuna) | 0.8440 | +0.0101 | Hyperparameter tuning |
| **XGBoost (Optuna)** | **0.8485** | **+0.0146** | **Best model** ✅ |
| Ensemble | 0.8428 | +0.0089 | Weighted voting, underperformed |

## Submissions Ready (6 files)
1. `baseline_submission.csv` — 0.8339 CV
2. `features_v2_submission.csv` — 0.8315 CV
3. `features_v3_submission.csv` — 0.8406 CV
4. `lgbm_v2_submission.csv` — 0.8451 CV
5. `ensemble_submission.csv` — 0.8428 CV
6. **`optuna_best_submission.csv`** — **0.8485 CV** ⭐ **RECOMMENDED**

## Feature Engineering (features_v3)
The winning model uses these engineered features:

**Created features:**
- `Title` — extracted from Name (Mr, Miss, Mrs, Master, Rare)
- `FamilySize` — SibSp + Parch + 1
- `IsAlone` — binary (1 if FamilySize == 1)
- `Age*Class` — interaction between Age and Pclass
- `Fare_per_person` — Fare / FamilySize
- `Title_encoded` — numerical encoding of titles
- `Cabin_deck` — extracted first letter from Cabin
- `Ticket_prefix` — extracted prefix from Ticket number

**Imputation:**
- Age: filled with median by Title and Pclass
- Fare: filled with median
- Embarked: filled with mode ('S')
- Cabin: treated as missing indicator

## Kaggle API Issue
**Error:** 401 Client Error: Unauthorized

**Current credentials:**
- Username: spoisseroux
- API Key: KGAT_8faa...e57
- Location: `~/.kaggle/kaggle.json`

**Status:** Awaiting human to regenerate API key or provide updated credentials.

## Next Steps
1. Fix Kaggle API authentication
2. Submit `optuna_best_submission.csv` (0.8485 CV)
3. Record leaderboard score
4. If LB score differs significantly from CV, investigate potential overfitting
5. Consider additional feature engineering if time permits

## MLflow Tracking
- **Tracking URI:** http://localhost:5000
- **Experiment name:** titanic
- **Runs logged:** All major experiments (baseline, v2, v3, optuna results)

## Notes
- Competition is a tutorial/practice competition (no deadline pressure)
- Titanic dataset: 891 train samples, 418 test samples
- Target balance: 38.4% survived
- Missing data handled: Age (177), Cabin (687), Embarked (2)
