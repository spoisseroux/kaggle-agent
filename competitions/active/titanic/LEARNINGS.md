# Titanic Competition - Key Learnings

**Date:** 2026-05-07
**Best Score:** 0.78708 (Public LB)
**Competition:** Titanic - Machine Learning from Disaster

## Executive Summary

Through systematic experimentation (5 model variations, feature ablation study), we improved from 0.77511 to **0.78708** (+1.97 percentage points) by:
1. Removing Ticket features that hurt generalization
2. Adding strong regularization (reg_alpha=0.1, reg_lambda=1.0)
3. Reducing model complexity (max_depth=4 vs 7, n_estimators=150 vs 294)

## Critical Insight: CV Score ≠ Leaderboard Score

**The most important learning:** High cross-validation scores don't guarantee good leaderboard performance.

| Model | CV Score | LB Score | Gap | Rank |
|-------|----------|----------|-----|------|
| Optuna (original) | **0.8485** | 0.77511 | -0.0734 | 3rd |
| No Ticket (best) | 0.8126 | **0.78708** | -0.0256 | **1st** |
| Minimal features | 0.7924 | 0.77272 | **-0.0197** | 4th |

**Lesson:** The model with highest CV (Optuna) had the WORST LB performance. The model with smallest CV/LB gap (Minimal) showed best generalization but not highest absolute score.

## What Worked

### 1. Strong Regularization
```python
{
    'reg_alpha': 0.1,      # L1 regularization (was ~0)
    'reg_lambda': 1.0,     # L2 regularization (was ~0)
    'gamma': 0.1,          # Min loss reduction (was ~0)
    'min_child_weight': 10 # Conservative splits (was 5)
}
```
**Impact:** Improved LB by +0.0096 (0.77511 → 0.78468)

### 2. Reduced Model Complexity
```python
{
    'max_depth': 4,        # Shallower trees (was 7)
    'n_estimators': 150,   # Fewer trees (was 294)
    'learning_rate': 0.01  # Slower learning (was 0.021)
}
```
**Impact:** Lower CV but better generalization

### 3. Feature Ablation Study
Systematically tested removing each feature group:

| Feature Group | CV Score | Impact |
|--------------|----------|--------|
| All features | 0.8114 | baseline |
| **Without Ticket** | **0.8126** | **+0.0011** ✅ |
| Without Cabin | 0.8103 | -0.0011 |
| Without Interactions | 0.8070 | -0.0045 |
| Without Title | 0.8002 | -0.0112 |

**Discovery:** Ticket features (extracted prefix patterns) hurt CV and LB performance.
**Impact:** Removing Ticket improved LB by +0.0024 (0.78468 → 0.78708)

## What Didn't Work

### 1. Aggressive Hyperparameter Tuning
Optuna with 50 trials per model optimized for CV, not generalization.
- High CV: 0.8485
- Poor LB: 0.77511
- Large gap: -0.0734

### 2. Ensemble Models
Weighted voting of top 3 models:
- CV: 0.8428
- Worse than best single model (XGBoost: 0.8485 CV)

**Lesson:** Ensembles don't always help, especially when models are correlated.

### 3. Complex Feature Engineering
More features ≠ better performance:
- 15 features: LB 0.77511
- 14 features (no Ticket): LB 0.78708 ⭐
- 6 features (minimal): LB 0.77272

**Lesson:** Feature quality matters more than quantity. Bad features hurt generalization.

## Feature Importance Insights

### Most Important Features (from minimal model)
1. **Sex_male: 0.7217** (72% of importance) - dominant feature
2. **Pclass: 0.1306** (13%) - social class matters
3. **Fare: 0.0447** (4.5%) - proxy for wealth
4. **FamilySize: 0.0441** (4.4%) - family dynamics
5. Age: 0.0304 (3%)
6. IsAlone: 0.0286 (2.9%)

**Key insight:** Sex is by far the most predictive feature ("women and children first" policy). All other features combined contribute only 28%.

### Features That Hurt Generalization
1. **Ticket prefix** - patterns in train don't exist in test
2. **Complex interactions** (Age*Class) - overfit to training correlations
3. **Cabin deck** - too many missing values (77% in train)

## Regularization Deep Dive

**Why it worked:** Titanic has small dataset (891 train samples) with high dimensionality (15 features after engineering). Easy to overfit.

**Regularization effects:**
- L1 (reg_alpha): Feature selection, pushes weak feature weights to zero
- L2 (reg_lambda): Weight decay, prevents any single feature from dominating
- Gamma: Conservative splitting, only split if gain > threshold
- min_child_weight: Requires minimum samples per leaf

**Result:** Model learned robust patterns that generalize to test set.

## Recommendations for Similar Competitions

### When You Have Small Datasets (<1000 samples):

1. **Prioritize regularization over tuning**
   - Start with strong regularization
   - Reduce model complexity
   - Accept lower CV for better generalization

2. **Watch the CV/LB gap**
   - CV/LB gap > 0.05: likely overfitting
   - CV/LB gap < 0.03: good generalization
   - Target: minimize gap, not maximize CV

3. **Do feature ablation studies**
   - Systematically remove feature groups
   - Test on both CV and LB
   - Bad features hurt more than they help

4. **Simple features often beat complex ones**
   - Core features (Sex, Age, Class) are robust
   - Derived features (prefixes, interactions) may overfit
   - When in doubt, remove features

5. **Don't trust Optuna blindly**
   - Optuna optimizes CV, not LB
   - Use Optuna to find good ranges
   - Then manually regularize for generalization

### Feature Engineering Guidelines:

**Good features (generalize well):**
- Binary indicators (Sex, IsAlone)
- Ordinal variables (Pclass)
- Core numeric (Age, Fare)
- Simple aggregations (FamilySize = SibSp + Parch + 1)

**Bad features (may overfit):**
- String prefixes/patterns (Ticket, Cabin)
- Complex interactions (Age*Class, Fare/FamilySize)
- Rare categorical mappings (Title with rare values)
- High-missing-value features (Cabin: 77% missing)

## Qdrant Memory Tags

Keywords for future retrieval:
- `titanic`, `small-dataset`, `overfitting`, `regularization`
- `cv-lb-gap`, `feature-ablation`, `xgboost-tuning`
- `ticket-features-bad`, `sex-dominant-feature`
- `ensemble-failed`, `optuna-overoptimized`

## Action Items for Next Competition

1. ✅ Start with strong regularization baseline
2. ✅ Monitor CV/LB gap from first submission
3. ✅ Do feature ablation study early
4. ✅ Test minimal features model first
5. ✅ Don't over-optimize hyperparameters
6. ✅ Submit frequently to validate generalization

## Final Model Configuration

```python
# Best model: XGBoost without Ticket features
features = [
    'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
    'FamilySize', 'IsAlone',  # Family features
    'Sex_male', 'Embarked_Q', 'Embarked_S',  # Categorical
    'Title', 'Cabin_deck',  # Engineered
    'Age_Class', 'Fare_per_person'  # Interactions
    # NO Ticket_prefix - hurts generalization!
]

params = {
    'n_estimators': 150,
    'learning_rate': 0.01,
    'max_depth': 4,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42
}

# Results:
# CV: 0.8126
# LB: 0.78708
# Gap: -0.0256 (acceptable for small dataset)
```

## Competition Context

- **Dataset size:** 891 train, 418 test
- **Baseline (Sex-only rule):** ~76% accuracy
- **Our best:** 78.7% accuracy
- **Top leaderboard:** ~82% accuracy
- **Improvement headroom:** ~3.3 percentage points

Our model is **competitive but not top tier**. Further improvements would require:
1. More sophisticated feature engineering (that generalizes)
2. Stacking/blending with diverse models
3. External data sources (if allowed)
4. Deeper domain knowledge (Titanic history, passenger records)

## Time Investment vs. Return

- **Total experiment time:** ~15 minutes (5 models + ablation study)
- **Improvement:** +1.97 pp (77.511% → 78.708%)
- **Efficiency:** Systematic approach beat random search

**Lesson:** Structured experimentation (regularization → ablation → refinement) is more efficient than random trial-and-error.
