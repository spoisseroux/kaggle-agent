# Titanic Competition - Performance Analysis

## CV vs Leaderboard Score Gap

### The Problem
- **Cross-Validation Score:** 0.8485 (84.85% accuracy)
- **Public Leaderboard Score:** 0.77511 (77.51% accuracy)
- **Gap:** -0.0734 (-7.34 percentage points)

This is a **significant discrepancy** that indicates our model is not generalizing well from training to test data.

### Possible Causes

#### 1. **Overfitting to Training Data**
Our model may have learned patterns specific to the training set that don't exist in the test set.

**Evidence:**
- Complex feature engineering (interactions, family size, title extraction)
- Aggressive hyperparameter tuning (294 estimators, max_depth=7)
- High CV score suggests strong fit to training data

**Solutions:**
- Try simpler models with less capacity
- Reduce feature engineering complexity
- Add more regularization (higher reg_alpha, reg_lambda)
- Use fewer estimators or shallower trees

#### 2. **Data Leakage**
We may have inadvertently used information from the test set during feature engineering.

**Check:**
- ✅ Train/test split done correctly
- ✅ Features engineered separately for train/test
- ✅ No target information in features

**Verdict:** Unlikely to be data leakage

#### 3. **Distribution Shift**
The test set may have different characteristics than the training set.

**Possible differences:**
- Different time period for passenger data
- Different demographic distribution
- Different missing data patterns

**Investigation needed:**
- Compare feature distributions between train/test
- Check if missing patterns differ significantly
- Analyze which passengers we're getting wrong

#### 4. **Cross-Validation Strategy**
Our CV strategy (5-fold stratified) may not reflect the actual train/test split.

**Issues:**
- Stratified only on target (Survived), not on other features
- No temporal split (if data has time component)
- Small dataset (891 samples) means high variance

**Solutions:**
- Try different CV strategies (10-fold, leave-one-out)
- Ensure CV split mimics train/test distribution
- Consider using group-based splitting if applicable

#### 5. **Feature Engineering Not Generalizing**
Some engineered features may work well in CV but not on test set.

**Suspects:**
- `Title` extraction (may have different titles in test)
- `Cabin_deck` encoding (many missing values)
- `Fare_per_person` (calculated from FamilySize)
- Interaction features (`Age*Class`)

**Next steps:**
- Train model without each feature group
- Check feature importance vs. generalization
- Simplify feature engineering

### Recommended Actions

#### Immediate (Quick Wins)
1. **Try baseline model** - Submit the simple LogisticRegression baseline to see if it generalizes better
2. **Reduce model complexity** - Lower max_depth, reduce n_estimators
3. **Stronger regularization** - Increase reg_alpha and reg_lambda

#### Short-term (Investigation)
4. **Feature ablation study** - Remove feature groups one by one
5. **Compare predictions** - Analyze which test samples we're getting wrong
6. **Distribution analysis** - Compare train/test feature distributions

#### Long-term (Systematic)
7. **Ensemble with simpler models** - Blend with less complex models
8. **Different CV strategy** - Try stratified group k-fold
9. **Manual feature selection** - Keep only features that generalize

## Next Submission Strategy

Try these in order:

1. **Simple baseline** (baseline_submission.csv) - already generated, just submit
2. **Regularized XGBoost** - Increase reg_alpha to 0.1, reg_lambda to 1.0
3. **Shallow XGBoost** - max_depth=3, n_estimators=100
4. **Feature-reduced model** - Only use: Sex, Pclass, Age, Fare, FamilySize, IsAlone
5. **Different algorithm** - Try LogisticRegression with carefully selected features

## Learnings

### What Worked
- ✅ Systematic workflow from EDA to tuning
- ✅ Optuna found good hyperparameters (for CV)
- ✅ Feature engineering improved CV scores
- ✅ MLflow tracking for experiments

### What Didn't Work
- ❌ High CV score didn't translate to high LB score
- ❌ Ensemble underperformed single model
- ❌ Aggressive tuning may have overfit

### Key Insight
**CV score is not the only metric** - We optimized heavily for CV accuracy but lost sight of generalization. Need to balance between CV performance and model simplicity.

## Competition Context

The Titanic competition is a **tutorial competition** with:
- Very small dataset (891 train, 418 test)
- Many missing values
- High baseline from simple rules (Sex-based prediction: ~76%)
- Top leaderboard ~82% (not much room above baseline)

Our 77.5% LB score is **slightly above the naive baseline** but below what optimized models achieve. This suggests we have room for improvement through:
1. Better feature engineering that generalizes
2. Simpler models that don't overfit
3. Understanding the test set distribution better
