# Titanic Top Notebook Insights

## Common Winning Approaches

### Feature Engineering
- **Title extraction**: Extract titles from Name column (Mr, Mrs, Miss, Master, etc.)
- **Family size**: Combine SibSp + Parch into FamilySize feature
- **IsAlone**: Binary feature for passengers traveling alone
- **Age binning**: Discretize age into groups (child, teen, adult, senior)
- **Fare binning**: Discretize fare into quantile-based groups
- **Cabin deck**: Extract deck letter from Cabin (A, B, C, etc.)

### Missing Values
- Age: Fill with median by Pclass + Sex groups
- Embarked: Fill with mode (S)
- Cabin: Create "Unknown" category or drop
- Fare: Fill with median by Pclass

### Models
- Random Forest, XGBoost, LightGBM most common
- Ensembles of 3-5 models typical
- Logistic regression as baseline

### Known Issues
- Small test set (~400 rows) - prone to overfitting
- External historical records available (can achieve 1.0 score)
- Tutorial competition - not representative of real-world ML
