# Public notebook insights — store-sales-time-series-forecasting
Generated: 2026-05-08T14:57:28.539552+00:00

## Exercise: Linear Regression With Time Series
**Ref:** ryanholbrook/exercise-linear-regression-with-time-series | **Votes:** 63371

### 1. **Key Features Engineered**  
- **Time dummy**: `Time` column created using `np.arange(len(book_sales.index))` (sequential integer encoding of time).  
- **Lag feature**: `Lag_1` column generated via `book_sales['Hardcover'].shift(1)` (1-period lag of target variable).  
- **Store sales preprocessing**: For `store_sales`, grouped daily average sales (`average_sales`) and used `Time` and `Lag_1` as features.  

---

### 2. **Model Architecture & Hyperparameters**  
- **Model**: Simple `LinearRegression` from `sklearn.linear_model`.  
- **Hyperparameters**: Default settings (no explicit tuning).  
- **Features used**: `Time` (trend) and `Lag_1` (serial dependence).  

---

### 3. **Preprocessing / Data Cleaning Tricks**  
- **Handling missing data**: Used `dropna()` when creating lag features and aligned `X` and `y` via `y.align(X, join='inner')` to remove misaligned rows.  
- **Data types**: Categorized `store_nbr` and `family` for memory efficiency.  
- **Indexing**: Set `date` as the index and converted to `Period` type for time-series alignment.  

---

### 4. **Competition-Specific Insights / Leaks**  
- **No explicit leaks**: The notebook is an educational example, not a full competition solution.  
- **Simplification**: Used aggregated daily average sales (`average_sales`) instead of per-store/family series.  
- **Focus**: Demonstrated linear regression concepts (e.g., interpreting lag coefficients) rather than optimizing for competition metrics.  

---

### 5. **CV / LB Scores**  
- **Not reported**: The notebook is part of a learning exercise, not a competition submission. No validation or leaderboard scores are provided.

---

## Exercise: Trend
**Ref:** ryanholbrook/exercise-trend | **Votes:** 36612

### 1. **Key Features Engineered**  
- **Polynomial trend features**: Created using `statsmodels.tsa.deterministic.DeterministicProcess` with `order=3` (cubic trend) and later `order=11` (high-order polynomial). Features include polynomial terms (e.g., `x1`, `x2`, ..., `x11`) and a constant term.  
- **Rolling mean for trend estimation**: A 365-day rolling mean (centered, min_periods=183) applied to `average_sales` to smooth the time series.  
- **Forecast features**: Generated via `dp.out_of_sample(steps=90)` for 90-day future predictions.  

---

### 2. **Model Architecture & Hyperparameters**  
- **Model**: Simple `LinearRegression` from scikit-learn.  
- **Hyperparameters**:  
  - `order=3` (cubic trend) or `order=11` (high-order polynomial) for `DeterministicProcess`.  
  - No other hyperparameters explicitly tuned.  

---

### 3. **Preprocessing / Data Cleaning Tricks**  
- **Data loading**:  
  - Read `train.csv` with specified dtypes (`store_nbr`, `family` as categories; `sales` as float32).  
  - Set `date` as the index, converted to `Period` type.  
- **Aggregation**: Grouped `store_sales` by `date` to compute `average_sales` (mean of `sales` per day).  
- **Smoothing**: Used rolling mean (365-day window) to estimate trends.  

---

### 4. **Competition-Specific Insights / Leaks**  
- **No explicit leaks**: The notebook is part of a course exercise, not a competition submission.  
- **Focus on methodology**: Demonstrates polynomial trend modeling and risks of overfitting with high-order polynomials (e.g., unstable forecasts).  

---

### 5. **CV / LB Scores**  
- **Not reported**: The notebook is an educational example, not a competition submission. No validation or leaderboard scores are provided.  

--- 

**Summary**:

---

## Exercise: Seasonality
**Ref:** ryanholbrook/exercise-seasonality | **Votes:** 30008

### 1. **Key Features Engineered**  
- **Seasonal Features**:  
  - Weekly season indicators (e.g., `week`, `dayofweek` as categorical features).  
  - Fourier features of order 4 for monthly seasonality (via `CalendarFourier`).  
- **Holiday Features**:  
  - Binary indicators for **national/regional holidays** (from `holidays_events.csv`), joined on `date` and filled with 0s.  

---

### 2. **Model Architecture & Hyperparameters**  
- **Model**: Simple **Linear Regression** (`sklearn.linear_model.LinearRegression`).  
- **Hyperparameters**: Default settings (no explicit tuning).  

---

### 3. **Preprocessing / Data Cleaning Tricks**  
- **Data Type Optimization**:  
  - Categorical columns (`store_nbr`, `family`, `type`, etc.) encoded as `category` dtype.  
  - `sales` converted to `float32` for memory efficiency.  
- **Date Handling**:  
  - `date` parsed as `datetime`, converted to `period['D']` for alignment.  
  - Index set to `['store_nbr', 'family', 'date']` for hierarchical structuring.  
- **Detrending/Deseasonalizing**:  
  - Subtract fitted values from the original series to remove seasonality.  

---

### 4. **Competition-Specific Insights / Leaks**  
- **Holiday Feature Engineering**:  
  - Used **national/regional holidays** (filtered to `2017-08-15`) as predictive features.  
  - Avoided leakage by not including future holidays.  
- **No Explicit Leaks**:  
  - No mention of using test data or future information.  

---

### 5. **CV / LB Scores**  
- **Not Reported**: The notebook is an **exercise/tutorial** (not a competition

---

## Exercise: Time Series as Features
**Ref:** ryanholbrook/exercise-time-series-as-features | **Votes:** 25045

### 1. **Key Features Engineered**  
- **Lag Features**: Created lagged values for deseasonalized sales (`sales_lag_1`, `sales_lag_2`, etc.) using `make_lags`.  
- **Onpromotion Features**: Used lagged and leading values of `onpromotion` (e.g., `onpromotion_lag_1`, `onpromotion_lead_1`) as potential predictors.  
- **Time Components**: Included Fourier terms (monthly, order=4) and deterministic trends (seasonal, order=1) via `CalendarFourier` and `DeterministicProcess`.  
- **Holiday Indicator**: Added `NewYearsDay` as a binary feature (dayofyear == 1).  

---

### 2. **Model Architecture & Hyperparameters**  
- **Model**: Linear Regression (`LinearRegression(fit_intercept=False)`).  
- **Features**: Combined time-based features (trend, seasonality, Fourier terms) and `onpromotion` lags/leads.  
- **No Complex Models**: Focused on linear modeling for interpretability, not deep learning or ensemble methods.  

---

### 3. **Preprocessing / Data Cleaning**  
- **Data Loading**: Read `train.csv` with optimized dtypes (`category` for `store_nbr`, `family`; `float32` for `sales`).  
- **Deseasonalization**: Used linear regression on Fourier terms and deterministic trends to remove seasonality from sales.  
- **Handling Missing Data**: Dropped days with `onpromotion <= 1` during lag/lead analysis.  
- **Indexing**: Set multi-index (`store_nbr`, `family`, `date`) for efficient grouping and aggregation.  

---

### 4. **Competition-Specific Insights**  
- **Avoiding Leaks**: Used `onpromotion` as a leading indicator (e.g., Tuesday’s promotions predict Monday’s sales) without lookahead leakage.  
- **Cyclic Patterns**: Focused on deseasonalized sales of "School and Office Supplies" to isolate cyclic behavior.  
- **Feature Selection**: Highlighted the importance of lagged `onpromotion` and Fourier terms for capturing non-seasonal cycles.  

---

### 5. **CV / LB Scores**  
- **Not Provided**: This is a course exercise notebook, not a competition submission. Scores are not explicitly reported.

---

## Exercise: Hybrid Models
**Ref:** ryanholbrook/exercise-hybrid-models | **Votes:** 21971

### 1. **Key Features Engineered**  
- **Time-based features**: `DeterministicProcess` (from `statsmodels`) generates polynomial trends (order=1) for linear regression (`X_1`).  
- **Categorical encoding**: `LabelEncoder` applied to `family` (product category) and `day` (day of the month) for XGBoost (`X_2`).  
- **Onpromotion**: Raw `onpromotion` count used as a feature in `X_2`.  
- **Stacked data**: `X_2` is stacked (long format) for XGBoost, with `family` encoded as integers.  

---

### 2. **Model Architecture & Hyperparameters**  
- **Hybrid model**: `BoostedHybrid` class combines:  
  - **Model 1**: `LinearRegression` (for trend).  
  - **Model 2**: `XGBRegressor` (for residuals).  
- **Alternative models**: Tested combinations like `Ridge` + `KNeighborsRegressor`.  
- **No explicit hyperparameter tuning** mentioned (defaults used).  

---

### 3. **Preprocessing / Data Cleaning**  
- **Date parsing**: Dates converted to `period('D')` and set as index.  
- **Label encoding**: `family` (product categories) and `day` of the month encoded as integers.  
- **Feature engineering**:  
  - `DeterministicProcess` creates polynomial time trends.  
  - `onpromotion` used as-is in XGBoost.  
- **Data stacking**: `X_2` is stacked (long format) for XGBoost.  

---

### 4. **Competition-Specific Insights / Leaks**  
- **No explicit leaks** mentioned.  
- **Focus on hybrid modeling**: Combines linear trend modeling (via `LinearRegression`) with tree-based residuals (via `XGBRegressor`).  
- **Course-based example**: Notebook is part of a Kaggle course, not a full competition submission.  

---

### 5. **CV / LB Scores**  
- **Not provided** in the truncated notebook content.  
- **No evaluation metrics** (e.g., RMSE) shown in the code or text.  

--- 

**