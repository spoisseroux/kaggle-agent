# Experiment Plan — store-sales-time-series-forecasting
Generated: 2026-05-08T14:58:02.451735+00:00

## 1. Add More Lag and Rolling Features (priority 1)
**Description:** Enhance temporal dependencies with additional lag and rolling statistics
**Approach:** Add Lag_2, Lag_3, 7-day rolling mean/median to existing Time and Lag_1 features. Use LinearRegression with all features
**Expected CV delta:** +0.005
**Est. runtime:** 15 min

## 2. Incorporate Seasonal Time Features (priority 2)
**Description:** Add explicit seasonal/time-of-year information
**Approach:** Create month, day_of_week, is_holiday features from date. Combine with existing Time/Lag features in LinearRegression
**Expected CV delta:** +0.005
**Est. runtime:** 15 min

## 3. Switch to Random Forest with Feature Engineering (priority 3)
**Description:** Test non-linear model with existing feature set
**Approach:** Use RandomForestRegressor with max_depth=10, n_estimators=100 on Time, Lag_1, and seasonal features
**Expected CV delta:** +0.01
**Est. runtime:** 15 min

## 4. Apply Ridge Regression with L2 Regularization (priority 4)
**Description:** Test regularized linear model for better generalization
**Approach:** Use Ridge(alpha=0.1) with Time, Lag_1, and seasonal features
**Expected CV delta:** +0.005
**Est. runtime:** 15 min

## 5. Feature Interaction Engineering (priority 5)
**Description:** Add multiplicative interactions between key features
**Approach:** Create Time*Lag_1 interaction term. Use LinearRegression with original features + interaction
**Expected CV delta:** +0.005
**Est. runtime:** 15 min
