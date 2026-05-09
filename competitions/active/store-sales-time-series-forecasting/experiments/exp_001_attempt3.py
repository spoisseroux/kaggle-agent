import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna
import mlflow
from mlflow.tracking import MlflowClient
from competitions.active.store_sales_time_series_forecasting.data import load_train_test, load_stores, load_holidays, load_external_data
from competitions.active.store_sales_time_series_forecasting.preprocessing import preprocess_data
from competitions.active.store_sales_time_series_forecasting.utils import rmsle

def check_data_leakage(test_df, external_data):
    """Check for data leakage between test and external datasets."""
    try:
        external_columns = set(external_data.columns)
        test_columns = set(test_df.columns)
        overlapping_columns = external_columns.intersection(test_columns)
        if overlapping_columns:
            mlflow.log_metric("leakage_detected", 1)
            print(f"Data leakage detected in columns: {overlapping_columns}")
            return False
        mlflow.log_metric("leakage_detected", 0)
        return True
    except Exception as e:
        mlflow.log_metric("leakage_detected", 1)
        print(f"Error during leakage check: {str(e)}")
        return False

def objective(trial, X, y, cv_splits=5):
    """Optuna objective function with regularization parameters."""
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'lambda': trial.suggest_float('lambda', 1e-8, 1e2),
        'alpha': trial.suggest_float('alpha', 1e-8, 1e2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_estimators': 100
    }
    
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(params, dtrain, num_boost_round=100, 
                         early_stopping_rounds=10, 
                         evals=[(dval, 'val')], 
                         verbose_eval=False)
        
        preds = model.predict(dval)
        score = rmsle(y_val, preds)
        scores.append(score)
    
    return np.mean(scores)

def run_experiment():
    try:
        mlflow.set_experiment("exp_001_Regularization_and_Data_Leakage_Check")
        with mlflow.start_run(run_name="exp_001"):
            mlflow.log_param("experiment_id", "exp_001")
            
            # Load data
            train_df, test_df = load_train_test()
            stores_df = load_stores()
            holidays_df = load_holidays()
            external_df = load_external_data()
            
            # Preprocess data
            train_df = preprocess_data(train_df, stores_df, holidays_df)
            test_df = preprocess_data(test_df, stores_df, holidays_df)
            
            # Check for data leakage
            leakage_free = check_data_leakage(test_df, external_df)
            if not leakage_free:
                return {"cv_score": None, "lb_score": None, "success": False}
            
            # Prepare features and target
            X = train_df.drop(columns=['sales'])
            y = train_df['sales']
            
            # Hyperparameter tuning with Optuna
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, X, y), n_trials=50)
            
            # Best parameters
            best_params = study.best_params
            mlflow.log_params(best_params)
            
            # Final training with best parameters
            final_model = xgb.XGBRegressor(**best_params, n_estimators=1000)
            final_model.fit(X, y)
            
            # Cross-validation score
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                final_model.fit(X_train, y_train)
                preds = final_model.predict(X_val)
                cv_scores.append(rmsle(y_val, preds))
            
            cv_score = np.mean(cv_scores)
            mlflow.log_metric("cv_score", cv_score)
            
            # Generate submission
            test_pred = final_model.predict(test_df)
            submission_df = pd.DataFrame({
                'id': test_df.index,
                'sales': test_pred
            })
            submission_df.to_csv('submissions/exp_001_submission.csv', index=False)
            
            # Success criteria
            success = cv_score < 0.15  # Assuming LB is 0.0 and CV is 0.15
            return {"cv_score": cv_score, "lb_score": None, "success": success}
            
    except Exception as e:
        mlflow.log_metric("error", 1)
        print(f"Experiment failed: {str(e)}")
        return {"cv_score": None, "lb_score": None, "success": False}