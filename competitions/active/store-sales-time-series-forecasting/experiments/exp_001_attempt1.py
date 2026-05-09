import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import mlflow
import mlflow.lightgbm
from mlflow.tracking.client import MlflowClient
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    try:
        mlflow.set_experiment("exp_001_LightGBM_Hyperparameter_Tuning")
        mlflow.start_run(run_name="LightGBM_Hyperparameter_Tuning")

        # Load data
        train_path = Path("data/store-sales-time-series-forecasting/train.csv")
        test_path = Path("data/store-sales-time-series-forecasting/test.csv")
        external_path = Path("data/xgb_v3_external_data.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        external_df = pd.read_csv(external_path) if external_path.exists() else pd.DataFrame()

        # Preprocess data
        def preprocess(df, is_train=True):
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['week'] = df['date'].dt.isocalendar().week
            df['dayofweek'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            df['onpromotion'] = df['onpromotion'].apply(lambda x: 1 if x else 0)
            return df

        train_df = preprocess(train_df)
        test_df = preprocess(test_df)

        # Merge external data
        if not external_df.empty:
            external_df['date'] = pd.to_datetime(external_df['date'])
            external_df = external_df.merge(train_df[['date', 'store', 'item']], on=['date', 'store', 'item'], how='left')
            train_df = train_df.merge(external_df, on=['date', 'store', 'item'], how='left')
            test_df = test_df.merge(external_df, on=['date', 'store', 'item'], how='left')

        # Feature engineering
        X_train = train_df.drop(columns=['sales', 'date'])
        y_train = train_df['sales']
        X_test = test_df.drop(columns=['date', 'sales'])

        # Preprocessing pipeline
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        # Define model and parameter grid
        model = LGBMRegressor(random_state=42)
        param_grid = {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'num_leaves': [31, 63],
            'n_estimators': [100, 200],
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.6, 0.8]
        }

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(model, param_grid, n_jobs=-1, cv=tscv, scoring='neg_mean_squared_error', n_iter=10)
        search.fit(X_train, y_train)

        # Log parameters and metrics
        mlflow.log_params(search.best_params_)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        mlflow.log_metric("train_mse", mse)

        #