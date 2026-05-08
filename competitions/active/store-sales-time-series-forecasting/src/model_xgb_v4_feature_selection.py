#!/usr/bin/env python3
"""XGBoost v4 - Feature selection (reduced feature set for better generalization)"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
import sys
from pathlib import Path

# Reduced feature set - removing potentially redundant features
SELECTED_FEATURES = [
    'Time', 'day_of_week', 'month', 'is_weekend',
    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 'Lag_14', 'Lag_28',
    'Roll_mean_7', 'Roll_mean_14', 'Roll_mean_30',
    'Roll_std_7', 'Roll_std_14', 'Roll_std_30',
    'onpromotion', 'onpromotion_lag1', 'is_holiday'
]

def load_and_prepare_data():
    """Load data and create reduced feature set"""
    # Import from model_xgb_v1 to reuse feature engineering
    sys.path.insert(0, str(Path(__file__).parent))
    from model_xgb_v1 import load_data, create_features
    
    train_df, test_df, holidays = load_data()
    train_df = create_features(train_df, holidays)
    test_df = create_features(test_df, holidays)
    
    return train_df, test_df

def main():
    print("XGBoost v4 - Feature Selection")
    print(f"Using {len(SELECTED_FEATURES)} features (vs 29 in v1)")
    print()
    
    train_df, test_df = load_and_prepare_data()
    
    # Split train/val
    val_size = int(len(train_df) * 0.1)
    val_df = train_df.tail(val_size).copy()
    train_df = train_df.head(len(train_df) - val_size).copy()
    
    # Prepare data with selected features only
    X_train = train_df[SELECTED_FEATURES]
    y_train = train_df['sales']
    X_val = val_df[SELECTED_FEATURES]
    y_val = val_df['sales']
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print()
    
    # Train XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'eval_metric': 'rmse',
        'seed': 42
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Evaluate
    y_pred = model.predict(dval)
    y_pred = np.maximum(y_pred, 0)  # No negative sales
    
    # Calculate RMSLE
    mask = y_val > 0
    cv_rmsle = np.sqrt(mean_squared_log_error(
        y_val[mask],
        y_pred[mask]
    ))
    
    print(f"\nValidation RMSLE: {cv_rmsle:.6f}")
    print(f"Best iteration: {model.best_iteration}")
    print()
    
    # Feature importance
    importance = model.get_score(importance_type='gain')
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 features by importance:")
    for feat, score in importance_sorted[:10]:
        print(f"  {feat}: {score:.2f}")
    
    print(f"\n✅ XGBoost v4 (feature selection) complete - CV RMSLE: {cv_rmsle:.6f}")
    print(f"   Features: {len(SELECTED_FEATURES)} (reduced from 29)")
    
    improvement = 0.321032 - cv_rmsle
    if improvement > 0:
        print(f"   Improvement: -{improvement:.6f} 🎉")
    else:
        print(f"   Change: +{abs(improvement):.6f}")

if __name__ == "__main__":
    main()
