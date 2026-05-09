"""Model building and training tools for agents.

Validated functions for model training, evaluation, and ensembling:
1. Model selection helper
2. Cross-validation training
3. Ensemble builder
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    log_loss,
)
import logging

log = logging.getLogger(__name__)


def select_model(
    problem_type: str,
    eval_metric: Optional[str] = None,
    prefer: Optional[str] = None,
) -> Any:
    """
    Select appropriate model for problem type.

    Args:
        problem_type: "binary", "multiclass", "regression", "time_series"
        eval_metric: Evaluation metric (helps select model)
        prefer: Preferred model family ("lgbm", "xgb", "catboost")

    Returns:
        Instantiated model

    Example:
        model = select_model("regression", eval_metric="rmse", prefer="lgbm")
    """
    try:
        if prefer == "lgbm" or prefer is None:
            import lightgbm as lgb

            if problem_type in ["binary", "multiclass"]:
                model = lgb.LGBMClassifier(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=7,
                    num_leaves=31,
                    random_state=42,
                    verbose=-1,
                )
            else:  # regression
                model = lgb.LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=7,
                    num_leaves=31,
                    random_state=42,
                    verbose=-1,
                )

        elif prefer == "xgb":
            import xgboost as xgb

            if problem_type in ["binary", "multiclass"]:
                model = xgb.XGBClassifier(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=7,
                    random_state=42,
                    verbosity=0,
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=7,
                    random_state=42,
                    verbosity=0,
                )

        elif prefer == "catboost":
            import catboost as cb

            if problem_type in ["binary", "multiclass"]:
                model = cb.CatBoostClassifier(
                    iterations=1000,
                    learning_rate=0.05,
                    depth=7,
                    random_state=42,
                    verbose=False,
                )
            else:
                model = cb.CatBoostRegressor(
                    iterations=1000,
                    learning_rate=0.05,
                    depth=7,
                    random_state=42,
                    verbose=False,
                )

        else:
            raise ValueError(f"Unknown model preference: {prefer}")

        log.info(f"Selected {type(model).__name__} for {problem_type}")
        return model

    except ImportError as e:
        log.error(f"Failed to import model library: {e}")
        raise


def train_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    problem_type: str,
    eval_metric: Optional[str] = None,
    n_splits: int = 5,
    time_series: bool = False,
    early_stopping_rounds: Optional[int] = 50,
) -> Dict[str, Any]:
    """
    Train model with cross-validation.

    Args:
        X: Feature matrix
        y: Target vector
        model: Model instance
        problem_type: "binary", "multiclass", "regression"
        eval_metric: Evaluation metric name
        n_splits: Number of CV folds
        time_series: Use TimeSeriesSplit instead of KFold
        early_stopping_rounds: Early stopping (for gradient boosting)

    Returns:
        Dict with:
        - model: trained model (on full data)
        - cv_scores: cross-validation scores
        - mean_cv: mean CV score
        - std_cv: std CV score
        - oof_predictions: out-of-fold predictions
        - feature_importance: feature importances (if available)
    """
    log.info(f"Training {type(model).__name__} with {n_splits}-fold CV")

    # Select CV strategy
    if time_series:
        cv = TimeSeriesSplit(n_splits=n_splits)
    elif problem_type == "multiclass" or problem_type == "binary":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Select scoring metric
    scoring = _get_scoring_func(problem_type, eval_metric)

    # Cross-validation
    cv_scores = []
    oof_predictions = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train
        if early_stopping_rounds and hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
            # Gradient boosting with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=_get_eval_metric_name(eval_metric),
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        # Predict
        if problem_type in ["binary", "multiclass"]:
            preds = model.predict_proba(X_val)
            if problem_type == "binary":
                preds = preds[:, 1]  # Take positive class probability
        else:
            preds = model.predict(X_val)

        oof_predictions[val_idx] = preds if problem_type == "regression" or problem_type == "binary" else preds.argmax(axis=1)

        # Score
        score = scoring(y_val, preds)
        cv_scores.append(score)

        log.info(f"Fold {fold}: {score:.4f}")

    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    log.info(f"CV Score: {mean_cv:.4f} ± {std_cv:.4f}")

    # Train on full data
    model.fit(X, y)

    # Feature importance
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(X.columns, model.feature_importances_))

    return {
        "model": model,
        "cv_scores": cv_scores,
        "mean_cv": mean_cv,
        "std_cv": std_cv,
        "oof_predictions": oof_predictions,
        "feature_importance": feature_importance,
    }


def build_ensemble(
    models: List[Dict[str, Any]],
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    method: str = "weighted_average",
    weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Build ensemble from multiple models.

    Args:
        models: List of trained model dicts (from train_with_cv)
        X: Feature matrix for predictions
        y: Target (optional, for scoring)
        method: "weighted_average", "rank_average", or "stacking"
        weights: Custom weights (optional, default: based on CV scores)

    Returns:
        Dict with:
        - predictions: ensemble predictions
        - weights: model weights used
        - individual_predictions: predictions from each model
    """
    log.info(f"Building ensemble from {len(models)} models using {method}")

    # Get predictions from each model
    all_preds = []
    cv_scores = []

    for model_dict in models:
        model = model_dict["model"]
        cv_score = model_dict["mean_cv"]

        preds = model.predict(X)
        all_preds.append(preds)
        cv_scores.append(cv_score)

    all_preds = np.array(all_preds)

    # Calculate weights
    if weights is None:
        if method == "weighted_average":
            # Weight by CV score (higher is better for accuracy, lower for error metrics)
            # Assume higher is better for now
            weights = np.array(cv_scores) / sum(cv_scores)
        else:
            # Equal weights
            weights = np.ones(len(models)) / len(models)

    weights = np.array(weights)

    # Ensemble predictions
    if method == "weighted_average":
        ensemble_preds = np.average(all_preds, axis=0, weights=weights)

    elif method == "rank_average":
        # Rank-based ensemble
        ranked = np.array([np.argsort(np.argsort(p)) for p in all_preds])
        ensemble_preds = np.average(ranked, axis=0, weights=weights)

    elif method == "stacking":
        # Simple stacking (average for now, could train meta-model)
        ensemble_preds = np.mean(all_preds, axis=0)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    log.info(f"Ensemble weights: {weights}")

    result = {
        "predictions": ensemble_preds,
        "weights": weights.tolist(),
        "individual_predictions": all_preds.tolist(),
    }

    # Score if target provided
    if y is not None:
        score = mean_squared_error(y, ensemble_preds, squared=False)
        log.info(f"Ensemble RMSE: {score:.4f}")
        result["score"] = score

    return result


def _get_scoring_func(problem_type: str, eval_metric: Optional[str] = None):
    """Get scoring function for CV."""
    if eval_metric == "rmse" or (problem_type == "regression" and eval_metric is None):
        return lambda y_true, y_pred: -mean_squared_error(y_true, y_pred, squared=False)

    elif eval_metric == "mae":
        return lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred)

    elif eval_metric == "r2":
        return r2_score

    elif eval_metric == "auc" or problem_type == "binary":
        return roc_auc_score

    elif eval_metric == "logloss":
        return lambda y_true, y_pred: -log_loss(y_true, y_pred)

    elif problem_type == "multiclass":
        return accuracy_score

    else:
        # Default: negative MSE for regression, accuracy for classification
        if problem_type == "regression":
            return lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
        else:
            return accuracy_score


def _get_eval_metric_name(eval_metric: Optional[str]) -> str:
    """Convert metric name to library-specific name."""
    if eval_metric in ["rmse", "mse"]:
        return "rmse"
    elif eval_metric == "mae":
        return "mae"
    elif eval_metric == "auc":
        return "auc"
    elif eval_metric == "logloss":
        return "logloss"
    else:
        return "rmse"  # default
