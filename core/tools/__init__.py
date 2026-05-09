"""ML tools library for agents.

Validated functions for:
- Data cleaning (7 tools)
- Feature engineering (11 tools)
- Model building (3 tools)
"""
from core.tools.cleaning import (
    handle_missing_values,
    detect_outliers,
    remove_duplicates,
    validate_dtypes,
    fix_inconsistent_values,
    parse_dates,
    standardize_strings,
)

from core.tools.features import (
    one_hot_encode,
    target_encode,
    frequency_encode,
    standard_scale,
    minmax_scale,
    robust_scale,
    correlation_analysis,
    create_lag_features,
    create_rolling_features,
    create_polynomial_features,
    create_bins,
)

from core.tools.modeling import (
    select_model,
    train_with_cv,
    build_ensemble,
)

# Organized by category for agents
TOOLS_LIBRARY = {
    "data_cleaning": [
        {"name": "handle_missing_values", "func": handle_missing_values, "description": "Handle missing values with various strategies (mean, median, mode, drop)"},
        {"name": "detect_outliers", "func": detect_outliers, "description": "Detect outliers using IQR or Z-score"},
        {"name": "remove_duplicates", "func": remove_duplicates, "description": "Remove duplicate rows"},
        {"name": "validate_dtypes", "func": validate_dtypes, "description": "Validate and coerce column data types"},
        {"name": "fix_inconsistent_values", "func": fix_inconsistent_values, "description": "Fix inconsistent categorical values"},
        {"name": "parse_dates", "func": parse_dates, "description": "Parse date columns to datetime"},
        {"name": "standardize_strings", "func": standardize_strings, "description": "Standardize string columns (lowercase, strip, etc.)"},
    ],
    "feature_engineering": [
        {"name": "one_hot_encode", "func": one_hot_encode, "description": "One-hot encode categorical columns"},
        {"name": "target_encode", "func": target_encode, "description": "Target encoding with smoothing"},
        {"name": "frequency_encode", "func": frequency_encode, "description": "Frequency/count encoding"},
        {"name": "standard_scale", "func": standard_scale, "description": "Standard scaling (mean=0, std=1)"},
        {"name": "minmax_scale", "func": minmax_scale, "description": "MinMax scaling to [0, 1]"},
        {"name": "robust_scale", "func": robust_scale, "description": "Robust scaling (median, IQR)"},
        {"name": "correlation_analysis", "func": correlation_analysis, "description": "Analyze correlations and find redundant features"},
        {"name": "create_lag_features", "func": create_lag_features, "description": "Create lag features for time series"},
        {"name": "create_rolling_features", "func": create_rolling_features, "description": "Create rolling aggregation features"},
        {"name": "create_polynomial_features", "func": create_polynomial_features, "description": "Create polynomial and interaction features"},
        {"name": "create_bins", "func": create_bins, "description": "Discretize continuous variables into bins"},
    ],
    "modeling": [
        {"name": "select_model", "func": select_model, "description": "Select appropriate model for problem type"},
        {"name": "train_with_cv", "func": train_with_cv, "description": "Train model with cross-validation"},
        {"name": "build_ensemble", "func": build_ensemble, "description": "Build ensemble from multiple models"},
    ],
}

__all__ = [
    # Cleaning
    "handle_missing_values",
    "detect_outliers",
    "remove_duplicates",
    "validate_dtypes",
    "fix_inconsistent_values",
    "parse_dates",
    "standardize_strings",
    # Features
    "one_hot_encode",
    "target_encode",
    "frequency_encode",
    "standard_scale",
    "minmax_scale",
    "robust_scale",
    "correlation_analysis",
    "create_lag_features",
    "create_rolling_features",
    "create_polynomial_features",
    "create_bins",
    # Modeling
    "select_model",
    "train_with_cv",
    "build_ensemble",
    # Library dict
    "TOOLS_LIBRARY",
]
