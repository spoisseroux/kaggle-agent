"""Feature engineering tools for agents.

Validated functions for common feature engineering operations:
1. One-hot encoding
2. Target encoding
3. Frequency encoding
4. Standard scaling
5. MinMax scaling
6. Robust scaling
7. Correlation analysis
8. Lag features (time series)
9. Rolling aggregations (time series)
10. Polynomial features
11. Binning/discretization
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
import logging

log = logging.getLogger(__name__)


def one_hot_encode(
    df: pd.DataFrame,
    columns: List[str],
    drop_first: bool = True,
    prefix_sep: str = "_",
) -> pd.DataFrame:
    """
    One-hot encode categorical columns.

    Args:
        df: Input dataframe
        columns: Columns to encode
        drop_first: Drop first category (avoid multicollinearity)
        prefix_sep: Separator for new column names

    Returns:
        Dataframe with one-hot encoded columns
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, prefix_sep=prefix_sep, drop_first=drop_first)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
            log.info(f"One-hot encoded {col} into {len(dummies.columns)} columns")

    return df


def target_encode(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    columns: List[str],
    target_col: str,
    smoothing: float = 1.0,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Target encoding (mean encoding) with smoothing.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe (optional)
        columns: Columns to encode
        target_col: Target column name
        smoothing: Smoothing parameter (higher = more regularization)

    Returns:
        (train_encoded, test_encoded)
    """
    train_df = train_df.copy()
    test_df = test_df.copy() if test_df is not None else None

    global_mean = train_df[target_col].mean()

    for col in columns:
        if col not in train_df.columns:
            continue

        # Calculate smoothed means
        agg = train_df.groupby(col)[target_col].agg(['mean', 'count'])
        smoothed_mean = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)

        # Apply to train
        train_df[f"{col}_target_enc"] = train_df[col].map(smoothed_mean).fillna(global_mean)

        # Apply to test
        if test_df is not None:
            test_df[f"{col}_target_enc"] = test_df[col].map(smoothed_mean).fillna(global_mean)

        log.info(f"Target encoded {col}")

    return train_df, test_df


def frequency_encode(
    df: pd.DataFrame,
    columns: List[str],
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Frequency encoding (count of each category).

    Args:
        df: Input dataframe
        columns: Columns to encode
        normalize: Normalize counts to [0, 1]

    Returns:
        Dataframe with frequency-encoded columns
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            freq = df[col].value_counts()

            if normalize:
                freq = freq / len(df)

            df[f"{col}_freq"] = df[col].map(freq)
            log.info(f"Frequency encoded {col}")

    return df


def standard_scale(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    columns: List[str],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], StandardScaler]:
    """
    Standard scaling (mean=0, std=1).

    Args:
        train_df: Training dataframe
        test_df: Test dataframe (optional)
        columns: Columns to scale

    Returns:
        (train_scaled, test_scaled, scaler)
    """
    train_df = train_df.copy()
    test_df = test_df.copy() if test_df is not None else None

    scaler = StandardScaler()
    train_df[columns] = scaler.fit_transform(train_df[columns])

    if test_df is not None:
        test_df[columns] = scaler.transform(test_df[columns])

    log.info(f"Standard scaled {len(columns)} columns")

    return train_df, test_df, scaler


def minmax_scale(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    columns: List[str],
    feature_range: Tuple[float, float] = (0, 1),
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], MinMaxScaler]:
    """
    MinMax scaling to [0, 1] or custom range.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe (optional)
        columns: Columns to scale
        feature_range: (min, max) range

    Returns:
        (train_scaled, test_scaled, scaler)
    """
    train_df = train_df.copy()
    test_df = test_df.copy() if test_df is not None else None

    scaler = MinMaxScaler(feature_range=feature_range)
    train_df[columns] = scaler.fit_transform(train_df[columns])

    if test_df is not None:
        test_df[columns] = scaler.transform(test_df[columns])

    log.info(f"MinMax scaled {len(columns)} columns to {feature_range}")

    return train_df, test_df, scaler


def robust_scale(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    columns: List[str],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], RobustScaler]:
    """
    Robust scaling (using median and IQR, resistant to outliers).

    Args:
        train_df: Training dataframe
        test_df: Test dataframe (optional)
        columns: Columns to scale

    Returns:
        (train_scaled, test_scaled, scaler)
    """
    train_df = train_df.copy()
    test_df = test_df.copy() if test_df is not None else None

    scaler = RobustScaler()
    train_df[columns] = scaler.fit_transform(train_df[columns])

    if test_df is not None:
        test_df[columns] = scaler.transform(test_df[columns])

    log.info(f"Robust scaled {len(columns)} columns")

    return train_df, test_df, scaler


def correlation_analysis(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    threshold: float = 0.95,
    method: str = "pearson",
) -> Dict[str, Any]:
    """
    Analyze correlations and identify redundant features.

    Args:
        df: Input dataframe
        target_col: Target column (optional)
        threshold: Correlation threshold for redundancy
        method: "pearson", "spearman", or "kendall"

    Returns:
        Dict with:
        - corr_matrix: correlation matrix
        - high_corr_pairs: [(col1, col2, corr), ...] with |corr| > threshold
        - redundant_features: columns to consider dropping
        - target_correlations: correlations with target (if provided)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr(method=method)

    # Find high correlation pairs
    high_corr_pairs = []
    redundant = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]

            if abs(corr) > threshold:
                high_corr_pairs.append((col1, col2, corr))
                # Mark second column as redundant
                if col1 != target_col:  # Don't drop target
                    redundant.add(col2)

    # Target correlations
    target_corrs = {}
    if target_col and target_col in df.columns:
        target_corrs = corr_matrix[target_col].drop(target_col).to_dict()

    log.info(f"Found {len(high_corr_pairs)} high-correlation pairs (>{threshold})")
    log.info(f"Redundant features: {len(redundant)}")

    return {
        "corr_matrix": corr_matrix,
        "high_corr_pairs": high_corr_pairs,
        "redundant_features": list(redundant),
        "target_correlations": target_corrs,
    }


def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create lag features for time series.

    Args:
        df: Input dataframe (must be sorted by time)
        columns: Columns to create lags for
        lags: List of lag periods (e.g., [1, 7, 14])
        group_by: Group by column (e.g., store_id for multi-series)

    Returns:
        Dataframe with lag features
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        for lag in lags:
            if group_by:
                df[f"{col}_lag_{lag}"] = df.groupby(group_by)[col].shift(lag)
            else:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

            log.info(f"Created {col}_lag_{lag}")

    return df


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    functions: List[str] = ["mean", "std", "min", "max"],
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create rolling aggregation features.

    Args:
        df: Input dataframe (must be sorted by time)
        columns: Columns to aggregate
        windows: Window sizes (e.g., [7, 14, 30])
        functions: Aggregation functions
        group_by: Group by column

    Returns:
        Dataframe with rolling features
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        for window in windows:
            for func in functions:
                if group_by:
                    rolled = df.groupby(group_by)[col].transform(
                        lambda x: x.rolling(window).agg(func)
                    )
                else:
                    rolled = df[col].rolling(window).agg(func)

                df[f"{col}_roll_{func}_{window}"] = rolled
                log.info(f"Created {col}_roll_{func}_{window}")

    return df


def create_polynomial_features(
    df: pd.DataFrame,
    columns: List[str],
    degree: int = 2,
    interaction_only: bool = False,
) -> pd.DataFrame:
    """
    Create polynomial and interaction features.

    Args:
        df: Input dataframe
        columns: Columns to use
        degree: Polynomial degree
        interaction_only: Only interactions (no x^2, x^3, etc.)

    Returns:
        Dataframe with polynomial features
    """
    df = df.copy()

    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False
    )

    poly_features = poly.fit_transform(df[columns])
    feature_names = poly.get_feature_names_out(columns)

    # Add new features (skip original columns)
    for i, name in enumerate(feature_names):
        if name not in columns:
            df[name] = poly_features[:, i]

    log.info(f"Created {len(feature_names) - len(columns)} polynomial features")

    return df


def create_bins(
    df: pd.DataFrame,
    column: str,
    bins: int = 10,
    strategy: str = "quantile",
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Discretize continuous variable into bins.

    Args:
        df: Input dataframe
        column: Column to bin
        bins: Number of bins
        strategy: "quantile", "uniform", or "kmeans"
        labels: Custom bin labels (optional)

    Returns:
        Dataframe with binned column
    """
    df = df.copy()

    if column not in df.columns:
        return df

    if strategy == "quantile":
        df[f"{column}_binned"] = pd.qcut(df[column], q=bins, labels=labels, duplicates="drop")
    elif strategy == "uniform":
        df[f"{column}_binned"] = pd.cut(df[column], bins=bins, labels=labels)
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    log.info(f"Binned {column} into {bins} bins using {strategy} strategy")

    return df
