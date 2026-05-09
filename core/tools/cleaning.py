"""Data cleaning tools for agents.

Validated functions for common data cleaning operations:
1. Handle missing values
2. Detect and handle outliers
3. Remove duplicates
4. Validate data types
5. Fix inconsistent values
6. Handle date/time parsing
7. Encode categorical variables
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging

log = logging.getLogger(__name__)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "auto",
    numeric_fill: Optional[float] = None,
    categorical_fill: Optional[str] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Handle missing values with various strategies.

    Args:
        df: Input dataframe
        strategy: "auto", "mean", "median", "mode", "drop", "forward_fill"
        numeric_fill: Value to fill numeric columns (overrides strategy)
        categorical_fill: Value to fill categorical columns (overrides strategy)
        threshold: Drop columns with missing % > threshold

    Returns:
        Dataframe with missing values handled
    """
    df = df.copy()

    # Drop columns with too many missing values
    missing_pct = df.isnull().sum() / len(df)
    drop_cols = missing_pct[missing_pct > threshold].index.tolist()

    if drop_cols:
        log.info(f"Dropping {len(drop_cols)} columns with >{threshold*100}% missing")
        df = df.drop(columns=drop_cols)

    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            if numeric_fill is not None:
                df[col].fillna(numeric_fill, inplace=True)
            elif strategy == "mean" or strategy == "auto":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "forward_fill":
                df[col].ffill(inplace=True)

    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        if df[col].isnull().any():
            if categorical_fill is not None:
                df[col].fillna(categorical_fill, inplace=True)
            elif strategy == "mode" or strategy == "auto":
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
                df[col].fillna(mode_val, inplace=True)
            elif strategy == "forward_fill":
                df[col].ffill(inplace=True)

    # Drop rows with any remaining nulls
    if strategy == "drop":
        df = df.dropna()

    return df


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Detect outliers using IQR or Z-score method.

    Args:
        df: Input dataframe
        columns: Columns to check (None = all numeric)
        method: "iqr" or "zscore"
        threshold: IQR multiplier (1.5 default) or Z-score threshold (3.0 recommended)

    Returns:
        Dataframe with outlier column added (True/False per row)
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_mask = pd.Series(False, index=df.index)

    for col in columns:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

        elif method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

        outlier_mask |= outliers

    df["is_outlier"] = outlier_mask

    return df


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """
    Remove duplicate rows.

    Args:
        df: Input dataframe
        subset: Columns to consider (None = all columns)
        keep: "first", "last", or False (remove all duplicates)

    Returns:
        Dataframe with duplicates removed
    """
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    after = len(df)

    if before > after:
        log.info(f"Removed {before - after} duplicate rows")

    return df


def validate_dtypes(
    df: pd.DataFrame,
    expected_types: Dict[str, str],
    coerce: bool = True,
) -> pd.DataFrame:
    """
    Validate and optionally coerce column dtypes.

    Args:
        df: Input dataframe
        expected_types: {column: dtype} mapping
        coerce: Attempt to convert types (vs raise error)

    Returns:
        Dataframe with validated types
    """
    df = df.copy()

    for col, expected_dtype in expected_types.items():
        if col not in df.columns:
            log.warning(f"Column {col} not found")
            continue

        if str(df[col].dtype) != expected_dtype:
            if coerce:
                try:
                    df[col] = df[col].astype(expected_dtype)
                    log.info(f"Converted {col} to {expected_dtype}")
                except Exception as e:
                    log.error(f"Failed to convert {col} to {expected_dtype}: {e}")
            else:
                raise TypeError(f"Column {col} has type {df[col].dtype}, expected {expected_dtype}")

    return df


def fix_inconsistent_values(
    df: pd.DataFrame,
    column: str,
    mapping: Dict[Any, Any],
) -> pd.DataFrame:
    """
    Fix inconsistent categorical values.

    Args:
        df: Input dataframe
        column: Column to fix
        mapping: {old_value: new_value}

    Returns:
        Dataframe with fixed values

    Example:
        fix_inconsistent_values(df, "gender", {"M": "Male", "F": "Female", "m": "Male"})
    """
    df = df.copy()

    if column in df.columns:
        df[column] = df[column].replace(mapping)
        log.info(f"Fixed inconsistent values in {column}")

    return df


def parse_dates(
    df: pd.DataFrame,
    date_columns: List[str],
    format: Optional[str] = None,
    errors: str = "coerce",
) -> pd.DataFrame:
    """
    Parse date columns to datetime.

    Args:
        df: Input dataframe
        date_columns: Columns to parse
        format: Date format string (None = infer)
        errors: "coerce", "raise", or "ignore"

    Returns:
        Dataframe with parsed dates
    """
    df = df.copy()

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format=format, errors=errors)
            log.info(f"Parsed {col} as datetime")

    return df


def standardize_strings(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lowercase: bool = True,
    strip: bool = True,
    remove_special: bool = False,
) -> pd.DataFrame:
    """
    Standardize string columns.

    Args:
        df: Input dataframe
        columns: Columns to standardize (None = all object columns)
        lowercase: Convert to lowercase
        strip: Strip whitespace
        remove_special: Remove special characters

    Returns:
        Dataframe with standardized strings
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()

    for col in columns:
        if col in df.columns:
            if lowercase:
                df[col] = df[col].str.lower()
            if strip:
                df[col] = df[col].str.strip()
            if remove_special:
                df[col] = df[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

    return df
