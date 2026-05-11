"""
Reusable DataFrame inspection and transformation helpers.
All functions are pure (no side effects) and accept/return pandas objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def infer_column_dtypes(df: pd.DataFrame) -> dict[str, str]:
    """
    Return a human-readable dtype mapping for each column.
    Maps numpy/pandas dtypes to simplified labels: numeric, categorical, datetime, boolean.
    """
    mapping: dict[str, str] = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            mapping[col] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            mapping[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(dtype):
            mapping[col] = "numeric"
        else:
            mapping[col] = "categorical"
    return mapping


def missing_value_report(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return per-column missing value counts and percentages.
    Only includes columns that have at least one missing value.
    """
    total = len(df)
    report: dict[str, Any] = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            report[col] = {
                "count": n_missing,
                "pct": round(n_missing / total * 100, 2),
            }
    return report


def cardinality_report(df: pd.DataFrame) -> dict[str, int]:
    """Return the number of unique values per column."""
    return {col: int(df[col].nunique()) for col in df.columns}


def detect_target_type(series: pd.Series) -> str:
    """
    Heuristic to decide whether a target column is classification or regression.
    Returns 'classification' if the column has ≤20 unique values or is non-numeric.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    n_unique = series.nunique()
    return "classification" if n_unique <= 20 else "regression"


def safe_sample(df: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
    """Return up to n rows as a list of dicts, safe for JSON serialisation."""
    sample = df.head(n).copy()
    # Replace NaN/Inf with None for JSON compatibility
    sample = sample.replace({np.nan: None, np.inf: None, -np.inf: None})
    return sample.to_dict(orient="records")


def column_stats(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return descriptive statistics for numeric columns and
    value-count summaries for categorical columns.
    """
    stats: dict[str, Any] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            stats[col] = {
                "type": "numeric",
                "mean": round(float(desc["mean"]), 4),
                "std": round(float(desc["std"]), 4),
                "min": float(desc["min"]),
                "max": float(desc["max"]),
                "median": float(df[col].median()),
            }
        else:
            top_values = df[col].value_counts().head(5).to_dict()
            stats[col] = {
                "type": "categorical",
                "top_values": {str(k): int(v) for k, v in top_values.items()},
            }
    return stats
