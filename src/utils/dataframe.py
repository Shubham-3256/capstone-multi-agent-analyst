"""Utility functions for analyzing and extracting profiles from Pandas DataFrames."""

from typing import Any

import pandas as pd


def get_dataframe_shape(df: pd.DataFrame) -> dict:
    """Get the row and column counts of a DataFrame.

    Args:
        df: Input Pandas DataFrame.

    Returns:
        dict: Keys "rows" and "columns" mapped to counts.
    """
    rows, cols = df.shape
    return {"rows": rows, "columns": cols}


def get_memory_footprint(df: pd.DataFrame) -> int:
    """Calculate total RAM footprint of a DataFrame in bytes.

    Args:
        df: Input Pandas DataFrame.

    Returns:
        int: Total memory usage in bytes.
    """
    return int(df.memory_usage(deep=True).sum())


def get_duplicate_count(df: pd.DataFrame) -> int:
    """Count number of duplicated rows in a DataFrame.

    Args:
        df: Input Pandas DataFrame.

    Returns:
        int: Number of duplicate rows.
    """
    return int(df.duplicated().sum())


def get_missing_summary(df: pd.DataFrame) -> dict:
    """Extract missing values counts and percentages per column.

    Args:
        df: Input Pandas DataFrame.

    Returns:
        dict: Keys "total_missing" and "column_missing" (mapping columns to details).
    """
    missing_series = df.isnull().sum()
    total_missing = int(missing_series.sum())

    col_missing = {}
    total_rows = len(df)
    for col, val in missing_series.items():
        val_int = int(val)
        pct = (val_int / total_rows * 100.0) if total_rows > 0 else 0.0
        col_missing[str(col)] = {
            "count": val_int,
            "percentage": round(pct, 2)
        }

    return {
        "total_missing": total_missing,
        "column_missing": col_missing
    }


def extract_column_statistics(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Generate descriptive statistics maps for each column in a DataFrame.

    Args:
        df: Input Pandas DataFrame.

    Returns:
        Dict[str, Dict[str, Any]]: Column names mapping to stats properties.
    """
    stats_dict = {}
    for col in df.columns:
        col_name = str(col)
        series = df[col]

        # Determine stats based on inferred pandas data types
        col_stats: dict[str, Any] = {}

        if pd.api.types.is_numeric_dtype(series):
            # Compute numeric metrics, skipping NaNs
            desc = series.describe()
            col_stats = {
                "mean": float(desc.get("mean", 0.0)) if not pd.isna(desc.get("mean")) else 0.0,
                "std": float(desc.get("std", 0.0)) if not pd.isna(desc.get("std")) else 0.0,
                "min": float(desc.get("min", 0.0)) if not pd.isna(desc.get("min")) else 0.0,
                "25%": float(desc.get("25%", 0.0)) if not pd.isna(desc.get("25%")) else 0.0,
                "50%": float(desc.get("50%", 0.0)) if not pd.isna(desc.get("50%")) else 0.0,
                "75%": float(desc.get("75%", 0.0)) if not pd.isna(desc.get("75%")) else 0.0,
                "max": float(desc.get("max", 0.0)) if not pd.isna(desc.get("max")) else 0.0,
            }
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_stats = {
                "min": series.min().isoformat() if not pd.isna(series.min()) else None,
                "max": series.max().isoformat() if not pd.isna(series.max()) else None,
            }
        else:
            # Categorical / string types
            desc = series.describe()
            mode = series.mode()
            col_stats = {
                "top": str(mode.iloc[0]) if not mode.empty else None,
                "freq": int(desc.get("freq", 0)) if not pd.isna(desc.get("freq")) else 0,
            }

        stats_dict[col_name] = col_stats

    return stats_dict
