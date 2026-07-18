"""Profiling engine for evaluating dataset statistics, correlations, tasks and recommendations."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.utils.dataframe import (
    get_duplicate_count,
    get_memory_footprint,
    get_missing_summary,
    extract_column_statistics,
)
from src.agents.data_intelligence.models import ColumnProfile, DatasetProfile, Recommendation

logger = get_logger(__name__)


class Profiler:
    """Profiler class to extract statistical features and model task suggestions from DataFrames."""

    @staticmethod
    def profile(df: pd.DataFrame, target_column: Optional[str] = None) -> DatasetProfile:
        """Run deep statistical profile evaluations on a Pandas DataFrame.

        Args:
            df: Cleaned dataset to profile.
            target_column: Optional label prediction target column.

        Returns:
            DatasetProfile: The generated statistical profile.
        """
        logger.info("Profiler: Starting dataset profile evaluations...")
        rows, cols = df.shape

        # 1. Audits memory and duplicates
        mem_usage = get_memory_footprint(df)
        duplicates = get_duplicate_count(df)
        missing_dict = get_missing_summary(df)
        stats_dict = extract_column_statistics(df)

        # 2. Build column profile lists
        columns_profile: Dict[str, ColumnProfile] = {}
        recommendations: List[Recommendation] = []

        for col in df.columns:
            col_name = str(col)
            series = df[col]

            null_count = int(series.isnull().sum())
            null_pct = (null_count / rows) * 100.0 if rows > 0 else 0.0
            unique_count = int(series.nunique())

            num_summary = None
            cat_summary = None
            date_summary = None

            col_stats = stats_dict.get(col_name, {})

            # Distribute statistics to type-specific fields
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                num_summary = {k: float(v) for k, v in col_stats.items()}
            elif pd.api.types.is_datetime64_any_dtype(series):
                date_summary = {k: str(v) for k, v in col_stats.items()}
            else:
                cat_summary = {
                    "top": str(col_stats.get("top", "")),
                    "freq": int(col_stats.get("freq", 0))
                }

            columns_profile[col_name] = ColumnProfile(
                name=col_name,
                dtype=str(series.dtype),
                unique_count=unique_count,
                null_count=null_count,
                null_percentage=round(null_pct, 2),
                numeric_summary=num_summary,
                categorical_summary=cat_summary,
                date_summary=date_summary
            )

            # Generate column-specific recommendations
            if unique_count == 1:
                recommendations.append(Recommendation(
                    title="Constant column detected",
                    description=f"Column '{col_name}' has a single constant value. It provides zero variance and should be dropped.",
                    severity="warning",
                    column=col_name
                ))
            if null_pct > 30.0:
                recommendations.append(Recommendation(
                    title="High missing values ratio",
                    description=f"Column '{col_name}' has {round(null_pct, 2)}% missing values. Evaluate imputation stability or drop.",
                    severity="warning",
                    column=col_name
                ))

        # 3. Pearson Correlation matrix calculation
        corr_matrix: Dict[str, Dict[str, float]] = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            try:
                corr = df[numeric_cols].corr(method="pearson").fillna(0.0)
                corr_matrix = {
                    str(c): {str(k): float(v) for k, v in row.items()}
                    for c, row in corr.to_dict(orient="index").items()
                }
                
                # Check for high multicollinearity (> 0.85)
                seen_pairs = set()
                for c1 in corr.columns:
                    for c2 in corr.columns:
                        if c1 != c2 and (c2, c1) not in seen_pairs:
                            val = float(corr.at[c1, c2])
                            if abs(val) > 0.85:
                                recommendations.append(Recommendation(
                                    title="Strong multicollinearity",
                                    description=f"Columns '{c1}' and '{c2}' are strongly correlated (Pearson R={round(val, 3)}). Consider regularizations or dropping one.",
                                    severity="info",
                                    column=c1
                                ))
                                seen_pairs.add((c1, c2))
            except Exception as e:
                logger.warning(f"Failed to calculate Pearson correlation matrix: {e}")

        # 4. Target variable analysis and recommended ML tasks
        recommended_task = "clustering"  # Default if no target
        target_dist = None

        if target_column:
            if target_column in df.columns:
                target_series = df[target_column]
                target_dist = {str(k): int(v) for k, v in target_series.value_counts(dropna=False).items()}
                
                # Infer ML task type
                unique_vals = target_series.nunique()
                if pd.api.types.is_numeric_dtype(target_series) and unique_vals > 10:
                    recommended_task = "regression"
                else:
                    recommended_task = "classification"

                # Check for class imbalance in classification targets
                if recommended_task == "classification" and unique_vals >= 2:
                    counts = target_series.value_counts()
                    max_class = counts.max()
                    min_class = counts.min()
                    total = counts.sum()
                    
                    if total > 0 and (max_class / min_class) > 4.0:
                        recommendations.append(Recommendation(
                            title="Highly imbalanced target",
                            description=(
                                f"Target column '{target_column}' is imbalanced. "
                                f"Skew ratio is {round(max_class / min_class, 2)}:1 (Max class: {max_class}, Min class: {min_class}). "
                                "Stratify class splits and apply balanced metric evaluation."
                            ),
                            severity="warning",
                            column=target_column
                        ))
            else:
                recommended_task = "unknown"
                logger.warning(f"Target column '{target_column}' specified but missing in DataFrame columns.")

        logger.info(f"Profiler: Profiling complete. Recommended ML Task: {recommended_task.upper()}")
        return DatasetProfile(
            row_count=rows,
            column_count=cols,
            memory_usage_bytes=mem_usage,
            columns=columns_profile,
            correlation_matrix=corr_matrix,
            target_distribution=target_dist,
            recommended_ml_task=recommended_task,
            recommendations=recommendations
        )
