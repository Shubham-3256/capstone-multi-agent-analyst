"""Analytical caption generator generating dynamic descriptions of visualizations."""

from typing import List, Optional
import pandas as pd
import numpy as np

from src.core.logger import get_logger
from src.agents.visualization.models import ChartCaption
from src.agents.machine_learning.models import Leaderboard, FeatureImportance

logger = get_logger(__name__)


class CaptionGenerator:
    """Generates structured, task-specific analytical summaries and details captions."""

    @staticmethod
    def generate_missing_heatmap_caption(df: pd.DataFrame) -> ChartCaption:
        """Generate caption for the missing value heatmap.

        Args:
            df: Target dataset.

        Returns:
            ChartCaption: Generated caption.
        """
        null_counts = df.isna().sum()
        total_nulls = null_counts.sum()
        
        if total_nulls == 0:
            summary = "The dataset has complete values without missing records."
            details = "All columns have 100% data density, requiring no imputation blocks."
        else:
            highest_col = null_counts.idxmax()
            highest_val = null_counts.max()
            pct = round((highest_val / len(df)) * 100, 2)
            summary = f"Detected missing values in the dataset (total nulls = {total_nulls})."
            details = f"Column '{highest_col}' has the highest missing cells: {highest_val} ({pct}% of rows)."

        return ChartCaption(summary=summary, details=details)

    @staticmethod
    def generate_correlation_heatmap_caption(df: pd.DataFrame) -> ChartCaption:
        """Generate caption for the correlation heatmap.

        Args:
            df: Target dataset.

        Returns:
            ChartCaption: Generated caption.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return ChartCaption(
                summary="Correlation analysis skipped.",
                details="Insufficient numeric features to compute correlation matrix."
            )

        corr = numeric_df.corr().abs()
        corr_values = corr.to_numpy(copy=True)
        # Find highest correlation excluding diagonal
        np.fill_diagonal(corr_values, 0)
        max_idx = corr_values.argmax()
        r, c = np.unravel_index(max_idx, corr_values.shape)
        
        max_val = corr_values[r, c]
        col1, col2 = corr.index[r], corr.columns[c]

        if max_val > 0.5:
            summary = f"Strong linear association observed between numeric columns."
            details = f"Pearson correlation coefficient R={round(max_val, 3)} for pairs '{col1}' and '{col2}'."
        else:
            summary = "Weak correlations observed across numeric variables."
            details = f"Highest Pearson correlation coefficient R={round(max_val, 3)} between '{col1}' and '{col2}'."

        return ChartCaption(summary=summary, details=details)

    @staticmethod
    def generate_distribution_caption(df: pd.DataFrame, col: str) -> ChartCaption:
        """Generate caption for single feature distributions.

        Args:
            df: Target dataset.
            col: Target column name.

        Returns:
            ChartCaption: Generated caption.
        """
        if col not in df.columns:
            return ChartCaption(summary="Distribution skipped.", details="Column not found.")
            
        series = df[col].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_numeric:
            skew = series.skew()
            skew_desc = "positively skewed" if skew > 1 else ("negatively skewed" if skew < -1 else "symmetric")
            summary = f"Numeric distribution metrics for feature '{col}'."
            details = f"The variable shows a {skew_desc} shape (skewness={round(skew, 3)}, mean={round(float(series.mean()), 3)})."
        else:
            top_val = series.mode().iloc[0] if not series.mode().empty else "N/A"
            top_pct = round((series == top_val).sum() / len(series) * 100, 2)
            summary = f"Categorical frequencies mapping for feature '{col}'."
            details = f"The value '{top_val}' is the most frequent label, representing {top_pct}% of cells."

        return ChartCaption(summary=summary, details=details)

    @staticmethod
    def generate_leaderboard_caption(leaderboard: Leaderboard, task_type: str) -> ChartCaption:
        """Generate caption comparing scores on the leaderboard.

        Args:
            leaderboard: AutoML leaderboard.
            task_type: Task type.

        Returns:
            ChartCaption: Generated caption.
        """
        if not leaderboard.entries:
            return ChartCaption(summary="Leaderboard empty.", details="No candidate models evaluated.")

        best_entry = leaderboard.entries[0]
        metric_name = "macro F1 score" if task_type == "classification" else "RMSE"
        
        summary = f"The best performing algorithm is '{best_entry.model_name}'."
        details = f"It achieved validation {metric_name} of {round(best_entry.score, 4)} on test subsets splits."
        
        return ChartCaption(summary=summary, details=details)

    @staticmethod
    def generate_importance_caption(importances: List[FeatureImportance]) -> ChartCaption:
        """Generate caption ranking feature importances.

        Args:
            importances: Ranked feature importances.

        Returns:
            ChartCaption: Generated caption.
        """
        if not importances:
            return ChartCaption(summary="No importances extracted.", details="Feature importance profiles are empty.")

        best_feat = importances[0]
        summary = f"The feature column '{best_feat.column}' has the highest predictive value."
        details = f"Calculated model coefficient score: {round(best_feat.importance, 4)}."

        return ChartCaption(summary=summary, details=details)
