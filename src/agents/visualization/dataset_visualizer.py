"""Visualizer generating missing value, correlation matrices, and distributions plots."""

from typing import Any, List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from src.agents.visualization.figure_factory import FigureFactory
from src.core.logger import get_logger

logger = get_logger(__name__)


class DatasetVisualizer:
    """Generates figures auditing missing values, correlations, and feature distributions."""

    @staticmethod
    def get_valid_columns(df: pd.DataFrame) -> List[str]:
        """Filter out identifier, constant, and empty columns from a DataFrame."""
        valid_cols = []
        for col in df.columns:
            col_name = str(col)
            col_lower = col_name.lower()
            # 1. Identifier Check
            is_id = col_lower == "id" or col_lower.endswith("_id") or col_lower.startswith("id_")
            # 2. Constant Check
            is_constant = df[col].nunique() <= 1
            # 3. Empty Check
            is_empty = df[col].isna().all()
            
            if not (is_id or is_constant or is_empty):
                valid_cols.append(col_name)
        return valid_cols

    @staticmethod
    def generate_missing_heatmap(df: pd.DataFrame, theme_name: str = "corporate") -> dict[str, Any]:
        """Generate a heatmap showing missing values across features.

        Args:
            df: Input DataFrame.
            theme_name: Styling theme.

        Returns:
            Dict[str, Any]: Mapped figures containing 'matplotlib' and 'plotly' handles.
        """
        logger.info("DatasetVisualizer: Creating missingness heatmap...")
        
        valid_cols = DatasetVisualizer.get_valid_columns(df)
        plot_df = df[valid_cols] if valid_cols else df

        # 1. Matplotlib Heatmap
        fig_mpl, ax = FigureFactory.create_matplotlib_figure(
            title="Missing Value Matrix Heatmap",
            subtitle="White stripes indicate missing null values presence",
            xlabel="Features",
            ylabel="Observations (Rows)",
            theme_name=theme_name
        )
        # Use simple bool mask as numeric array for heatmap
        mask = plot_df.isna()
        sns.heatmap(mask, cmap="Blues", cbar=False, yticklabels=False, ax=ax)
        fig_mpl.tight_layout()

        # 2. Plotly Heatmap
        # Convert bool mask to numeric integers for Plotly color scale
        mask_numeric = mask.astype(int).values
        fig_plotly = FigureFactory.create_plotly_figure(
            title="Missing Value Matrix Heatmap",
            subtitle="Values equal to 1 indicate missing null cells",
            xlabel="Features",
            ylabel="Observations Index",
            theme_name=theme_name
        )
        fig_plotly.add_trace(go.Heatmap(
            z=mask_numeric,
            x=list(plot_df.columns),
            y=list(plot_df.index),
            colorscale="Blues",
            showscale=False
        ))

        return {"matplotlib": fig_mpl, "plotly": fig_plotly}

    @staticmethod
    def generate_correlation_heatmap(df: pd.DataFrame, theme_name: str = "corporate") -> dict[str, Any]:
        """Generate a correlation matrix heatmap for numeric features.

        Args:
            df: Input DataFrame.
            theme_name: Styling theme.

        Returns:
            Dict[str, Any]: Mapped figures.
        """
        logger.info("DatasetVisualizer: Creating correlation matrix heatmap...")
        
        valid_cols = DatasetVisualizer.get_valid_columns(df)
        plot_df = df[valid_cols] if valid_cols else df
        numeric_df = plot_df.select_dtypes(include=[np.number])

        if numeric_df.empty or numeric_df.shape[1] < 2:
            logger.warning("Insufficient numeric variables for correlation matrix. Creating dummy empty plots.")
            numeric_df = pd.DataFrame({"dummy_1": [1.0, 2.0], "dummy_2": [2.0, 1.0]})

        corr = numeric_df.corr().round(2)

        # 1. Matplotlib Correlation
        fig_mpl, ax = FigureFactory.create_matplotlib_figure(
            title="Pearson Correlation Heatmap",
            subtitle="Numeric variables correlation scores matrix",
            theme_name=theme_name
        )
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1.0, vmax=1.0, square=True, ax=ax)
        fig_mpl.tight_layout()

        # 2. Plotly Correlation
        fig_plotly = FigureFactory.create_plotly_figure(
            title="Pearson Correlation Heatmap",
            subtitle="Interactive correlation grid values matrix",
            theme_name=theme_name
        )
        fig_plotly.add_trace(go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu",
            text=corr.values.astype(str),
            texttemplate="%{text}",
            hoverongaps=False
        ))

        return {"matplotlib": fig_mpl, "plotly": fig_plotly}

    @staticmethod
    def generate_distribution_plot(
        df: pd.DataFrame,
        column_name: str,
        theme_name: str = "corporate"
    ) -> dict[str, Any]:
        """Generate distribution plots (histogram/kde) for a target feature column.

        Args:
            df: Input DataFrame.
            column_name: Name of column to plot.
            theme_name: Styling theme.

        Returns:
            Dict[str, Any]: Mapped figures.
        """
        logger.info(f"DatasetVisualizer: Plotting distribution for '{column_name}'...")

        # Guard if column not found
        col = column_name
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found. Selecting first column '{df.columns[0]}'.")
            col = df.columns[0]

        series = df[col].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_numeric:
            # 1. Matplotlib Histogram
            fig_mpl, ax = FigureFactory.create_matplotlib_figure(
                title=f"Numerical Distribution: {col}",
                subtitle=f"Histogram and KDE distribution curve for feature column '{col}'",
                xlabel=col,
                ylabel="Density",
                theme_name=theme_name
            )
            sns.histplot(series, kde=True, stat="density", ax=ax)
            fig_mpl.tight_layout()

            # 2. Plotly Histogram
            fig_plotly = FigureFactory.create_plotly_figure(
                title=f"Numerical Distribution: {col}",
                subtitle=f"Interactive histogram bars mapping for column '{col}'",
                xlabel=col,
                ylabel="Count",
                theme_name=theme_name
            )
            fig_plotly.add_trace(go.Histogram(
                x=series.values,
                name=col
            ))
        else:
            # Categorical bar count plots
            counts = series.value_counts()

            # 1. Matplotlib Count Plot
            fig_mpl, ax = FigureFactory.create_matplotlib_figure(
                title=f"Categorical Value Counts: {col}",
                subtitle=f"Bar counts representing frequencies of labels in column '{col}'",
                xlabel=col,
                ylabel="Observations Count",
                theme_name=theme_name
            )
            sns.barplot(x=list(counts.index), y=list(counts.values), ax=ax)
            fig_mpl.tight_layout()

            # 2. Plotly Bar
            fig_plotly = FigureFactory.create_plotly_figure(
                title=f"Categorical Value Counts: {col}",
                subtitle=f"Interactive counts representing frequencies of labels in column '{col}'",
                xlabel=col,
                ylabel="Count",
                theme_name=theme_name
            )
            fig_plotly.add_trace(go.Bar(
                x=list(counts.index),
                y=list(counts.values),
                name=col
            ))

        return {"matplotlib": fig_mpl, "plotly": fig_plotly}
