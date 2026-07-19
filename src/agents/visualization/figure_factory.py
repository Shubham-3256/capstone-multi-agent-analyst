"""Figure creation factory constructing styled Matplotlib and Plotly canvas wrappers."""

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.agents.visualization.themes import ThemeManager
from src.core.logger import get_logger

logger = get_logger(__name__)


class FigureFactory:
    """Provides reusable builders for Plotly and Matplotlib templates applying themes."""

    @staticmethod
    def create_matplotlib_figure(
        title: str,
        subtitle: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        theme_name: str = "corporate",
        figsize: tuple[float, float] = (10.0, 6.0)
    ) -> tuple[plt.Figure, plt.Axes]:
        """Instantiate and style a Matplotlib canvas.

        Args:
            title: Main chart title.
            subtitle: Subtitle context details.
            xlabel: Horizontal axis label.
            ylabel: Vertical axis label.
            theme_name: Key of selected theme.
            figsize: Figure dimensions in inches.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Canvas figure and axis.
        """
        # Apply theme
        ThemeManager.apply_matplotlib_theme(theme_name)

        fig, ax = plt.subplots(figsize=figsize)

        # Add labels
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        # Style title and subtitle
        if subtitle:
            # Shift main title up, add subtitle
            ax.set_title(f"{title}\n", fontsize=14, fontweight="bold", pad=15)
            # Annotate subtitle text
            ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, style="italic")
        else:
            ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

        # Clean layouts
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def create_plotly_figure(
        title: str,
        subtitle: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        theme_name: str = "corporate"
    ) -> go.Figure:
        """Instantiate and style an interactive Plotly graph figure.

        Args:
            title: Main chart title.
            subtitle: Subtitle context details.
            xlabel: Horizontal axis label.
            ylabel: Vertical axis label.
            theme_name: Key of selected theme.

        Returns:
            go.Figure: Styled Plotly Figure object.
        """
        layout_params = ThemeManager.get_plotly_layout(theme_name)

        # Assemble title HTML representation
        title_text = f"<b>{title}</b>"
        if subtitle:
            title_text += f"<br><sup><i>{subtitle}</i></sup>"

        fig = go.Figure()

        # Apply layouts
        fig.update_layout(
            title={"text": title_text, "x": 0.05, "y": 0.95},
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            margin={"t": 80, "b": 50, "l": 50, "r": 50},
            **layout_params
        )

        return fig
