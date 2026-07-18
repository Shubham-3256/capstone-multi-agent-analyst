"""Visualizer generating leaderboard comparison dashboards charts."""

from typing import Dict, Any
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.core.logger import get_logger
from src.agents.visualization.figure_factory import FigureFactory
from src.agents.machine_learning.models import Leaderboard

logger = get_logger(__name__)


class DashboardVisualizer:
    """Generates figures comparing candidate models across execution history and leaderboards."""

    @staticmethod
    def generate_leaderboard_chart(
        leaderboard: Leaderboard,
        task_type: str,
        theme_name: str = "corporate"
    ) -> Dict[str, Any]:
        """Generate candidate models metrics comparisons leaderboard bar chart.

        Args:
            leaderboard: Compiled leaderboard.
            task_type: Mapped task type ('classification' or 'regression').
            theme_name: Styling theme.

        Returns:
            Dict[str, Any]: Mapped figures containing 'matplotlib' and 'plotly' handles.
        """
        logger.info("DashboardVisualizer: Creating leaderboard comparison bar chart...")
        
        # Parse leaderboard to pandas DataFrame
        records = [
            {"model_name": entry.model_name, "score": entry.score}
            for entry in leaderboard.entries
        ]
        
        # Fallback dummy record if leaderboard empty
        if not records:
            records = [{"model_name": "No models trained", "score": 0.0}]

        df_leaderboard = pd.DataFrame(records)

        # Main metrics mapping
        metric_name = "Macro F1 Score" if task_type == "classification" else "RMSE (Lower is Better)"

        # 1. Matplotlib Leaderboard
        fig_mpl, ax = FigureFactory.create_matplotlib_figure(
            title="AutoML Candidate Model Leaderboard",
            subtitle=f"Comparing candidate algorithms performance by {metric_name}",
            xlabel=metric_name,
            ylabel="Candidate Model",
            theme_name=theme_name
        )
        sns.barplot(x="score", y="model_name", data=df_leaderboard, ax=ax)
        fig_mpl.tight_layout()

        # 2. Plotly Leaderboard
        fig_plotly = FigureFactory.create_plotly_figure(
            title="AutoML Candidate Model Leaderboard",
            subtitle=f"Comparing candidate algorithms performance by {metric_name}",
            xlabel=metric_name,
            ylabel="Candidate Model",
            theme_name=theme_name
        )
        fig_plotly.add_trace(go.Bar(
            x=df_leaderboard["score"],
            y=df_leaderboard["model_name"],
            orientation="h",
            name=metric_name
        ))

        return {"matplotlib": fig_mpl, "plotly": fig_plotly}
