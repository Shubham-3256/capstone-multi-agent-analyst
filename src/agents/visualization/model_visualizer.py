"""Visualizer generating Confusion Matrices, ROC Curves, and Residual Plots."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

from src.core.logger import get_logger
from src.agents.visualization.figure_factory import FigureFactory
from src.agents.machine_learning.models import FeatureImportance

logger = get_logger(__name__)


class ModelVisualizer:
    """Generates figures auditing prediction performance (ROC curves, confusion matrices, residuals)."""

    @staticmethod
    def generate_confusion_matrix(
        matrix_data: List[List[int]],
        classes: List[str],
        theme_name: str = "corporate"
    ) -> Dict[str, Any]:
        """Generate confusion matrix visualizations.

        Args:
            matrix_data: Nested list representing cell values.
            classes: Class labels.
            theme_name: Styling theme.

        Returns:
            Dict[str, Any]: Mapped figures.
        """
        logger.info("ModelVisualizer: Creating confusion matrix heatmap...")
        df_matrix = pd.DataFrame(matrix_data, index=classes, columns=classes)

        # 1. Matplotlib Conf Matrix
        fig_mpl, ax = FigureFactory.create_matplotlib_figure(
            title="Model Confusion Matrix Heatmap",
            subtitle="Rows represent true target labels, columns represent predictions",
            xlabel="Predicted Class",
            ylabel="True Class",
            theme_name=theme_name
        )
        sns.heatmap(df_matrix, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax)
        fig_mpl.tight_layout()

        # 2. Plotly Conf Matrix
        fig_plotly = FigureFactory.create_plotly_figure(
            title="Model Confusion Matrix Heatmap",
            subtitle="Hover cells to read true vs predicted count matches",
            xlabel="Predicted Class",
            ylabel="True Class",
            theme_name=theme_name
        )
        fig_plotly.add_trace(go.Heatmap(
            z=matrix_data,
            x=classes,
            y=classes,
            colorscale="Greens",
            text=[[str(val) for val in row] for row in matrix_data],
            texttemplate="%{text}",
            showscale=False
        ))

        return {"matplotlib": fig_mpl, "plotly": fig_plotly}

    @staticmethod
    def generate_roc_curve(
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        theme_name: str = "corporate"
    ) -> Dict[str, Any]:
        """Generate ROC validation curve plots.

        Args:
            model: Fitted estimator model.
            X_val: Validation features.
            y_val: Validation labels.
            theme_name: Styling theme.

        Returns:
            Dict[str, Any]: Mapped figures.
        """
        logger.info("ModelVisualizer: Creating ROC validation curve...")
        
        # Clean inputs
        X_clean = X_val.fillna(0.0)
        y_clean = y_val.fillna(0)

        # Compute ROC metrics if model supports probas
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_clean)[:, 1]
                fpr, tpr, _ = roc_curve(y_clean, y_prob)
                roc_auc = auc(fpr, tpr)
            except Exception as e:
                logger.warning(f"Failed to extract ROC curves: {e}. Fallback to dummy curve.")
                fpr, tpr, roc_auc = [0.0, 1.0], [0.0, 1.0], 0.5
        else:
            fpr, tpr, roc_auc = [0.0, 1.0], [0.0, 1.0], 0.5

        # 1. Matplotlib ROC
        fig_mpl, ax = FigureFactory.create_matplotlib_figure(
            title="Receiver Operating Characteristic (ROC)",
            subtitle=f"Validation area under curve (AUC = {round(roc_auc, 4)})",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            theme_name=theme_name
        )
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {round(roc_auc, 3)})", linewidth=2.0)
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right")
        fig_mpl.tight_layout()

        # 2. Plotly ROC
        fig_plotly = FigureFactory.create_plotly_figure(
            title="Receiver Operating Characteristic (ROC)",
            subtitle=f"Validation area under curve (AUC = {round(roc_auc, 4)})",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            theme_name=theme_name
        )
        fig_plotly.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC={round(roc_auc, 3)})",
            fill="tozeroy"
        ))
        fig_plotly.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "grey"},
            name="Baseline Chance"
        ))

        return {"matplotlib": fig_mpl, "plotly": fig_plotly}

    @staticmethod
    def generate_residual_plot(
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        theme_name: str = "corporate"
    ) -> Dict[str, Any]:
        """Generate residuals vs predictions scatter plots for regression validation diagnostics.

        Args:
            model: Fitted regression estimator model.
            X_val: Validation features.
            y_val: Validation target labels.
            theme_name: Styling theme.

        Returns:
            Dict[str, Any]: Mapped figures.
        """
        logger.info("ModelVisualizer: Creating residual diagnostic plots...")
        
        # Make predictions
        X_clean = X_val.fillna(0.0)
        y_clean = y_val.fillna(0)
        preds = model.predict(X_clean)
        residuals = y_clean.values - preds

        # 1. Matplotlib Residuals
        fig_mpl, ax = FigureFactory.create_matplotlib_figure(
            title="Regression Residuals Analysis",
            subtitle="Values scattered randomly around zero line indicate high fit validation",
            xlabel="Predicted Value",
            ylabel="Residual Error (Actual - Pred)",
            theme_name=theme_name
        )
        ax.scatter(preds, residuals, alpha=0.6)
        ax.axhline(y=0.0, color="red", linestyle="--")
        fig_mpl.tight_layout()

        # 2. Plotly Residuals
        fig_plotly = FigureFactory.create_plotly_figure(
            title="Regression Residuals Analysis",
            subtitle="Scatter showing errors distributions per prediction point",
            xlabel="Predicted Value",
            ylabel="Residual (Actual - Pred)",
            theme_name=theme_name
        )
        fig_plotly.add_trace(go.Scatter(
            x=preds,
            y=residuals,
            mode="markers",
            name="Residuals",
            marker={"opacity": 0.7}
        ))
        fig_plotly.add_trace(go.Scatter(
            x=[min(preds), max(preds)],
            y=[0, 0],
            mode="lines",
            line={"dash": "dash", "color": "red"},
            name="Zero Reference"
        ))

        return {"matplotlib": fig_mpl, "plotly": fig_plotly}

    @staticmethod
    def generate_feature_importance_plot(
        importances: List[FeatureImportance],
        theme_name: str = "corporate"
    ) -> Dict[str, Any]:
        """Generate ranked feature importance horizontal bar plots.

        Args:
            importances: Feature importances.
            theme_name: Styling theme.

        Returns:
            Dict[str, Any]: Mapped figures.
        """
        logger.info("ModelVisualizer: Creating feature importances chart...")
        
        # Sort and map to DataFrame
        df_imp = pd.DataFrame([
            {"feature": item.column, "importance": item.importance}
            for item in importances
        ]).sort_values(by="importance", ascending=True)

        if df_imp.empty:
            df_imp = pd.DataFrame({"feature": ["dummy_f"], "importance": [1.0]})

        # 1. Matplotlib Bar
        fig_mpl, ax = FigureFactory.create_matplotlib_figure(
            title="Estimator Feature Importance Profiles",
            subtitle="Scores indicating relative predictive importance",
            xlabel="Relative Importance",
            ylabel="Feature Column",
            theme_name=theme_name
        )
        ax.barh(df_imp["feature"], df_imp["importance"])
        fig_mpl.tight_layout()

        # 2. Plotly Bar
        fig_plotly = FigureFactory.create_plotly_figure(
            title="Estimator Feature Importance Profiles",
            subtitle="Interactive scores indicating relative predictive importance",
            xlabel="Relative Importance",
            ylabel="Feature Column",
            theme_name=theme_name
        )
        fig_plotly.add_trace(go.Bar(
            x=df_imp["importance"],
            y=df_imp["feature"],
            orientation="h",
            name="Importance"
        ))

        return {"matplotlib": fig_mpl, "plotly": fig_plotly}
