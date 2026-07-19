"""Exporter manager saving figures to PNG images, interactive HTML files and JSON metadata."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.agents.visualization.models import ChartCaption, ChartMetadata
from src.core.logger import get_logger
from src.core.paths import Paths

logger = get_logger(__name__)


class Exporter:
    """Serializes Matplotlib figures to static images and Plotly figures to interactive HTML layouts."""

    @staticmethod
    def export_chart(
        chart_id: str,
        title: str,
        chart_type: str,
        figures: dict[str, Any],
        caption: ChartCaption,
        base_dir: Path = Paths.WORKSPACE_DIR / "artifacts",
    ) -> ChartMetadata:
        """Export Matplotlib figure to PNG/SVG and Plotly figure to HTML.

        Args:
            chart_id: Slug identifier of target chart.
            title: Title text.
            chart_type: Category (heatmap, bar, etc.).
            figures: Dict mapping 'matplotlib' and 'plotly' keys to figure handles.
            caption: ChartCaption explaining the plot insights.
            base_dir: Output base directory.

        Returns:
            ChartMetadata: Metadata detailing generated assets.
        """
        logger.info(f"Exporter: Saving chart '{chart_id}'...")

        # Setup paths
        plots_dir = base_dir / "plots"
        html_dir = base_dir / "html"

        plots_dir.mkdir(parents=True, exist_ok=True)
        html_dir.mkdir(parents=True, exist_ok=True)

        png_filepath = plots_dir / f"{chart_id}.png"
        html_filepath = html_dir / f"{chart_id}.html"

        # 1. Export Matplotlib counterpart to static PNG
        fig_mpl = figures.get("matplotlib")
        if fig_mpl is not None:
            try:
                fig_mpl.savefig(png_filepath, dpi=150, bbox_inches="tight")
                # Close figure to release memory
                plt.close(fig_mpl)
                logger.info(f"  Saved static PNG to: {png_filepath}")
            except Exception as e:
                logger.error(f"  Failed to save matplotlib PNG: {e}")
        else:
            logger.warning(
                f"  No matplotlib figure handle found for '{chart_id}' static export."
            )

        # 2. Export Plotly counterpart to interactive HTML
        fig_plotly = figures.get("plotly")
        if fig_plotly is not None:
            try:
                fig_plotly.write_html(
                    str(html_filepath), include_plotlyjs="cdn", full_html=False
                )
                logger.info(f"  Saved interactive HTML to: {html_filepath}")
            except Exception as e:
                logger.error(f"  Failed to save plotly HTML: {e}")
        else:
            logger.warning(
                f"  No plotly figure handle found for '{chart_id}' interactive export."
            )

        return ChartMetadata(
            chart_id=chart_id,
            title=title,
            chart_type=chart_type,
            file_path=str(png_filepath),
            html_path=str(html_filepath) if fig_plotly is not None else None,
            caption=caption,
        )

    @staticmethod
    def save_report_metadata(report_data: dict[str, Any], filepath: Path) -> None:
        """Serialize run parameters and chart descriptions to JSON.

        Args:
            report_data: Metadata payload.
            filepath: Location path.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Exporter: Report metadata saved to: {filepath}")
