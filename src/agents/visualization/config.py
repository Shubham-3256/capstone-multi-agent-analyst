"""YAML-driven configuration parser for visualization formatting rules."""

from typing import List, Optional
import yaml
from pathlib import Path
from pydantic import BaseModel, Field

from src.core.logger import get_logger

logger = get_logger(__name__)


class PlotSizeConfig(BaseModel):
    """Configuration for output image sizing parameters."""

    width_inches: float = Field(default=10.0, description="Matplotlib figure width in inches")
    height_inches: float = Field(default=6.0, description="Matplotlib figure height in inches")
    plotly_width_pixels: int = Field(default=1000, description="Plotly figure width in pixels")
    plotly_height_pixels: int = Field(default=600, description="Plotly figure height in pixels")


class VisualizationConfig(BaseModel):
    """Unified configuration mapping for Visualization Agent runs."""

    default_theme: str = Field(default="corporate", description="Theme key to apply ('light', 'dark', 'corporate')")
    export_formats: List[str] = Field(default=["png", "html"], description="File formats to save ('png', 'svg', 'html', 'pdf')")
    size: PlotSizeConfig = Field(default_factory=PlotSizeConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: Optional[Path] = None) -> "VisualizationConfig":
        """Load configuration parameters from YAML file or return defaults.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            VisualizationConfig: Parsed configuration object.
        """
        if yaml_path and yaml_path.exists():
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f) or {}
                logger.info(f"Loaded visualization config from: {yaml_path}")
                return cls(**content)
            except Exception as e:
                logger.warning(f"Failed to load yaml config from {yaml_path}: {e}. Using defaults.")
        return cls()
