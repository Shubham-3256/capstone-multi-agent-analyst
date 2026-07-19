"""YAML-driven configuration parser for Report Generation paths and theme choices."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.core.logger import get_logger
from src.core.paths import Paths

logger = get_logger(__name__)


class ReportConfig(BaseModel):
    """Configuration mapping output directories, layout templates, and exporter toggles."""

    reports_dir: Path = Field(
        default=Paths.WORKSPACE_DIR / "artifacts" / "reports",
        description="Base directory for compiled document exports"
    )
    theme_name: str = Field(default="corporate", description="Layout styling themes ('light', 'dark', 'corporate')")
    include_manifest: bool = Field(default=True, description="Whether to generate report_manifest.json index")
    auto_resize_assets: bool = Field(default=True, description="Enable automatic graph image scaling")

    @classmethod
    def load_from_yaml(cls, yaml_path: Path | None = None) -> "ReportConfig":
        """Load document settings from YAML file or return defaults.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            ReportConfig: Parsed configuration object.
        """
        if yaml_path and yaml_path.exists():
            try:
                with open(yaml_path, encoding="utf-8") as f:
                    content = yaml.safe_load(f) or {}
                # Resolve report subset block if present
                report_content = content.get("report", content)
                logger.info(f"Loaded Report configurations from: {yaml_path}")
                return cls(**report_content)
            except Exception as e:
                logger.warning(f"Failed to load yaml config from {yaml_path}: {e}. Using defaults.")
        return cls()
