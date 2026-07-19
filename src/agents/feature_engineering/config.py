"""YAML-driven Configuration parser for feature engineering parameters."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.core.logger import get_logger

logger = get_logger(__name__)


class EncodingConfig(BaseModel):
    """Configuration for feature encoding strategies."""

    default: str = Field(default="auto", description="Default encoding ('auto', 'onehot', 'ordinal', 'frequency', 'label')")
    low_cardinality_threshold: int = Field(default=10, description="Unique values limit for One-Hot encoding")
    medium_cardinality_threshold: int = Field(default=25, description="Unique values limit for Ordinal encoding")


class ScalingConfig(BaseModel):
    """Configuration for scaling parameters."""

    default: str = Field(default="auto", description="Default scaler ('auto', 'standard', 'robust', 'minmax', 'maxabs', 'none')")
    outlier_threshold: float = Field(default=0.05, description="Ratio of outlier samples before RobustScaler is chosen")


class GenerationConfig(BaseModel):
    """Configuration for feature generation."""

    polynomial_degree: int = Field(default=2, description="Degree for polynomial features (0 to disable)")
    interaction_only: bool = Field(default=True, description="Only generate interaction numerical terms (no self-squares)")
    date_features: bool = Field(default=True, description="Extract year, month, day, weekday, quarter from dates")
    log_transform: bool = Field(default=True, description="Apply log1p on highly skewed numerical features")
    skew_threshold: float = Field(default=1.5, description="Skewness limit for log transformations")


class SelectionConfig(BaseModel):
    """Configuration for feature selection parameters."""

    method: str = Field(default="correlation", description="Selection method ('variance', 'correlation', 'mutual_info', 'vif', 'none')")
    variance_threshold: float = Field(default=0.01, description="Minimum variance cutoff for low variance filter")
    correlation_threshold: float = Field(default=0.85, description="Pearson correlation limit for multicollinearity drops")
    vif_threshold: float = Field(default=10.0, description="Max Variance Inflation Factor limit")
    k_best: int = Field(default=10, description="Number of best features to select via SelectKBest")


class SplitConfig(BaseModel):
    """Configuration for train/validation/test splitting ratios."""

    strategy: str = Field(default="random", description="Split strategy ('random', 'stratified', 'time_series')")
    train_ratio: float = Field(default=0.70, description="Train set ratio")
    val_ratio: float = Field(default=0.15, description="Validation set ratio")
    test_ratio: float = Field(default=0.15, description="Test set ratio")
    random_seed: int = Field(default=42, description="Random split seed")


class FeatureEngineeringConfig(BaseModel):
    """Unified configuration parameters mapping for feature engineering agent."""

    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: Path | None = None) -> "FeatureEngineeringConfig":
        """Load configuration parameters from YAML file or return defaults.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            FeatureEngineeringConfig: Parsed configuration object.
        """
        if yaml_path and yaml_path.exists():
            try:
                with open(yaml_path, encoding="utf-8") as f:
                    content = yaml.safe_load(f) or {}
                logger.info(f"Loaded feature engineering config from: {yaml_path}")
                return cls(**content)
            except Exception as e:
                logger.warning(f"Failed to load yaml config from {yaml_path}: {e}. Using defaults.")
        return cls()
