"""YAML-driven configuration parser for Machine Learning modeling parameters."""

from typing import Any, Dict, List, Optional
import yaml
from pathlib import Path
from pydantic import BaseModel, Field

from src.core.logger import get_logger

logger = get_logger(__name__)


class ModelingConfig(BaseModel):
    """Configuration for cross validation and hyperparameter tuning sweeps."""

    cv_folds: int = Field(default=5, description="Number of folds to use for cross validation")
    tuning_mode: str = Field(default="fast", description="Tuning grid depth ('fast' or 'full')")
    tuning_search_type: str = Field(default="random", description="Tuning search algorithm ('random' or 'grid')")
    n_iter_search: int = Field(default=5, description="Number of parameter settings sampled in RandomizedSearchCV")
    n_jobs: int = Field(default=-1, description="Number of parallel workers to execute (-1 to use all cores)")
    random_seed: int = Field(default=42, description="Random split and estimator initialization seed")
    timeout_seconds: int = Field(default=600, description="Max allowed execution timeout limit in seconds per model")


class ModelSelectionConfig(BaseModel):
    """Configuration for selected target classifiers and regressors."""

    classification_candidates: List[str] = Field(
        default=[
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "gradient_boosting",
            "k_neighbors",
            "gaussian_nb"
        ],
        description="Candidate models to train for classification tasks"
    )
    regression_candidates: List[str] = Field(
        default=[
            "linear_regression",
            "ridge",
            "lasso",
            "decision_tree",
            "random_forest",
            "gradient_boosting"
        ],
        description="Candidate models to train for regression tasks"
    )


class MachineLearningConfig(BaseModel):
    """Unified configuration mapping for AutoML machine learning agent."""

    modeling: ModelingConfig = Field(default_factory=ModelingConfig)
    selection: ModelSelectionConfig = Field(default_factory=ModelSelectionConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: Optional[Path] = None) -> "MachineLearningConfig":
        """Load configuration parameters from YAML file or return defaults.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            MachineLearningConfig: Parsed configuration object.
        """
        if yaml_path and yaml_path.exists():
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f) or {}
                logger.info(f"Loaded machine learning config from: {yaml_path}")
                return cls(**content)
            except Exception as e:
                logger.warning(f"Failed to load yaml config from {yaml_path}: {e}. Using defaults.")
        return cls()
