"""Pipeline builder for compiling and serializing sklearn Preprocessing Pipelines."""

from pathlib import Path
from typing import List, Optional
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.core.logger import get_logger
from src.agents.feature_engineering.config import FeatureEngineeringConfig
from src.agents.feature_engineering.encoder import CategoricalEncoder
from src.agents.feature_engineering.scaler import NumericalScaler
from src.agents.feature_engineering.generator import FeatureGenerator
from src.agents.feature_engineering.selector import FeatureSelector
from src.agents.feature_engineering.models import PipelineReport

logger = get_logger(__name__)


class PipelineBuilder:
    """Builds, fits, and serializes scikit-learn preprocessing pipelines."""

    @staticmethod
    def build_and_fit(
        df_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        config: Optional[FeatureEngineeringConfig] = None,
        target_column: Optional[str] = None
    ) -> Pipeline:
        """Instantiate and fit the unified preprocessing pipeline on training data.

        Args:
            df_train: Training features DataFrame.
            y_train: Optional target labels Series.
            config: Configuration settings context.
            target_column: Name of target column.

        Returns:
            Pipeline: Fitted scikit-learn Pipeline.
        """
        cfg = config or FeatureEngineeringConfig()
        logger.info("PipelineBuilder: Constructing scikit-learn preprocessor pipeline...")

        # Exclude target column from pipeline features
        features_df = df_train.drop(columns=[target_column]) if target_column and target_column in df_train.columns else df_train

        # 1. Step 1: Feature Generation
        generator = FeatureGenerator(
            polynomial_degree=cfg.generation.polynomial_degree,
            interaction_only=cfg.generation.interaction_only,
            date_features=cfg.generation.date_features,
            log_transform=cfg.generation.log_transform,
            skew_threshold=cfg.generation.skew_threshold
        )

        # 2. Step 2: Categorical Encoding
        encoder = CategoricalEncoder(
            low_cardinality_threshold=cfg.encoding.low_cardinality_threshold,
            medium_cardinality_threshold=cfg.encoding.medium_cardinality_threshold,
            default_strategy=cfg.encoding.default
        )

        # 3. Step 3: Numeric Scaling
        scaler = NumericalScaler(
            outlier_threshold=cfg.scaling.outlier_threshold,
            default_scaler=cfg.scaling.default
        )

        # 4. Step 4: Feature Selection
        selector = FeatureSelector(
            method=cfg.selection.method,
            variance_threshold=cfg.selection.variance_threshold,
            correlation_threshold=cfg.selection.correlation_threshold,
            k_best=cfg.selection.k_best,
            task_type="classification" if y_train is not None and y_train.nunique() <= 10 else "regression"
        )

        # 5. Assemble scikit-learn Pipeline
        pipeline = Pipeline([
            ("generator", generator),
            ("encoder", encoder),
            ("scaler", scaler),
            ("selector", selector)
        ])

        logger.info("PipelineBuilder: Fitting assembled pipeline on training data...")
        pipeline.fit(features_df, y_train)
        
        return pipeline

    @staticmethod
    def save_pipeline(pipeline: Pipeline, filepath: Path) -> PipelineReport:
        """Serialize the fitted pipeline to disk using joblib.

        Args:
            pipeline: Fitted scikit-learn Pipeline.
            filepath: Target output file location.

        Returns:
            PipelineReport: Details of the saved pipeline.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, str(filepath))
        logger.info(f"PipelineBuilder: Preprocessing pipeline saved to: {filepath}")

        # Fetch names of pipeline step keys
        components = list(pipeline.named_steps.keys())

        return PipelineReport(
            pipeline_filepath=str(filepath),
            components=components
        )

    @staticmethod
    def load_pipeline(filepath: Path) -> Pipeline:
        """Load a serialized preprocessing pipeline from disk.

        Args:
            filepath: Location of the joblib file.

        Returns:
            Pipeline: Re-instantiated fitted scikit-learn Pipeline.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        
        pipeline = joblib.load(str(filepath))
        logger.info(f"PipelineBuilder: Successfully loaded preprocessing pipeline from: {filepath}")
        return pipeline
