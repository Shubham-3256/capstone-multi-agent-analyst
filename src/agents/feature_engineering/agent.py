"""Feature Engineering Agent orchestrating structural profiling, selectors, and pipelines saves."""

import time
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from src.agents.feature_engineering.config import FeatureEngineeringConfig
from src.agents.feature_engineering.detector import FeatureDetector
from src.agents.feature_engineering.models import (
    EncodingReport,
    FeatureEngineeringResult,
    ScalingReport,
    SelectionReport,
)
from src.agents.feature_engineering.pipeline import PipelineBuilder
from src.agents.feature_engineering.splitter import TrainValidationSplitter
from src.agents.feature_engineering.validators import LeakageDetector
from src.core.exceptions import DatasetException
from src.core.logger import get_logger
from src.core.paths import Paths
from src.database.models import DatasetRecord, ExecutionLog
from src.repositories.dataset_repository import DatasetRepository
from src.repositories.log_repository import ExecutionLogRepository
from src.services.dataset_service import DatasetService
from src.utils.serialization import serialize_dataframe

logger = get_logger(__name__)


class FeatureEngineeringAgent:
    """Agent managing type classification, leakage warnings, train-splits and pipeline builds."""

    def __init__(self, db_session: Session) -> None:
        """Initialize FeatureEngineeringAgent with database session.

        Args:
            db_session: Active SQLAlchemy connection session context.
        """
        self.db = db_session
        self.log_repo = ExecutionLogRepository(db_session)
        self.dataset_repo = DatasetRepository(db_session)
        self.dataset_service = DatasetService(db_session)

    def run(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        config_path: Path | None = None
    ) -> FeatureEngineeringResult:
        """Execute feature engineering pipeline: Detect -> Generate -> Encode -> Scale -> Select -> Split -> Save Pipeline.

        Args:
            dataframe: Input clean dataset DataFrame (output from Data Intelligence Agent).
            target_column: Name of the target label variable.
            config_path: Optional path to YAML configuration settings.

        Returns:
            FeatureEngineeringResult: Output containing metadata, reports, splits, and pipeline references.
        """
        logger.info("FeatureEngineeringAgent: Running pre-processing pipeline...")
        start_time = time.time()

        # Load configurations
        config = FeatureEngineeringConfig.load_from_yaml(config_path)

        # Create audit execution log
        log_record = ExecutionLog(
            task_name="feature_engineering_pipeline",
            agent_name="FeatureEngineeringAgent",
            status="running",
            parameters=str({
                "target_column": target_column,
                "strategy": config.split.strategy,
                "selection_method": config.selection.method
            })
        )
        self.log_repo.create(log_record)
        self.db.commit()

        try:
            # 1. Feature Detection
            detect_report = FeatureDetector.detect(dataframe, target_column)
            feature_types = detect_report["feature_types"]
            identifier_cols = [col for col, t in feature_types.items() if t == "identifier"]

            # 2. Leakage Detection
            leak_report = LeakageDetector.detect_leakage(dataframe, target_column, identifier_cols)

            # 3. Train/Validation/Test Split
            train_df, val_df, test_df, split_report = TrainValidationSplitter.split(
                df=dataframe,
                target_column=target_column,
                strategy=config.split.strategy,
                train_ratio=config.split.train_ratio,
                val_ratio=config.split.val_ratio,
                test_ratio=config.split.test_ratio,
                random_seed=config.split.random_seed
            )

            # Extract target vectors for fitting
            y_train = train_df[target_column] if target_column in train_df.columns else None

            # 4. Build and Fit scikit-learn preprocessing pipeline on train features
            pipeline = PipelineBuilder.build_and_fit(
                df_train=train_df,
                y_train=y_train,
                config=config,
                target_column=target_column
            )

            # 5. Transform Train, Validation, and Test datasets using fitted pipeline
            X_train = train_df.drop(columns=[target_column]) if target_column in train_df.columns else train_df
            X_train_trans = pipeline.transform(X_train)
            if target_column in train_df.columns:
                X_train_trans[target_column] = train_df[target_column].values

            X_val = val_df.drop(columns=[target_column]) if target_column in val_df.columns else val_df
            X_val_trans = pipeline.transform(X_val)
            if target_column in val_df.columns:
                X_val_trans[target_column] = val_df[target_column].values

            X_test = test_df.drop(columns=[target_column]) if target_column in test_df.columns else test_df
            X_test_trans = pipeline.transform(X_test)
            if target_column in test_df.columns:
                X_test_trans[target_column] = test_df[target_column].values

            # 6. Save transformed dataset splits to Paths.PROCESSED_DIR
            Paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

            train_filepath = Paths.PROCESSED_DIR / "train.csv"
            val_filepath = Paths.PROCESSED_DIR / "val.csv"
            test_filepath = Paths.PROCESSED_DIR / "test.csv"
            pipeline_filepath = Paths.PROCESSED_DIR / "pipeline.joblib"

            serialize_dataframe(X_train_trans, train_filepath, fmt=".csv")
            serialize_dataframe(X_val_trans, val_filepath, fmt=".csv")
            serialize_dataframe(X_test_trans, test_filepath, fmt=".csv")

            # 7. Serialize preprocessing pipeline
            pipeline_report = PipelineBuilder.save_pipeline(pipeline, pipeline_filepath)

            # Extract parameter reports
            encoder_step = pipeline.named_steps["encoder"]
            encoding_report = EncodingReport(
                strategy_used=encoder_step.strategies_,
                mappings={k: {str(cat): int(idx) for idx, cat in enumerate(v.categories_[0])} for k, v in encoder_step.onehot_transformers_.items()}
            )

            scaler_step = pipeline.named_steps["scaler"]
            scaling_report = ScalingReport(
                scaler_type=scaler_step.scalers_,
                scaling_parameters=scaler_step.scaling_params_
            )

            selector_step = pipeline.named_steps["selector"]
            selection_report = SelectionReport(
                method=config.selection.method,
                original_count=X_train.shape[1],
                selected_count=len(selector_step.columns_to_keep_),
                selected_features=selector_step.columns_to_keep_,
                feature_importances=selector_step.feature_importances_
            )

            # 8. Register split files in database DatasetRecords
            logger.info("Registering train/val/test data splits inside SQL database...")

            # Helper to calculate and persist DatasetRecord
            def register_split(filepath: Path, status_label: str, rows: int, cols: int):
                file_size = filepath.stat().st_size
                file_hash = self.dataset_service.calculate_checksum(filepath)

                record = self.dataset_repo.get_by_hash(file_hash)
                if not record:
                    record = DatasetRecord(
                        filename=filepath.name,
                        filepath=str(filepath),
                        file_hash=file_hash,
                        file_size_bytes=file_size,
                        row_count=rows,
                        column_count=cols,
                        status=status_label
                    )
                    self.dataset_repo.create(record)
                else:
                    record.status = status_label
                    record.row_count = rows
                    record.column_count = cols
                return record

            train_record = register_split(train_filepath, "train_split", *X_train_trans.shape)
            register_split(val_filepath, "val_split", *X_val_trans.shape)
            register_split(test_filepath, "test_split", *X_test_trans.shape)

            # Update log as completed
            duration = time.time() - start_time
            self.log_repo.update_status(
                log_id=log_record.id,
                status="completed",
                duration_seconds=duration
            )
            self.db.commit()

            logger.info(f"FeatureEngineeringAgent: Pipeline run completed in {round(duration, 4)}s")
            return FeatureEngineeringResult(
                dataset_id=train_record.id,
                is_success=True,
                feature_types=feature_types,
                encoding_report=encoding_report,
                scaling_report=scaling_report,
                selection_report=selection_report,
                leakage_report=leak_report,
                split_report=split_report,
                pipeline_report=pipeline_report,
                train_filepath=str(train_filepath),
                val_filepath=str(val_filepath),
                test_filepath=str(test_filepath),
                duration_seconds=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"FeatureEngineeringAgent: Pipeline execution failed: {e}")
            self.log_repo.update_status(
                log_id=log_record.id,
                status="failed",
                duration_seconds=duration,
                error_message=f"Pipeline error: {str(e)}"
            )
            self.db.commit()
            raise DatasetException(f"Pipeline error executing FeatureEngineeringAgent: {e}") from e
