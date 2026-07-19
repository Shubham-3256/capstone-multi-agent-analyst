"""Orchestration Agent coordinating dataset validation, cleaning, profiling, and metadata saves."""

import time
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from src.agents.data_intelligence.cleaner import Cleaner
from src.agents.data_intelligence.models import (
    CleaningReport,
    DataIntelligenceResult,
    DatasetProfile,
    ValidationReport,
)
from src.agents.data_intelligence.profiler import Profiler
from src.agents.data_intelligence.validator import Validator
from src.core.exceptions import DatasetException
from src.core.logger import get_logger
from src.core.paths import Paths
from src.core.security import secure_filename
from src.database.models import DatasetRecord, ExecutionLog
from src.repositories.dataset_repository import DatasetRepository
from src.repositories.log_repository import ExecutionLogRepository
from src.services.dataset_service import DatasetService
from src.services.metadata_service import MetadataService
from src.utils.serialization import serialize_dataframe

logger = get_logger(__name__)


class DataIntelligenceAgent:
    """Agent implementing automated end-to-end dataset validation, sanitization, and statistical profiling."""

    def __init__(self, db_session: Session) -> None:
        """Initialize DataIntelligenceAgent with an active database session.

        Args:
            db_session: Active SQLAlchemy connection session context.
        """
        self.db = db_session
        self.log_repo = ExecutionLogRepository(db_session)
        self.dataset_repo = DatasetRepository(db_session)
        self.dataset_service = DatasetService(db_session)

    def validate(self, df: pd.DataFrame, file_path: Path, target_column: str | None = None) -> ValidationReport:
        """Execute structural check sequences on the dataset DataFrame.

        Args:
            df: Active dataset representation.
            file_path: Source filepath context.
            target_column: Optional modeling target column name.

        Returns:
            ValidationReport: Generated validation checks summary.
        """
        return Validator.validate(df, file_path, target_column)

    def clean(
        self,
        df: pd.DataFrame,
        imputation_strategies: dict[str, str] | None = None,
        outlier_strategies: dict[str, str] | None = None,
        datatype_conversions: dict[str, str] | None = None
    ) -> tuple[pd.DataFrame, CleaningReport]:
        """Normalize types, impute missing values, and handle outlier caps on the dataset.

        Args:
            df: Active DataFrame payload.
            imputation_strategies: Column-specific NaN strategies.
            outlier_strategies: Column-specific outlier cap structures.
            datatype_conversions: Column-specific type conversions.

        Returns:
            tuple[pd.DataFrame, CleaningReport]: Cleaned DataFrame and cleaning adjustments summary.
        """
        return Cleaner.clean(
            df=df,
            imputation_strategies=imputation_strategies,
            outlier_strategies=outlier_strategies,
            datatype_conversions=datatype_conversions
        )

    def profile(self, df: pd.DataFrame, target_column: str | None = None) -> DatasetProfile:
        """Extract multi-column correlations, distributions, and model suggestions from the DataFrame.

        Args:
            df: Cleaned DataFrame payload.
            target_column: Optional target prediction variable.

        Returns:
            DatasetProfile: The computed statistical profile metadata.
        """
        return Profiler.profile(df, target_column)

    def run(
        self,
        file_path: Path,
        target_column: str | None = None,
        imputation_strategies: dict[str, str] | None = None,
        outlier_strategies: dict[str, str] | None = None,
        datatype_conversions: dict[str, str] | None = None
    ) -> DataIntelligenceResult:
        """Execute the complete Data Intelligence Pipeline: Ingest -> Validate -> Clean -> Profile.

        Updates database logs, registers final cleaned files, and publishes profiles.

        Args:
            file_path: Path to the target dataset file.
            target_column: Optional target label column.
            imputation_strategies: Imputation configurations.
            outlier_strategies: Capping configurations.
            datatype_conversions: Datatype transformations.

        Returns:
            DataIntelligenceResult: Struct containing pipeline summary metrics.
        """
        logger.info(f"DataIntelligenceAgent: Running pipeline on {file_path.name}")
        start_time = time.time()

        # Create audit execution log
        log_record = ExecutionLog(
            task_name="data_intelligence_pipeline",
            agent_name="DataIntelligenceAgent",
            status="running",
            parameters=str({
                "file_path": str(file_path),
                "target_column": target_column,
                "imputation_strategies": imputation_strategies,
                "outlier_strategies": outlier_strategies
            })
        )
        self.log_repo.create(log_record)
        self.db.commit()

        try:
            # 1. Ingest dataset
            logger.info("Step 1: Ingesting source dataset...")
            # Automatically registers dataset in database if not already tracked
            record = self.dataset_service.register_dataset(file_path, file_path.name)
            df = self.dataset_service.load_dataset_df(record.id)

            # 2. Dataset Validation
            logger.info("Step 2: Performing dataset validations...")
            v_report = self.validate(df, file_path, target_column)

            # If validation has blocker errors, fail the run immediately
            if not v_report.is_valid:
                logger.error("Dataset validation failed with errors. Aborting pipeline.")
                duration = time.time() - start_time
                self.log_repo.update_status(
                    log_id=log_record.id,
                    status="failed",
                    duration_seconds=duration,
                    error_message=f"Validation errors detected: {v_report.summary.get('errors')} errors."
                )
                self.db.commit()
                return DataIntelligenceResult(
                    dataset_id=record.id,
                    is_valid=False,
                    validation_report=v_report,
                    duration_seconds=duration
                )

            # 3. Dataset Cleaning
            logger.info("Step 3: Executing data cleaning filters...")
            cleaned_df, c_report = self.clean(
                df=df,
                imputation_strategies=imputation_strategies,
                outlier_strategies=outlier_strategies,
                datatype_conversions=datatype_conversions
            )

            # Save clean dataset file inside Paths.CLEANED_DIR
            clean_filename = f"clean_{secure_filename(file_path.name)}"
            cleaned_path = Paths.CLEANED_DIR / clean_filename
            serialize_dataframe(cleaned_df, cleaned_path, fmt=file_path.suffix)

            # Calculate metrics for registering the clean file
            clean_size = cleaned_path.stat().st_size
            clean_hash = self.dataset_service.calculate_checksum(cleaned_path)
            clean_rows, clean_cols = cleaned_df.shape

            # Register or update clean dataset record
            clean_record = self.dataset_repo.get_by_hash(clean_hash)
            if not clean_record:
                clean_record = DatasetRecord(
                    filename=clean_filename,
                    filepath=str(cleaned_path),
                    file_hash=clean_hash,
                    file_size_bytes=clean_size,
                    row_count=clean_rows,
                    column_count=clean_cols,
                    status="cleaned"
                )
                self.dataset_repo.create(clean_record)
            else:
                clean_record.status = "cleaned"
                clean_record.row_count = clean_rows
                clean_record.column_count = clean_cols
            self.db.commit()

            # 4. Dataset Profiling
            logger.info("Step 4: Executing data profiling summaries...")

            clean_target_col = None
            if target_column:
                import re
                clean_target_col = str(target_column).strip().lower().replace(" ", "_")
                clean_target_col = re.sub(r"[^\w\-]", "", clean_target_col)
                if clean_target_col not in cleaned_df.columns:
                    clean_target_col = None

            profile = self.profile(cleaned_df, clean_target_col)

            # 5. Extract and persist detailed column metadata details for cleaned dataset
            MetadataService.extract_metadata(
                df=cleaned_df,
                dataset_id=clean_record.id,
                filename=clean_record.filename,
                filepath=clean_record.filepath,
                file_hash=clean_record.file_hash,
                file_size_bytes=clean_record.file_size_bytes,
                status="cleaned"
            )

            # Finalize execution log as complete
            duration = time.time() - start_time
            self.log_repo.update_status(
                log_id=log_record.id,
                status="completed",
                duration_seconds=duration
            )
            self.db.commit()

            logger.info(f"DataIntelligenceAgent: Pipeline completed successfully in {round(duration, 4)}s")
            return DataIntelligenceResult(
                dataset_id=clean_record.id,
                is_valid=True,
                validation_report=v_report,
                cleaning_report=c_report,
                profile=profile,
                cleaned_filepath=str(cleaned_path),
                duration_seconds=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"DataIntelligenceAgent: Execution failure: {e}")
            self.log_repo.update_status(
                log_id=log_record.id,
                status="failed",
                duration_seconds=duration,
                error_message=f"Pipeline error: {str(e)}"
            )
            self.db.commit()
            raise DatasetException(f"Pipeline error executing DataIntelligenceAgent: {e}") from e
