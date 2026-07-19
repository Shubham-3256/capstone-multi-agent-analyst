"""Service for extracting semantic schema and structural metadata from datasets."""

import pandas as pd

from src.core.logger import get_logger
from src.schemas.dataset import ColumnInfo
from src.schemas.metadata import DatasetMetadata
from src.utils.dataframe import (
    extract_column_statistics,
    get_duplicate_count,
    get_memory_footprint,
    get_missing_summary,
)

logger = get_logger(__name__)


class MetadataService:
    """Service coordinates structural and descriptive metadata extraction from Pandas DataFrames."""

    @staticmethod
    def extract_metadata(
        df: pd.DataFrame,
        dataset_id: str,
        filename: str,
        filepath: str,
        file_hash: str,
        file_size_bytes: int,
        status: str = "uploaded",
    ) -> DatasetMetadata:
        """Analyze a DataFrame to construct a complete DatasetMetadata profile.

        Args:
            df: The active dataset represented as a Pandas DataFrame.
            dataset_id: UUID reference identifier.
            filename: Target naming alias.
            filepath: Location path on disk.
            file_hash: Checksum fingerprint.
            file_size_bytes: Physical bytes size on disk.
            status: Ingestion status value.

        Returns:
            DatasetMetadata: Generated data profile schema representation.
        """
        logger.info(f"Extracting detailed metadata schemas for dataset: {filename}")

        # 1. Dimensions shape
        rows, cols = df.shape

        # 2. DataFrame diagnostics
        mem_usage = get_memory_footprint(df)
        duplicates = get_duplicate_count(df)
        missing_dict = get_missing_summary(df)
        stats_dict = extract_column_statistics(df)

        # 3. Create ColumnInfo profile maps
        columns_profile = {}
        for col in df.columns:
            col_name = str(col)
            series = df[col]

            # Extract sample values (convert to JSON-serializable types, drop NaNs)
            samples = series.dropna().head(5).tolist()

            missing_info = missing_dict["column_missing"].get(
                col_name, {"count": 0, "percentage": 0.0}
            )
            non_null_count = int(series.notnull().sum())
            unique_count = int(series.nunique())

            columns_profile[col_name] = ColumnInfo(
                name=col_name,
                dtype=str(series.dtype),
                non_null_count=non_null_count,
                null_count=missing_info["count"],
                null_percentage=missing_info["percentage"],
                unique_count=unique_count,
                sample_values=samples,
                statistics=stats_dict.get(col_name, {}),
            )

        # 4. Construct response schema
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            filename=filename,
            filepath=filepath,
            file_hash=file_hash,
            file_size_bytes=file_size_bytes,
            row_count=rows,
            column_count=cols,
            columns=columns_profile,
            missing_value_count=missing_dict["total_missing"],
            duplicate_rows_count=duplicates,
            memory_usage_bytes=mem_usage,
            status=status,
        )

        logger.info("Metadata extraction completed successfully.")
        return metadata
