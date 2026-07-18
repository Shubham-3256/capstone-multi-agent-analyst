"""Profiler tool for analyzing dataset structure, column types, and data quality metrics."""

from pathlib import Path
import pandas as pd

from src.core.logger import get_logger
from src.core.exceptions import DatasetException
from src.schemas.dataset import ColumnInfo, DatasetSummary
from src.utils.dataframe import (
    get_duplicate_count,
    get_memory_footprint,
    get_missing_summary,
    extract_column_statistics,
)
from src.tools.data_loader import DataLoader

logger = get_logger(__name__)


class DatasetProfiler:
    """Tool class to inspect, profiles, and generate tabular summaries from datasets."""

    @staticmethod
    def profile_dataframe(df: pd.DataFrame, filename: str, file_size_bytes: int = 0) -> DatasetSummary:
        """Analyze a DataFrame structure and build a detailed DatasetSummary schema.

        Args:
            df: Input Pandas DataFrame to inspect.
            filename: Target file name.
            file_size_bytes: Optional physical storage size.

        Returns:
            DatasetSummary: Populated profiling schema model.
        """
        logger.info(f"DatasetProfiler: Generating profile summary for {filename}")
        
        try:
            rows, cols = df.shape
            mem_usage = get_memory_footprint(df)
            duplicates = get_duplicate_count(df)
            missing_dict = get_missing_summary(df)
            stats_dict = extract_column_statistics(df)

            columns_profile = []
            for col in df.columns:
                col_name = str(col)
                series = df[col]
                
                # Fetch non-null, unique counts and samples
                non_null_count = int(series.notnull().sum())
                unique_count = int(series.nunique())
                samples = series.dropna().head(5).tolist()
                
                missing_info = missing_dict["column_missing"].get(col_name, {"count": 0, "percentage": 0.0})
                
                col_info = ColumnInfo(
                    name=col_name,
                    dtype=str(series.dtype),
                    non_null_count=non_null_count,
                    null_count=missing_info["count"],
                    null_percentage=missing_info["percentage"],
                    unique_count=unique_count,
                    sample_values=samples,
                    statistics=stats_dict.get(col_name, {})
                )
                columns_profile.append(col_info)

            summary = DatasetSummary(
                filename=filename,
                row_count=rows,
                column_count=cols,
                file_size_bytes=file_size_bytes,
                columns=columns_profile,
                duplicate_rows_count=duplicates,
                memory_usage_bytes=mem_usage
            )
            
            logger.info("DatasetProfiler: Summary profile created successfully.")
            return summary
            
        except Exception as e:
            logger.error(f"DatasetProfiler: Profiling failure: {e}")
            raise DatasetException(f"Error profiling DataFrame: {e}") from e

    @staticmethod
    def profile_file(filepath: Path) -> DatasetSummary:
        """Load a file, inspect its schema, and return a Pydantic DatasetSummary.

        Args:
            filepath: Path to the target CSV/Parquet/Excel dataset file.

        Returns:
            DatasetSummary: Constructed metadata profile.
        """
        logger.info(f"DatasetProfiler: Profiling file target: {filepath}")
        if not filepath.exists() or not filepath.is_file():
            raise FileNotFoundError(f"Target profiling file does not exist: {filepath}")
            
        try:
            # 1. Gather stats
            file_size = filepath.stat().st_size
            filename = filepath.name
            
            # 2. Ingest
            df = DataLoader.load_file(filepath)
            
            # 3. Profile
            return DatasetProfiler.profile_dataframe(df, filename, file_size)
            
        except Exception as e:
            logger.error(f"DatasetProfiler: Failed to profile file: {e}")
            raise DatasetException(f"Error profiling file {filepath.name}: {e}") from e
