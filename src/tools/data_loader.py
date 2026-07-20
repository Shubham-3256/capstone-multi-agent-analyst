"""Tool for loading datasets from CSV, Parquet, and Excel formats with format auto-detection."""

from pathlib import Path

import pandas as pd

from src.core.exceptions import DatasetException
from src.core.logger import get_logger
from src.optimization.config import OptimizationConfig
from src.optimization.memory import downcast_dataframe
from src.utils.validators import detect_file_encoding

logger = get_logger(__name__)


class DataLoader:
    """Tool class providing static loaders for reading tabular data formats."""

    @staticmethod
    def load_file(filepath: Path) -> pd.DataFrame:
        """Load a dataset file into a Pandas DataFrame with auto-format detection.

        Supports CSV, Parquet, and Excel formats.

        Args:
            filepath: Path to the dataset file on disk.

        Returns:
            pd.DataFrame: Ingested tabular data.

        Raises:
            DatasetException: If format is unsupported or read operations fail.
        """
        logger.info(f"DataLoader: Ingesting file from: {filepath}")
        if not filepath.exists() or not filepath.is_file():
            from src.core.paths import Paths

            fallback = Paths.UPLOAD_DIR / filepath.name
            if fallback.exists() and fallback.is_file():
                logger.info(
                    f"DataLoader: Resolved missing path '{filepath}' to fallback '{fallback}'"
                )
                filepath = fallback
            else:
                raise DatasetException(
                    f"Target dataset file does not exist: {filepath}"
                )

        suffix = filepath.suffix.lower()

        try:
            if suffix == ".csv":
                encoding = detect_file_encoding(filepath)
                logger.info(f"DataLoader: Ingesting CSV using encoding='{encoding}'")
                df = pd.read_csv(
                    filepath, encoding=encoding, memory_map=True, low_memory=True
                )

            elif suffix in {".parquet", ".pq"}:
                logger.info("DataLoader: Ingesting Parquet dataset")
                df = pd.read_parquet(filepath)

            elif suffix in {".xlsx", ".xls"}:
                logger.info("DataLoader: Ingesting Excel spreadsheet")
                # uses openpyxl as default engine for newer formats
                df = pd.read_excel(filepath)

            else:
                logger.error(f"DataLoader: Unsupported file extension: {suffix}")
                raise DatasetException(
                    message=f"Unsupported file format suffix: '{suffix}'",
                    details={"filepath": str(filepath)},
                )

            if OptimizationConfig.ENABLE_DOWNCASTING:
                df = downcast_dataframe(df, inplace=True)
            return df

        except Exception as e:
            if isinstance(e, DatasetException):
                raise
            logger.error(f"DataLoader: Ingestion failed for {filepath.name}: {e}")
            raise DatasetException(f"Error loading file {filepath.name}: {e}") from e
