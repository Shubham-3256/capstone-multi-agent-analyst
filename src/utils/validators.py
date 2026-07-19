"""Data format validators for CSV, Parquet, and Excel encoding and structures."""

import csv
from pathlib import Path

import pandas as pd

from src.core.exceptions import ValidationException
from src.core.logger import get_logger

logger = get_logger(__name__)


def detect_file_encoding(filepath: Path) -> str:
    """Detect textual encoding (UTF-8, UTF-16, Latin-1, CP1252) of a dataset file.

    Args:
        filepath: Target file path.

    Returns:
        str: Inferred encoding name.
    """
    logger.debug(f"Detecting encoding for file: {filepath}")

    encodings = ["utf-8", "latin1", "cp1252", "utf-16"]
    for encoding in encodings:
        try:
            with open(filepath, encoding=encoding) as f:
                # Read a small subset of lines to test decoding
                f.read(4096)
            logger.debug(f"Detected encoding: {encoding} for {filepath.name}")
            return encoding
        except (UnicodeDecodeError, LookupError):
            continue

    # Default fallback
    logger.warning(
        f"Could not conclusively detect encoding for {filepath.name}. Falling back to utf-8."
    )
    return "utf-8"


def validate_csv_structure(filepath: Path) -> bool:
    """Validate if a file contains structurally valid CSV data (headers, separators, row bounds).

    Args:
        filepath: Target file path.

    Returns:
        bool: True if validation passes.

    Raises:
        ValidationException: If CSV formatting is broken.
    """
    logger.info(f"Validating CSV structure: {filepath}")
    encoding = detect_file_encoding(filepath)

    try:
        with open(filepath, encoding=encoding, newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)

            if not headers:
                raise ValidationException(
                    "CSV validation failed: Empty file or headers missing."
                )

            # Verify the row column counts are uniform on first 100 rows
            col_count = len(headers)
            for idx, row in enumerate(reader):
                if idx > 100:
                    break
                if len(row) != col_count:
                    raise ValidationException(
                        f"CSV validation failed: Inconsistent column count on line {idx + 2}. "
                        f"Expected {col_count}, got {len(row)}."
                    )
        return True
    except Exception as e:
        if isinstance(e, ValidationException):
            raise
        logger.error(f"CSV validation error: {e}")
        raise ValidationException(f"Invalid CSV layout: {e}") from e


def validate_parquet_structure(filepath: Path) -> bool:
    """Validate if a file is a structurally valid Parquet binary dataset.

    Args:
        filepath: Target file path.

    Returns:
        bool: True if valid.

    Raises:
        ValidationException: If Parquet loading fails.
    """
    logger.info(f"Validating Parquet structure: {filepath}")
    try:
        # Load schema header metadata only to verify binary format
        pd.read_parquet(filepath, nrows=1)
        return True
    except Exception as e:
        logger.error(f"Parquet validation error: {e}")
        raise ValidationException(f"Invalid Parquet database layout: {e}") from e


def validate_excel_structure(filepath: Path) -> bool:
    """Validate if a file is a structurally valid Excel spreadsheet.

    Args:
        filepath: Target file path.

    Returns:
        bool: True if valid.

    Raises:
        ValidationException: If Excel loading fails.
    """
    logger.info(f"Validating Excel structure: {filepath}")
    try:
        # Read engine parser check
        pd.read_excel(filepath, nrows=1)
        return True
    except Exception as e:
        logger.error(f"Excel validation error: {e}")
        raise ValidationException(f"Invalid Excel database layout: {e}") from e
