"""Serialization utilities supporting JSON, YAML, Pickle, and Pandas DataFrame formatting."""

import json
import pickle
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

from src.core.logger import get_logger
from src.core.exceptions import ProjectException

logger = get_logger(__name__)


# --- JSON Serialization ---
def serialize_json(data: Any, filepath: Path) -> None:
    """Serialize Python objects to a JSON file.

    Args:
        data: Python dictionary/list/values to serialize.
        filepath: Target destination file path.
    """
    logger.debug(f"Serializing to JSON: {filepath}")
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, default=str)
    except Exception as e:
        logger.error(f"JSON serialization failed: {e}")
        raise ProjectException(f"JSON serialization error: {e}") from e


def deserialize_json(filepath: Path) -> Any:
    """Load and parse JSON file contents.

    Args:
        filepath: Source file path.

    Returns:
        Any: Inferred Python dictionary or list structure.
    """
    logger.debug(f"Deserializing from JSON: {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file does not exist: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"JSON deserialization failed: {e}")
        raise ProjectException(f"JSON deserialization error: {e}") from e


# --- YAML Serialization ---
def serialize_yaml(data: Any, filepath: Path) -> None:
    """Serialize Python objects to a YAML file.

    Args:
        data: Object to serialize.
        filepath: Target destination file path.
    """
    logger.debug(f"Serializing to YAML: {filepath}")
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
    except Exception as e:
        logger.error(f"YAML serialization failed: {e}")
        raise ProjectException(f"YAML serialization error: {e}") from e


def deserialize_yaml(filepath: Path) -> Any:
    """Load and parse YAML file contents.

    Args:
        filepath: Source file path.

    Returns:
        Any: Loaded Python object structure.
    """
    logger.debug(f"Deserializing from YAML: {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(f"YAML file does not exist: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"YAML deserialization failed: {e}")
        raise ProjectException(f"YAML deserialization error: {e}") from e


# --- Pickle Serialization ---
def serialize_pickle(data: Any, filepath: Path) -> None:
    """Pickle serialize a Python object to disk.

    Args:
        data: Object to pickle.
        filepath: Target destination file path.
    """
    logger.debug(f"Serializing to Pickle: {filepath}")
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.error(f"Pickle serialization failed: {e}")
        raise ProjectException(f"Pickle serialization error: {e}") from e


def deserialize_pickle(filepath: Path) -> Any:
    """Load a pickled Python object from disk.

    Args:
        filepath: Source pickle file path.

    Returns:
        Any: Loaded Python object.
    """
    logger.debug(f"Deserializing from Pickle: {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file does not exist: {filepath}")
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Pickle deserialization failed: {e}")
        raise ProjectException(f"Pickle deserialization error: {e}") from e


# --- DataFrame Serialization ---
def serialize_dataframe(df: pd.DataFrame, filepath: Path, fmt: str = "csv") -> None:
    """Serialize a Pandas DataFrame to a chosen file format.

    Args:
        df: Input Pandas DataFrame.
        filepath: Target file path.
        fmt: File format choice (csv, parquet, json, excel).
    """
    logger.info(f"Serializing DataFrame to {fmt.upper()}: {filepath}")
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fmt_lower = fmt.lower().strip(".")
        
        if fmt_lower == "csv":
            df.to_csv(filepath, index=False)
        elif fmt_lower == "parquet":
            df.to_parquet(filepath, index=False)
        elif fmt_lower == "json":
            df.to_json(filepath, orient="records", indent=4)
        elif fmt_lower in {"excel", "xlsx"}:
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported DataFrame format: {fmt}")
    except Exception as e:
        logger.error(f"DataFrame serialization failed: {e}")
        raise ProjectException(f"DataFrame serialization error: {e}") from e


def deserialize_dataframe(filepath: Path, fmt: str = "csv") -> pd.DataFrame:
    """Load a file into a Pandas DataFrame based on format choice.

    Args:
        filepath: Source file path.
        fmt: Inferred format choice (csv, parquet, json, excel).

    Returns:
        pd.DataFrame: Loaded table.
    """
    logger.info(f"Deserializing DataFrame from {fmt.upper()}: {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(f"DataFrame file does not exist: {filepath}")
        
    try:
        fmt_lower = fmt.lower().strip(".")
        if fmt_lower == "csv":
            return pd.read_csv(filepath)
        elif fmt_lower == "parquet":
            return pd.read_parquet(filepath)
        elif fmt_lower == "json":
            return pd.read_json(filepath)
        elif fmt_lower in {"excel", "xlsx"}:
            return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported DataFrame format: {fmt}")
    except Exception as e:
        logger.error(f"DataFrame deserialization failed: {e}")
        raise ProjectException(f"DataFrame deserialization error: {e}") from e
