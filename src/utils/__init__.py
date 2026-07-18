"""Utilities module exports for data helpers and serializations."""

from src.utils.dataframe import (
    get_dataframe_shape,
    get_memory_footprint,
    get_duplicate_count,
    get_missing_summary,
    extract_column_statistics,
)
from src.utils.file_utils import (
    is_allowed_file,
    calculate_file_sha256,
    compress_to_gzip,
    decompress_from_gzip,
    create_workspace_temp_dir,
)
from src.utils.validators import (
    detect_file_encoding,
    validate_csv_structure,
    validate_parquet_structure,
    validate_excel_structure,
)
from src.utils.serialization import (
    serialize_json,
    deserialize_json,
    serialize_yaml,
    deserialize_yaml,
    serialize_pickle,
    deserialize_pickle,
    serialize_dataframe,
    deserialize_dataframe,
)

__all__ = [
    "get_dataframe_shape",
    "get_memory_footprint",
    "get_duplicate_count",
    "get_missing_summary",
    "extract_column_statistics",
    "is_allowed_file",
    "calculate_file_sha256",
    "compress_to_gzip",
    "decompress_from_gzip",
    "create_workspace_temp_dir",
    "detect_file_encoding",
    "validate_csv_structure",
    "validate_parquet_structure",
    "validate_excel_structure",
    "serialize_json",
    "deserialize_json",
    "serialize_yaml",
    "deserialize_yaml",
    "serialize_pickle",
    "deserialize_pickle",
    "serialize_dataframe",
    "deserialize_dataframe",
]
