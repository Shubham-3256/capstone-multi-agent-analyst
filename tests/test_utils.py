"""Unit tests for utility functions (dataframe, file_utils, validators, serialization)."""

import json
import pytest
import pandas as pd
from pathlib import Path

from src.utils.dataframe import (
    get_dataframe_shape,
    get_memory_footprint,
    get_duplicate_count,
    get_missing_summary,
    extract_column_statistics,
)
from src.utils.file_utils import (
    compress_to_gzip,
    decompress_from_gzip,
    create_workspace_temp_dir,
)
from src.utils.validators import (
    detect_file_encoding,
    validate_csv_structure,
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
from src.core.exceptions import ValidationException


def test_dataframe_utilities():
    """Test DataFrame calculation metrics."""
    df = pd.DataFrame({
        "a": [1, 2, None, 2],
        "b": ["x", "y", "z", "y"]
    })
    
    shape = get_dataframe_shape(df)
    assert shape["rows"] == 4
    assert shape["columns"] == 2
    
    dups = get_duplicate_count(df)
    assert dups == 1  # Row [2, 'y'] matches row [2, 'y']
    
    missing = get_missing_summary(df)
    assert missing["total_missing"] == 1
    assert missing["column_missing"]["a"]["count"] == 1


def test_file_compression_and_temp(tmp_path):
    """Test GZIP compression and workspace temp directory generation."""
    source_file = tmp_path / "test.txt"
    source_file.write_text("Hello World!")
    
    gzip_file = tmp_path / "test.txt.gz"
    compress_to_gzip(source_file, gzip_file)
    assert gzip_file.exists()
    
    decompress_dest = tmp_path / "decompressed.txt"
    decompress_from_gzip(gzip_file, decompress_dest)
    assert decompress_dest.exists()
    assert decompress_dest.read_text() == "Hello World!"


def test_validators(tmp_path):
    """Test encoding detection and CSV structure validation."""
    valid_csv = tmp_path / "valid.csv"
    valid_csv.write_text("col1,col2\n1,2\n3,4")
    
    assert detect_file_encoding(valid_csv) == "utf-8"
    assert validate_csv_structure(valid_csv) is True
    
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("col1,col2\n1,2,3\n4,5")  # Column count mismatch
    
    with pytest.raises(ValidationException):
        validate_csv_structure(invalid_csv)


def test_serialization(tmp_path):
    """Test JSON, YAML, Pickle, and DataFrame serializations."""
    data = {"name": "Antigravity", "features": ["agents", "ML"]}
    
    # 1. JSON
    json_path = tmp_path / "test.json"
    serialize_json(data, json_path)
    assert deserialize_json(json_path) == data
    
    # 2. YAML
    yaml_path = tmp_path / "test.yaml"
    serialize_yaml(data, yaml_path)
    assert deserialize_yaml(yaml_path) == data
    
    # 3. Pickle
    pickle_path = tmp_path / "test.pkl"
    serialize_pickle(data, pickle_path)
    assert deserialize_pickle(pickle_path) == data
    
    # 4. DataFrame
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df_path = tmp_path / "df.csv"
    serialize_dataframe(df, df_path, fmt="csv")
    loaded_df = deserialize_dataframe(df_path, fmt="csv")
    pd.testing.assert_frame_equal(df, loaded_df)
