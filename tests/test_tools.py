"""Unit tests for tool modules (DataLoader, FileManager, DatasetProfiler)."""

import pytest
import pandas as pd
from pathlib import Path

from src.tools.data_loader import DataLoader
from src.tools.file_manager import FileManager
from src.tools.dataset_profiler import DatasetProfiler
from src.core.exceptions import DatasetException


@pytest.fixture
def mock_csv_file(tmp_path):
    """Fixture generating a valid dummy CSV dataset for profiling tests."""
    csv_file = tmp_path / "sample_data.csv"
    df = pd.DataFrame({
        "age": [32, 45, 22, 54],
        "salary": [50000.0, 75000.0, 42000.0, 96000.0],
        "city": ["New York", "Chicago", "Boston", "Chicago"]
    })
    df.to_csv(csv_file, index=False)
    return csv_file


def test_data_loader(mock_csv_file):
    """Test DataLoader format-detection and loading mechanics."""
    df = DataLoader.load_file(mock_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 3)
    assert list(df.columns) == ["age", "salary", "city"]


def test_dataset_profiler(mock_csv_file):
    """Test DatasetProfiler structural profiling metrics and Pydantic outputs."""
    summary = DatasetProfiler.profile_file(mock_csv_file)
    
    assert summary.filename == "sample_data.csv"
    assert summary.row_count == 4
    assert summary.column_count == 3
    assert len(summary.columns) == 3
    
    # Assert column specs
    age_col = next(col for col in summary.columns if col.name == "age")
    assert age_col.dtype.startswith("int")
    assert age_col.statistics["mean"] == 38.25
    assert age_col.statistics["min"] == 22.0
    
    city_col = next(col for col in summary.columns if col.name == "city")
    assert any(city_col.dtype.startswith(prefix) for prefix in ["object", "str", "string"])
    assert city_col.unique_count == 3


def test_file_manager_workspace_audit():
    """Test FileManager workspace auditing limits."""
    # Run storage audit
    summary = FileManager.get_workspace_disk_summary()
    assert "uploads" in summary
    assert "reports" in summary
    assert "total" in summary
    assert "file_count" in summary["total"]
