"""Unit tests for memory downcasting, lazy loaders, and profiler utilities."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.lazy_loader import LazyLoader
from src.optimization.memory import downcast_dataframe
from src.optimization.profiler import PerformanceProfiler


def test_downcast_dataframe_numeric() -> None:
    """Test integer and float columns downcast correctly."""
    # Create non-compressed DataFrame
    df = pd.DataFrame(
        {
            "ints": pd.Series([1, 2, 3, 4, 5], dtype=np.int64),
            "floats": pd.Series([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64),
        }
    )

    initial_ints_dtype = df["ints"].dtype
    initial_floats_dtype = df["floats"].dtype

    compressed_df = downcast_dataframe(df, inplace=False)

    assert compressed_df["ints"].dtype != initial_ints_dtype
    # int64 should downcast to int8 for small numbers
    assert compressed_df["ints"].dtype == np.int8

    # float64 should downcast to float32
    assert compressed_df["floats"].dtype == np.float32


def test_downcast_dataframe_categorical() -> None:
    """Test low cardinality string/object columns convert to category dtype."""
    # 100 rows, only 2 unique strings (ratio = 2/100 = 0.02 < 0.05 threshold)
    df = pd.DataFrame({"city": ["London", "Paris"] * 50})

    compressed_df = downcast_dataframe(df, inplace=False)
    assert isinstance(compressed_df["city"].dtype, pd.CategoricalDtype)


def test_lazy_loader_proxy() -> None:
    """Test LazyLoader dynamically loads the underlying library on attribute access."""
    # We can lazily load hashlib or json from standard library
    lazy_json = LazyLoader("json")

    # Assert not imported eagerly
    assert lazy_json._module is None

    # Assert behaves like regular module on access
    test_dict = {"a": 1}
    serialized = lazy_json.dumps(test_dict)

    assert lazy_json._module is not None
    assert serialized == '{"a": 1}'
    assert lazy_json.loads(serialized) == test_dict


def test_performance_profiler() -> None:
    """Test PerformanceProfiler records timing and peak memory metrics."""
    prof = PerformanceProfiler("Test Run")
    prof.start()

    # Small computational delay
    total = sum(i * i for i in range(100_000))
    assert total > 0

    metrics = prof.stop()
    assert metrics["name"] == "Test Run"
    assert metrics["elapsed_seconds"] > 0
    assert "peak_memory_mb" in metrics
    assert "cpu_load_percent" in metrics

    report_path = prof.export_report()
    assert report_path.exists()
    try:
        report_path.unlink()
    except Exception:
        pass
