"""Performance tests profiling memory footprint and peak memory usage of dataset parsing routines."""

import tracemalloc

import numpy as np
import pandas as pd

from src.optimization.memory import downcast_dataframe


def test_peak_memory_and_compress():
    """Verify that downcast_dataframe correctly compresses data types and profiles peak memory bounds."""
    # Create large dummy dataset with high precision float64 and int64 columns
    n_rows = 20000
    df = pd.DataFrame(
        {
            "int_col": np.random.randint(0, 100, n_rows),
            "float_col": np.random.randn(n_rows),
            "cat_col": ["category_name_long"] * n_rows,
        }
    )

    # Start tracking memory
    tracemalloc.start()

    # 1. Profile dataset compression
    initial_mem_bytes = df.memory_usage(deep=True).sum()

    compressed_df = downcast_dataframe(df)
    compressed_mem_bytes = compressed_df.memory_usage(deep=True).sum()

    # Assert data type compression was performed
    assert compressed_mem_bytes < initial_mem_bytes

    # 2. Trace memory allocation during data cleaning
    from src.agents.data_intelligence.cleaner import Cleaner

    snapshot1 = tracemalloc.take_snapshot()

    cleaned_df, _ = Cleaner.clean(df)

    snapshot2 = tracemalloc.take_snapshot()

    # Stop memory tracking
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Peak memory allocated must be reasonable (e.g. less than 50 MB for this operations sequence)
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 50.0
    assert cleaned_df.shape[0] == n_rows
