"""Memory downcasting and chunk-based streaming optimizations for large datasets."""

from collections.abc import Generator
from pathlib import Path

import pandas as pd
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
)

from src.core.logger import get_logger
from src.optimization.config import OptimizationConfig

logger = get_logger(__name__)


def downcast_dataframe(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Downcast numeric types and convert categorical fields to minimize DataFrame size in memory.

    Args:
        df: Input pandas DataFrame to compress.
        inplace: Whether to modify the DataFrame in-place or make a shallow copy.

    Returns:
        pd.DataFrame: Memory-optimized DataFrame.
    """
    initial_memory = df.memory_usage(deep=True).sum()

    # Process column-by-column to avoid high memory spikes from full copies
    working_df = df if inplace else df.copy(deep=False)

    num_rows = len(working_df)
    if num_rows == 0:
        return working_df

    for col in working_df.columns:
        col_type = working_df[col].dtype

        # 1. Downcast Integers
        if is_integer_dtype(col_type):
            working_df[col] = pd.to_numeric(working_df[col], downcast="integer")

        # 2. Downcast Floats
        elif is_float_dtype(col_type):
            # Downcast to float32 is safe and saves 50% memory
            working_df[col] = pd.to_numeric(working_df[col], downcast="float")

        # 3. Categorical Conversion
        elif (is_object_dtype(col_type) or is_string_dtype(col_type)) and not isinstance(col_type, pd.CategoricalDtype):
            unique_count = working_df[col].nunique()
            # If unique values are small fraction of total, convert to category
            if unique_count / num_rows < OptimizationConfig.CATEGORY_CARDINALITY_THRESHOLD:
                working_df[col] = working_df[col].astype("category")

    final_memory = working_df.memory_usage(deep=True).sum()
    saved = initial_memory - final_memory
    saving_pct = (saved / initial_memory) * 100 if initial_memory > 0 else 0.0
    logger.info(f"Memory Optimization: Compressed from {initial_memory / (1024*1024):.2f} MB to {final_memory / (1024*1024):.2f} MB ({saving_pct:.1f}% reduction)")

    return working_df


def stream_large_dataset(filepath: Path, chunk_size: int = OptimizationConfig.CHUNK_SIZE) -> Generator[pd.DataFrame, None, None]:
    """Yield compressed chunks of a large dataset iteratively to avoid full memory loading.

    Args:
        filepath: Source data file path (only supports CSV for chunk streaming).
        chunk_size: Row batch counts per yield.

    Yields:
        pd.DataFrame: Memory-downcasted chunks.
    """
    suffix = filepath.suffix.lower()
    if suffix != ".csv":
        # Fall back to single-load if not CSV
        df = pd.read_csv(filepath) if suffix == ".csv" else pd.read_parquet(filepath) if suffix in {".parquet", ".pq"} else pd.read_excel(filepath)
        yield downcast_dataframe(df, inplace=True)
        return

    logger.info(f"Streaming CSV in chunks of {chunk_size} rows: {filepath.name}")
    try:
        # Use pandas chunk iterator
        with pd.read_csv(filepath, chunksize=chunk_size) as reader:
            for chunk in reader:
                yield downcast_dataframe(chunk, inplace=True)
    except Exception as e:
        logger.error(f"Error occurred during dataset streaming: {e}")
        raise
