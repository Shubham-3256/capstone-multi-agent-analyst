"""Unit tests for thread-and-process-backed parallel execution utilities."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.parallel import parallel_map


def _square_number(x: int) -> int:
    """Helper target function to square a number in parallel workers."""
    return x * x


def test_parallel_map_threads() -> None:
    """Test parallel_map using threads."""
    inputs = [1, 2, 3, 4, 5]
    expected = [1, 4, 9, 16, 25]

    # Thread mapping
    results = parallel_map(_square_number, inputs, max_workers=2, use_processes=False)
    assert results == expected


def test_parallel_map_processes() -> None:
    """Test parallel_map using process workers."""
    inputs = [1, 2, 3, 4, 5]
    expected = [1, 4, 9, 16, 25]

    # Process mapping
    results = parallel_map(_square_number, inputs, max_workers=2, use_processes=True)
    assert results == expected


def test_parallel_map_single_worker() -> None:
    """Test parallel_map fallback for single-threaded tasks."""
    inputs = [10, 20]
    expected = [100, 400]

    results = parallel_map(_square_number, inputs, max_workers=1)
    assert results == expected
