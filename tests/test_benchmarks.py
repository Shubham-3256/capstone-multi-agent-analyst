"""Unit tests for the performance scaling benchmark suite."""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.benchmark import generate_benchmark_csv, IngestionBenchmark


def test_generate_benchmark_csv(tmp_path: Path) -> None:
    """Test generating a mock dataset CSV file of target size."""
    file_path = tmp_path / "test_benchmark.csv"
    # Generate a tiny 0.1 MB file to keep tests fast
    generate_benchmark_csv(file_path, 0.1)
    
    assert file_path.exists()
    assert file_path.stat().st_size > 0


def test_ingestion_benchmark_suite(tmp_path: Path) -> None:
    """Test running the benchmark suite on a small dataset size."""
    # Use temporary directory for benchmark files
    benchmark_harness = IngestionBenchmark(workspace_dir=tmp_path)
    
    # Run suite for a tiny 0.1 MB file
    results = benchmark_harness.run_suite([0.1])
    
    assert 0.1 in results
    size_results = results[0.1]
    
    assert "raw_load_time" in size_results
    assert "opt_load_time" in size_results
    assert "mem_savings_pct" in size_results
    assert "speedup_factor" in size_results
