"""Phase 12 performance and scalability optimization module exports."""

from src.optimization.config import OptimizationConfig
from src.optimization.lazy_loader import LazyLoader
from src.optimization.memory import downcast_dataframe, stream_large_dataset
from src.optimization.cache import OptimizationCache, compress_prompt_context
from src.optimization.parallel import parallel_map
from src.optimization.profiler import PerformanceProfiler
from src.optimization.benchmark import generate_benchmark_csv, IngestionBenchmark

__all__ = [
    "OptimizationConfig",
    "LazyLoader",
    "downcast_dataframe",
    "stream_large_dataset",
    "OptimizationCache",
    "compress_prompt_context",
    "parallel_map",
    "PerformanceProfiler",
    "generate_benchmark_csv",
    "IngestionBenchmark",
]
