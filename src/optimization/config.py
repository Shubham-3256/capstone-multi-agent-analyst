"""Configuration settings for Phase 12 performance optimizations."""

from pathlib import Path
from src.core.paths import Paths


class OptimizationConfig:
    """Configuration parameters for memory downcasting, parallel execution, and caching."""

    CACHE_DIR: Path = Paths.WORKSPACE_DIR / "cache"
    BENCHMARK_DIR: Path = Paths.WORKSPACE_DIR / "benchmarks"
    
    # Memory Optimizations
    ENABLE_DOWNCASTING: bool = True
    CATEGORY_CARDINALITY_THRESHOLD: float = 0.05  # convert object columns to category if unique values < 5% of rows
    CHUNK_SIZE: int = 50_000  # default rows per chunk when streaming large datasets
    
    # Concurrency / Parallelism
    PARALLEL_JOBS: int = -1  # -1 means use all available cores, 1 means single-threaded
    
    # LLM & Business Insights Prompt Optimization
    ENABLE_PROMPT_CACHING: bool = True
    CONTEXT_COMPRESSION_RATIO: float = 0.8  # reduce context window token size
    
    # Visualizations Caching
    CACHE_FIGURES: bool = True
