"""Parallel processing executors for ML training and independent pipeline calculations."""

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TypeVar

from src.core.logger import get_logger
from src.optimization.config import OptimizationConfig

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    func: Callable[[T], R],
    items: list[T],
    max_workers: int = OptimizationConfig.PARALLEL_JOBS,
    use_processes: bool = False,
) -> list[R]:
    """Execute a mapping function over a collection of items in parallel.

    Args:
        func: Callback function executing one task.
        items: List of inputs to process.
        max_workers: Concurrent workers count limits (-1 maps to cpu_count).
        use_processes: Use ProcessPoolExecutor if True (CPU-bound), ThreadPoolExecutor if False (I/O).

    Returns:
        List[R]: Aggregated outputs in order.
    """
    if not items:
        return []

    # Resolve workers count
    if max_workers == -1:
        workers = os.cpu_count() or 4
    else:
        workers = max_workers

    # If single-worker, bypass thread/process creation overhead
    if workers <= 1 or len(items) == 1:
        return [func(item) for item in items]

    logger.info(
        f"Parallel Execution: Processing {len(items)} items using {workers} workers (mode={'processes' if use_processes else 'threads'})"
    )

    results: list[R] = []
    if use_processes:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(func, items))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(func, items))

    return results
