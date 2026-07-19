"""Exponential backoff retry decorator with random jitter."""

import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from src.core.logger import get_logger

logger = get_logger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions_to_retry: list[type[Exception]] = None
) -> Callable:
    """Decorator retrying execution with exponential backoff and random jitter.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial sleep duration in seconds before first retry.
        backoff_factor: Multiplier applied to delay after each failure.
        exceptions_to_retry: List of exception classes that trigger retries.

    Returns:
        Callable: Wrapped decorated function.
    """
    exceptions = exceptions_to_retry or [Exception]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if exception is type-matched
                    if not any(isinstance(e, exc_type) for exc_type in exceptions):
                        raise e

                    if attempt == max_retries:
                        logger.error(f"Retry: Function {func.__name__} failed after {max_retries} attempts. Raising error: {e}")
                        raise e

                    # Compute next delay with jitter (adds between 0-50% variation)
                    jitter = random.uniform(0, 0.5) * delay
                    sleep_time = delay + jitter

                    logger.warning(
                        f"Retry: Exception caught: {e}. "
                        f"Attempt {attempt + 1}/{max_retries} failed. "
                        f"Retrying in {round(sleep_time, 2)}s..."
                    )

                    time.sleep(sleep_time)
                    delay *= backoff_factor

        return wrapper
    return decorator
