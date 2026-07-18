"""Safe helper for independent orchestration tasks."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict


def run_parallel(tasks: Dict[str, Callable[[], Any]]) -> Dict[str, Any]:
    """Run independent coordinator tasks concurrently and return keyed results."""
    with ThreadPoolExecutor(max_workers=len(tasks) or 1) as executor:
        futures = {name: executor.submit(task) for name, task in tasks.items()}
        return {name: future.result() for name, future in futures.items()}
