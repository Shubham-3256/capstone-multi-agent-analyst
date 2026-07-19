"""Performance measurement tools for tracking execution time, CPU load, and peak memory allocations."""

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

from src.core.logger import get_logger
from src.core.paths import Paths

logger = get_logger(__name__)


class PerformanceProfiler:
    """Instruments code segments to collect precise runtime metrics."""

    def __init__(self, name: str) -> None:
        """Initialize profiler.

        Args:
            name: Description label for this profiling session.
        """
        self.name = name
        self._start_time: float = 0.0
        self._start_cpu: float = 0.0
        self.metrics: dict[str, Any] = {}

    def start(self) -> None:
        """Start measuring resource metrics."""
        tracemalloc.start()
        self._start_time = time.perf_counter()
        self._start_cpu = time.process_time()
        logger.info(f"Profiler '{self.name}': Performance instrumentation started.")

    def stop(self) -> dict[str, Any]:
        """Stop tracking and compute usage deltas.

        Returns:
            Dict[str, Any]: Profiling report containing time and memory metrics.
        """
        end_time = time.perf_counter()
        end_cpu = time.process_time()

        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed_time = end_time - self._start_time
        cpu_time = end_cpu - self._start_cpu

        # Calculate approximate CPU load percent
        cpu_load = (cpu_time / elapsed_time) * 100 if elapsed_time > 0 else 0.0

        self.metrics = {
            "name": self.name,
            "elapsed_seconds": round(elapsed_time, 4),
            "cpu_seconds": round(cpu_time, 4),
            "cpu_load_percent": round(cpu_load, 2),
            "peak_memory_mb": round(peak_memory / (1024 * 1024), 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        logger.info(
            f"Profiler '{self.name}': Finished. Duration: {self.metrics['elapsed_seconds']}s, "
            f"Peak Memory: {self.metrics['peak_memory_mb']} MB, CPU: {self.metrics['cpu_load_percent']}%"
        )
        return self.metrics

    def export_report(self, output_filename: str | None = None) -> Path:
        """Export profiler metrics payload as JSON file.

        Args:
            output_filename: Output filename, defaults to name_report.json.

        Returns:
            Path: File path of the written report.
        """
        filename = output_filename or f"{self.name.lower().replace(' ', '_')}_report.json"
        log_dir = Paths.WORKSPACE_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        report_path = log_dir / filename

        try:
            report_path.write_text(json.dumps(self.metrics, indent=4), encoding="utf-8")
            logger.info(f"Performance Report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to export profiling report: {e}")

        return report_path
