"""System scaling benchmark harness generating datasets and measuring ingestion efficiency."""

import gc
from pathlib import Path
import time
from typing import Any, Dict, List
import pandas as pd
from src.core.logger import get_logger
from src.optimization.config import OptimizationConfig
from src.optimization.memory import downcast_dataframe
from src.optimization.profiler import PerformanceProfiler

logger = get_logger(__name__)


def generate_benchmark_csv(dest_path: Path, size_mb: float) -> None:
    """Generate a dummy CSV dataset of a target size on disk.

    Args:
        dest_path: Target file path to write.
        size_mb: Intended size in MB.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1 row is approx 50 bytes of sample structure
    row_template = "12345,12.34567,class_a,9876543210,0\n"
    header = "id,value,category,large_num,target\n"
    
    row_bytes = len(row_template.encode("utf-8"))
    total_bytes = int(size_mb * 1024 * 1024)
    row_count = total_bytes // row_bytes

    logger.info(f"Generating Benchmark CSV: {dest_path.name} (approx {size_mb} MB, {row_count:,} rows)")
    
    # Write in chunks of 50,000 rows for memory efficiency
    chunk_size = 50_000
    written_rows = 0
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(header)
        while written_rows < row_count:
            batch = min(chunk_size, row_count - written_rows)
            f.write(row_template * batch)
            written_rows += batch

    actual_size = dest_path.stat().st_size / (1024 * 1024)
    logger.info(f"Finished writing dataset. Actual size: {actual_size:.2f} MB")


class IngestionBenchmark:
    """Runs benchmarks comparing standard pandas ingestion versus memory-optimized downcasting."""

    def __init__(self, workspace_dir: Path = OptimizationConfig.BENCHMARK_DIR) -> None:
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def run_suite(self, sizes_mb: List[float]) -> Dict[float, Dict[str, Any]]:
        """Execute benchmarks across multiple file sizes.

        Args:
            sizes_mb: List of MB file sizes to generate and test.

        Returns:
            Dict[float, Dict[str, Any]]: Benchmark results mapped by file size.
        """
        results: Dict[float, Dict[str, Any]] = {}
        
        for size in sizes_mb:
            file_path = self.workspace_dir / f"benchmark_{size}mb.csv"
            generate_benchmark_csv(file_path, size)
            
            # Measure Unoptimized Ingestion
            gc.collect()
            prof_raw = PerformanceProfiler(f"Load Raw {size}MB")
            prof_raw.start()
            df_raw = pd.read_csv(file_path)
            raw_metrics = prof_raw.stop()
            raw_mem = df_raw.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Measure Optimized Ingestion
            gc.collect()
            prof_opt = PerformanceProfiler(f"Load Opt {size}MB")
            prof_opt.start()
            df_opt = pd.read_csv(file_path)
            downcast_dataframe(df_opt, inplace=True)
            opt_metrics = prof_opt.stop()
            opt_mem = df_opt.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Clean up datasets to free RAM
            del df_raw
            del df_opt
            gc.collect()
            
            try:
                file_path.unlink()
            except Exception:
                pass
                
            results[size] = {
                "raw_load_time": raw_metrics["elapsed_seconds"],
                "raw_mem_mb": round(raw_mem, 2),
                "opt_load_time": opt_metrics["elapsed_seconds"],
                "opt_mem_mb": round(opt_mem, 2),
                "mem_savings_pct": round(((raw_mem - opt_mem) / raw_mem) * 100, 1) if raw_mem > 0 else 0.0,
                "speedup_factor": round(raw_metrics["elapsed_seconds"] / opt_metrics["elapsed_seconds"], 2) if opt_metrics["elapsed_seconds"] > 0 else 1.0
            }
            
            logger.info(
                f"Benchmark Size {size}MB: RAM Reduction = {results[size]['mem_savings_pct']}%, "
                f"Speedup = {results[size]['speedup_factor']}x"
            )
            
        return results
