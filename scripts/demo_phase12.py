"""Demo script for Phase 12 - Performance & Memory Optimization Suite Verification."""

import sys
from pathlib import Path

# Add project root directory to path to enable local package importing
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.benchmark import IngestionBenchmark
from src.optimization.profiler import PerformanceProfiler
from src.database.database import init_db


def main() -> None:
    """Run Phase 12 optimizations benchmark demo."""
    print("==========================================================")
    print("Multi-Agent AI Data Analyst - Phase 12 Optimizations Demo")
    print("==========================================================")

    # 1. Initialize SQLite Database
    init_db()

    # 2. Setup benchmark harness
    print("\n[Step 1] Initializing Performance Profiler...")
    prof = PerformanceProfiler("Phase 12 Ingestion Optimization Benchmark")
    prof.start()

    benchmark_harness = IngestionBenchmark()
    
    # We will benchmark standard sizes: 1 MB, 5 MB, and 10 MB.
    # (These are kept small for rapid demo, but show real performance ratios.
    #  You can easily pass larger numbers like [100, 500, 1000] for larger stress runs)
    benchmark_sizes = [1.0, 5.0, 10.0]
    
    print(f"\n[Step 2] Running ingestion benchmarks on sizes: {benchmark_sizes} MB...")
    results = benchmark_harness.run_suite(benchmark_sizes)
    
    # Stop overall profiling
    prof_metrics = prof.stop()
    prof.export_report("phase12_benchmark_summary.json")

    # 3. Print Comparison Report
    print("\n" + "=" * 80)
    print("PERFORMANCE & MEMORY OPTIMIZATION COMPARISON REPORT")
    print("=" * 80)
    print(f"{'File Size':<10} | {'Unoptimized Time':<18} | {'Optimized Time':<16} | {'Speedup':<10} | {'RAM Saved':<10}")
    print("-" * 80)
    
    for size, metrics in results.items():
        size_str = f"{size} MB"
        raw_time = f"{metrics['raw_load_time']:.4f}s"
        opt_time = f"{metrics['opt_load_time']:.4f}s"
        speedup = f"{metrics['speedup_factor']}x"
        ram_saved = f"{metrics['mem_savings_pct']}%"
        print(f"{size_str:<10} | {raw_time:<18} | {opt_time:<16} | {speedup:<10} | {ram_saved:<10}")
        
    print("=" * 80)
    print(f"Overall Benchmark Wall Time: {prof_metrics['elapsed_seconds']} seconds")
    print(f"Peak Python Memory usage: {prof_metrics['peak_memory_mb']} MB")
    print("=" * 80 + "\n")
    
    print("Phase 12 optimization demo execution finished successfully!")


if __name__ == "__main__":
    main()
