#!/usr/bin/env python3
"""
HSM Session Detection — Framework Comparison Benchmark

Runs the same workload (gap-based L0 activity sessions + velocity-threshold
L1 movement sessions + summaries) across PySpark, Pandas, Polars, DuckDB,
and Dask, then prints a comparison table with throughput, memory, and a
"capacity" analysis.

Usage:
    python benchmarks/run_comparison.py --data-path ./data/robotics_demo
"""

import argparse
import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.competitors import BenchmarkResult


# ── PySpark benchmark ──────────────────────────────────────────────────────────

def run_pyspark_benchmark(data_path: str) -> BenchmarkResult:
    """Run PySpark session detection and return a BenchmarkResult."""
    import psutil
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from hsm_pyspark.spark.session_detector import SessionDetector, SessionConfig

    proc = psutil.Process()
    mem_before = proc.memory_info().rss

    t_total = time.perf_counter()

    spark = (
        SparkSession.builder.appName("HSM-Benchmark-PySpark")
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Load
    t_load = time.perf_counter()
    df = spark.read.parquet(data_path)
    total_records = df.count()
    load_time = time.perf_counter() - t_load

    # Process
    config = SessionConfig(
        entity_key="robot_id",
        timestamp_col="timestamp",
        signal_col="signal_type",
        value_col="signal_value",
    )
    detector = SessionDetector(config)

    t_proc = time.perf_counter()

    df_with_activity = detector.detect_gap_sessions(df, gap_threshold_seconds=300.0, session_name="activity")
    df_with_movement = detector.detect_threshold_sessions(
        df_with_activity,
        signal_type="velocity",
        threshold_condition="> 0.3",
        sustained_seconds=2.0,
        session_name="movement",
    )

    activity_summary = detector.summarize_sessions(
        df_with_movement, "activity_session_id", agg_cols=[("signal_value", "avg")]
    )
    movement_summary = detector.summarize_sessions(
        df_with_movement.filter(F.col("movement_session_id").isNotNull()),
        "movement_session_id",
        agg_cols=[("signal_value", "max")],
    )

    activity_count = activity_summary.count()
    movement_count = movement_summary.count()

    proc_time = time.perf_counter() - t_proc
    total_time = time.perf_counter() - t_total

    mem_after = proc.memory_info().rss
    peak_memory_mb = (mem_after - mem_before) / 1024 / 1024

    spark.stop()

    return BenchmarkResult(
        framework="PySpark",
        total_records=total_records,
        activity_sessions=activity_count,
        movement_sessions=movement_count,
        load_time_s=load_time,
        processing_time_s=proc_time,
        total_time_s=total_time,
        throughput_events_per_sec=total_records / proc_time,
        peak_memory_mb=max(0.0, peak_memory_mb),  # psutil RSS delta (Python heap only)
    )


# ── Runner helpers ─────────────────────────────────────────────────────────────

def _run_with_label(label: str, fn, *args, **kwargs) -> BenchmarkResult | None:
    print(f"\n  Running {label}...", flush=True)
    try:
        result = fn(*args, **kwargs)
        print(
            f"  ✓ {label}: {result.total_time_s:.2f}s total "
            f"({result.throughput_events_per_sec:,.0f} events/sec)",
            flush=True,
        )
        return result
    except Exception as exc:
        print(f"  ✗ {label} FAILED: {exc}", flush=True)
        return None


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt_memory(mb: float) -> str:
    if mb < 0:
        return "N/A"
    if mb < 0.5:
        return "< 1 *"   # native allocator not tracked by tracemalloc
    return f"{mb:,.0f}"


def print_comparison_table(results: list[BenchmarkResult]) -> None:
    """Print a formatted comparison table to stdout."""
    try:
        from tabulate import tabulate
    except ImportError:
        print("[tabulate not installed — install with: pip install tabulate]")
        _print_plain(results)
        return

    if not results:
        print("No results to display.")
        return

    slowest_time = max(r.processing_time_s for r in results)
    pyspark = next((r for r in results if r.framework == "PySpark"), None)
    pyspark_throughput = pyspark.throughput_events_per_sec if pyspark else 1.0
    total_records = results[0].total_records

    sorted_results = sorted(results, key=lambda r: r.processing_time_s)

    headers = [
        "Framework",
        "Time (s)",
        "Throughput\n(events/s)",
        "Memory\n(MB)",
        "vs PySpark",
        f"Capacity\n(in {slowest_time:.1f}s budget)",
    ]

    rows = []
    for r in sorted_results:
        capacity = int(r.throughput_events_per_sec * slowest_time)
        if r.framework == "PySpark":
            vs_spark = "baseline"
        else:
            ratio = r.throughput_events_per_sec / pyspark_throughput
            vs_spark = f"{ratio:.1f}x"
        rows.append(
            [
                r.framework,
                f"{r.processing_time_s:.2f}",
                f"{r.throughput_events_per_sec:>12,.0f}",
                _fmt_memory(r.peak_memory_mb),
                vs_spark,
                f"{capacity:,}",
            ]
        )

    print()
    print("=" * 90)
    print(f"  HSM SESSION DETECTION — FRAMEWORK COMPARISON  ({total_records:,} records)")
    print("=" * 90)
    print(tabulate(rows, headers=headers, tablefmt="double_grid", numalign="right"))
    print()
    print(
        f"  Budget = slowest processing time ({slowest_time:.2f}s). "
        "Capacity = how many records each framework\n"
        "  could process within that budget at its measured throughput.\n"
        "  * Memory < 1 MB: framework uses a native allocator (Rust/C++) not tracked\n"
        "    by Python tracemalloc. PySpark reports psutil RSS delta (Python heap only)."
    )
    print()


def _print_plain(results: list[BenchmarkResult]) -> None:
    """Fallback plain-text table if tabulate is unavailable."""
    print(f"\n{'Framework':<12} {'Time(s)':>8} {'Throughput':>12} {'Mem(MB)':>9} {'vsSpark':>9}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x.processing_time_s):
        print(
            f"{r.framework:<12} {r.processing_time_s:>8.2f} "
            f"{r.throughput_events_per_sec:>12,.0f} "
            f"{_fmt_memory(r.peak_memory_mb):>9}"
        )


def save_results(results: list[BenchmarkResult], output_dir: str) -> Path:
    """Save results to a timestamped JSON file."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"comparison_{ts}.json"

    payload = {
        "generated_at": ts,
        "results": [r.__dict__ for r in results],
    }
    out_file.write_text(json.dumps(payload, indent=2))
    return out_file


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HSM Session Detection — Framework Comparison Benchmark"
    )
    parser.add_argument(
        "--data-path",
        default="./data/robotics_demo",
        help="Directory containing parquet files (default: ./data/robotics_demo)",
    )
    parser.add_argument(
        "--skip-pyspark",
        action="store_true",
        help="Skip PySpark benchmark (useful when Spark is not configured)",
    )
    parser.add_argument(
        "--skip-dask",
        action="store_true",
        help="Skip Dask benchmark",
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmarks/results",
        help="Directory to save JSON results (default: ./benchmarks/results)",
    )
    args = parser.parse_args()

    data_path = str(Path(args.data_path).resolve())
    if not Path(data_path).exists():
        print(f"ERROR: data path not found: {data_path}")
        print("Run the demo first:  python scripts/run_demo.py --local --robots 5 --hours 1")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("HSM-PySpark: Framework Comparison Benchmark")
    print("=" * 60)
    print(f"Data path : {data_path}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    results: list[BenchmarkResult] = []

    # ── PySpark (baseline) ────────────────────────────────────────────────────
    if not args.skip_pyspark:
        r = _run_with_label("PySpark", run_pyspark_benchmark, data_path)
        if r:
            results.append(r)
    else:
        print("\n  [PySpark skipped]")

    # ── Pandas ────────────────────────────────────────────────────────────────
    from benchmarks.competitors import pandas_benchmark
    r = _run_with_label("Pandas", pandas_benchmark.run, data_path)
    if r:
        results.append(r)

    # ── Polars ────────────────────────────────────────────────────────────────
    from benchmarks.competitors import polars_benchmark
    r = _run_with_label("Polars", polars_benchmark.run, data_path)
    if r:
        results.append(r)

    # ── DuckDB ────────────────────────────────────────────────────────────────
    from benchmarks.competitors import duckdb_benchmark
    r = _run_with_label("DuckDB", duckdb_benchmark.run, data_path)
    if r:
        results.append(r)

    # ── Dask ─────────────────────────────────────────────────────────────────
    if not args.skip_dask:
        from benchmarks.competitors import dask_benchmark
        r = _run_with_label("Dask", dask_benchmark.run, data_path)
        if r:
            results.append(r)
    else:
        print("\n  [Dask skipped]")

    if not results:
        print("\nNo benchmarks succeeded.")
        sys.exit(1)

    # ── Report ────────────────────────────────────────────────────────────────
    print_comparison_table(results)

    out_file = save_results(results, args.output_dir)
    print(f"  Results saved → {out_file}\n")


if __name__ == "__main__":
    main()
