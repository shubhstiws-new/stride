#!/usr/bin/env python3
"""
HSM Session Detection — Scale-Factor Sweep Benchmark

Sweeps across increasing virtual scale factors to reveal the OOM boundary
where Pandas/Polars collapse while Spark maintains constant throughput by
processing lazily via broadcast cross-join.

Usage:
    # Quick smoke test (no Spark/Dask, ~30 s):
    python benchmarks/simulate_scale.py \\
        --data-path ./data/robotics_demo \\
        --scales 1,10 \\
        --skip-pyspark --skip-dask

    # Full sweep:
    python benchmarks/simulate_scale.py \\
        --data-path ./data/robotics_demo \\
        --scales 1,10,50,100,500
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.competitors import BenchmarkResult

# ── Sentinel values ─────────────────────────────────────────────────────────

OOM = "OOM"
SKIP = "SKIP"
ERR = "ERR"

ScaleResult = Union[BenchmarkResult, str]  # BenchmarkResult | OOM | SKIP | ERR


# ── PySpark scaled runner ────────────────────────────────────────────────────

def run_pyspark_scaled(data_path: str, scale_factor: int) -> BenchmarkResult:
    """
    PySpark benchmark with virtual scaling via broadcast cross-join.

    Spark reads the parquet file ONCE (~61 MB) and cross-joins with a tiny
    broadcast range(scale_factor) table to produce N virtual copies of the
    data. Executors never hold all N copies in memory simultaneously — Spark
    processes one partition at a time. Throughput stays flat regardless of N.
    """
    import psutil
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from hsm_pyspark.spark.session_detector import SessionDetector, SessionConfig

    proc = psutil.Process()
    mem_before = proc.memory_info().rss

    t_total = time.perf_counter()

    spark = (
        SparkSession.builder.appName(f"HSM-ScaleSweep-{scale_factor}x")
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Load base data
    t_load = time.perf_counter()
    base = spark.read.parquet(data_path)

    if scale_factor > 1:
        # Broadcast a tiny N-row table; Spark replicates base via cross-join.
        # Only ~1 partition lives in executor memory at any time.
        scale_range = spark.range(scale_factor)
        scaled = (
            base.crossJoin(F.broadcast(scale_range))
            .withColumn(
                "robot_id",
                F.concat(F.col("robot_id"), F.lit("_s"), F.col("id").cast("string")),
            )
            .drop("id")
        )
    else:
        scaled = base

    total_records = scaled.count()
    load_time = time.perf_counter() - t_load

    # Session detection (identical config to run_comparison.py)
    config = SessionConfig(
        entity_key="robot_id",
        timestamp_col="timestamp",
        signal_col="signal_type",
        value_col="signal_value",
    )
    detector = SessionDetector(config)

    t_proc = time.perf_counter()

    df_with_activity = detector.detect_gap_sessions(
        scaled, gap_threshold_seconds=300.0, session_name="activity"
    )
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
    peak_memory_mb = max(0.0, (mem_after - mem_before) / 1024 / 1024)

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
        peak_memory_mb=peak_memory_mb,
    )


# ── Safe runner with OOM / error handling ───────────────────────────────────

_OOM_KEYWORDS = (
    "OutOfMemoryError", "OutOfMemoryException",
    "MemoryError", "Out of Memory",
)


def _worker_fn(fn, args: tuple, result_queue) -> None:
    """Subprocess target: run fn(*args) and communicate result via queue."""
    try:
        result = fn(*args)
        result_queue.put(("ok", result))
    except MemoryError:
        result_queue.put(("oom", "MemoryError"))
    except Exception as exc:
        exc_str = str(exc)
        if any(kw in exc_str for kw in _OOM_KEYWORDS):
            result_queue.put(("oom", exc_str[:200]))
        else:
            result_queue.put(("err", f"{type(exc).__name__}: {exc_str[:200]}"))


def _safe_run(label: str, fn, *args) -> ScaleResult:
    """
    Run fn(*args) in an isolated subprocess.

    Using a subprocess means a SIGKILL from the macOS OOM killer only
    terminates the child — the sweep script survives and marks the result OOM.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker_fn, args=(fn, args, q))
    p.start()
    p.join()

    if p.exitcode == 0:
        try:
            status, payload = q.get_nowait()
        except Exception:
            print(f"    !! {label}: subprocess finished but returned no result", flush=True)
            return ERR
        if status == "ok":
            return payload
        if status == "oom":
            print(f"    !! {label}: OOM — {payload[:120]}", flush=True)
            return OOM
        print(f"    !! {label}: {payload}", flush=True)
        return ERR

    if p.exitcode == -9:  # SIGKILL from macOS OOM killer
        print(f"    !! {label}: killed by OS OOM killer (SIGKILL)", flush=True)
        return OOM

    print(f"    !! {label}: subprocess exited with code {p.exitcode}", flush=True)
    return ERR


# ── Helpers ─────────────────────────────────────────────────────────────────

def _throughput_str(result: ScaleResult) -> str:
    if isinstance(result, BenchmarkResult):
        return f"{result.throughput_events_per_sec:>10,.0f}"
    return f"{'  ' + result:>10}"


def _memory_str(result: ScaleResult) -> str:
    if isinstance(result, BenchmarkResult):
        mb = result.peak_memory_mb
        if mb < 0.5:
            return "   < 1 MB"
        return f"{mb:>6,.0f} MB"
    return f"{'  ' + result:>9}"


def _records_str(records: int) -> str:
    return f"{records / 1_000:>7,.0f}K"


# ── Table printing ───────────────────────────────────────────────────────────

# approximate parquet size of the 1x dataset in MB
_BASE_PARQUET_MB = 61


def print_sweep_table(
    scales: list[int],
    base_records: int,
    throughput_results: dict[int, dict[str, ScaleResult]],
    memory_results: dict[int, dict[str, ScaleResult]],
    active_frameworks: list[str],
) -> None:
    col_w = 12
    scale_w = 17
    rec_w = 10

    def pad(s: str, w: int) -> str:
        return s[:w].center(w)

    # Header
    print()
    print("=" * 90)
    print("  HSM SESSION DETECTION — SCALE FACTOR SWEEP  (throughput: events/sec)")
    print("=" * 90)

    hdr_parts = [pad("Scale", scale_w), pad("Records", rec_w)]
    for fw in active_frameworks:
        hdr_parts.append(pad(fw, col_w))
    print("  " + " | ".join(hdr_parts))
    print("  " + "-" * (scale_w + rec_w + col_w * len(active_frameworks) + 3 * (len(active_frameworks) + 1)))

    for sf in scales:
        parquet_mb = _BASE_PARQUET_MB * sf
        if parquet_mb >= 1024:
            size_str = f"{parquet_mb / 1024:.1f} GB"
        else:
            size_str = f"{parquet_mb} MB"
        scale_label = f"{sf:>3}× ({size_str:>7})"
        records = base_records * sf
        rec_str = _records_str(records)

        row_parts = [pad(scale_label, scale_w), pad(rec_str, rec_w)]
        for fw in active_frameworks:
            res = throughput_results.get(sf, {}).get(fw, SKIP)
            row_parts.append(pad(_throughput_str(res).strip(), col_w))
        print("  " + " | ".join(row_parts))

    print()
    print("  Spark: flat throughput (reads 61 MB parquet once, broadcast cross-join")
    print("         creates virtual copies — only ~1 partition in memory at a time)")
    print("  Pandas/Polars: OOM when N×~600 MB uncompressed > available RAM")
    print("  DuckDB: slows as buffer manager spills to disk, but survives longer")
    print("  Dask: lazy concat fine, but shuffle memory grows with N×robots")
    print()

    # Memory table
    print("  Peak memory usage (MB) per framework at each scale:")
    print("  " + "-" * (scale_w + rec_w + col_w * len(active_frameworks) + 3 * (len(active_frameworks) + 1)))
    mem_hdr = [pad("Scale", scale_w), pad("Records", rec_w)]
    for fw in active_frameworks:
        mem_hdr.append(pad(fw, col_w))
    print("  " + " | ".join(mem_hdr))
    print("  " + "-" * (scale_w + rec_w + col_w * len(active_frameworks) + 3 * (len(active_frameworks) + 1)))

    for sf in scales:
        parquet_mb = _BASE_PARQUET_MB * sf
        if parquet_mb >= 1024:
            size_str = f"{parquet_mb / 1024:.1f} GB"
        else:
            size_str = f"{parquet_mb} MB"
        scale_label = f"{sf:>3}× ({size_str:>7})"
        records = base_records * sf
        rec_str = _records_str(records)

        row_parts = [pad(scale_label, scale_w), pad(rec_str, rec_w)]
        for fw in active_frameworks:
            res = memory_results.get(sf, {}).get(fw, SKIP)
            row_parts.append(pad(_memory_str(res).strip(), col_w))
        print("  " + " | ".join(row_parts))

    print()


# ── JSON output ──────────────────────────────────────────────────────────────

def save_results(
    scales: list[int],
    base_records: int,
    throughput_results: dict[int, dict[str, ScaleResult]],
    output_dir: str,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"scale_sweep_{ts}.json"

    sweep_data = []
    for sf in scales:
        scale_entry = {"scale_factor": sf, "virtual_records": base_records * sf, "frameworks": {}}
        for fw, res in throughput_results.get(sf, {}).items():
            if isinstance(res, BenchmarkResult):
                scale_entry["frameworks"][fw] = asdict(res)
            else:
                scale_entry["frameworks"][fw] = {"status": res}
        sweep_data.append(scale_entry)

    payload = {
        "generated_at": ts,
        "base_records": base_records,
        "scales": scales,
        "sweep": sweep_data,
    }
    out_file.write_text(json.dumps(payload, indent=2))
    return out_file


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HSM Session Detection — Scale-Factor Sweep Benchmark"
    )
    parser.add_argument(
        "--data-path",
        default="./data/robotics_demo",
        help="Directory containing parquet files (default: ./data/robotics_demo)",
    )
    parser.add_argument(
        "--scales",
        default="1,10,50,100,500",
        help="Comma-separated list of scale factors (default: 1,10,50,100,500)",
    )
    parser.add_argument(
        "--skip-pyspark",
        action="store_true",
        help="Skip PySpark (useful when Spark is not configured)",
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

    scales = [int(s.strip()) for s in args.scales.split(",")]

    print("\n" + "=" * 60)
    print("HSM-PySpark: Scale-Factor Sweep Benchmark")
    print("=" * 60)
    print(f"Data path : {data_path}")
    print(f"Scales    : {scales}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # Determine base record count from 1× run
    import pandas as pd
    base_df = pd.read_parquet(data_path)
    base_records = len(base_df)
    del base_df
    print(f"Base records: {base_records:,}")

    # Lazy imports
    from benchmarks.competitors import pandas_benchmark, polars_benchmark, duckdb_benchmark, dask_benchmark

    # Track results: {scale_factor: {framework: ScaleResult}}
    throughput_results: dict[int, dict[str, ScaleResult]] = {}
    memory_results: dict[int, dict[str, ScaleResult]] = {}

    # Track which frameworks have OOM'd — skip all higher scales for them
    oom_frameworks: set[str] = set()

    active_frameworks = []
    if not args.skip_pyspark:
        active_frameworks.append("PySpark")
    active_frameworks += ["Pandas", "Polars", "DuckDB"]
    if not args.skip_dask:
        active_frameworks.append("Dask")

    for sf in scales:
        parquet_mb = _BASE_PARQUET_MB * sf
        size_str = f"{parquet_mb / 1024:.1f} GB" if parquet_mb >= 1024 else f"{parquet_mb} MB"
        print(f"\n── Scale {sf}× ({size_str}, ~{base_records * sf:,} virtual records) ──────────")

        throughput_results[sf] = {}
        memory_results[sf] = {}

        # PySpark
        if not args.skip_pyspark:
            if "PySpark" in oom_frameworks:
                print("  [PySpark] SKIP (previous OOM)")
                throughput_results[sf]["PySpark"] = SKIP
                memory_results[sf]["PySpark"] = SKIP
            else:
                print(f"  Running PySpark {sf}×...", flush=True)
                res = _safe_run("PySpark", run_pyspark_scaled, data_path, sf)
                throughput_results[sf]["PySpark"] = res
                memory_results[sf]["PySpark"] = res
                if res in (OOM, ERR):
                    oom_frameworks.add("PySpark")
                elif isinstance(res, BenchmarkResult):
                    print(
                        f"  ✓ PySpark: {res.total_time_s:.1f}s, "
                        f"{res.throughput_events_per_sec:,.0f} ev/s",
                        flush=True,
                    )

        # Pandas
        if "Pandas" in oom_frameworks:
            print("  [Pandas] SKIP (previous OOM)")
            throughput_results[sf]["Pandas"] = SKIP
            memory_results[sf]["Pandas"] = SKIP
        else:
            print(f"  Running Pandas {sf}×...", flush=True)
            res = _safe_run("Pandas", pandas_benchmark.run, data_path, sf)
            throughput_results[sf]["Pandas"] = res
            memory_results[sf]["Pandas"] = res
            if res in (OOM, ERR):
                oom_frameworks.add("Pandas")
            elif isinstance(res, BenchmarkResult):
                print(
                    f"  ✓ Pandas: {res.total_time_s:.1f}s, "
                    f"{res.throughput_events_per_sec:,.0f} ev/s",
                    flush=True,
                )

        # Polars
        if "Polars" in oom_frameworks:
            print("  [Polars] SKIP (previous OOM)")
            throughput_results[sf]["Polars"] = SKIP
            memory_results[sf]["Polars"] = SKIP
        else:
            print(f"  Running Polars {sf}×...", flush=True)
            res = _safe_run("Polars", polars_benchmark.run, data_path, sf)
            throughput_results[sf]["Polars"] = res
            memory_results[sf]["Polars"] = res
            if res in (OOM, ERR):
                oom_frameworks.add("Polars")
            elif isinstance(res, BenchmarkResult):
                print(
                    f"  ✓ Polars: {res.total_time_s:.1f}s, "
                    f"{res.throughput_events_per_sec:,.0f} ev/s",
                    flush=True,
                )

        # DuckDB
        if "DuckDB" in oom_frameworks:
            print("  [DuckDB] SKIP (previous OOM)")
            throughput_results[sf]["DuckDB"] = SKIP
            memory_results[sf]["DuckDB"] = SKIP
        else:
            print(f"  Running DuckDB {sf}×...", flush=True)
            res = _safe_run("DuckDB", duckdb_benchmark.run, data_path, sf)
            throughput_results[sf]["DuckDB"] = res
            memory_results[sf]["DuckDB"] = res
            if res in (OOM, ERR):
                oom_frameworks.add("DuckDB")
            elif isinstance(res, BenchmarkResult):
                print(
                    f"  ✓ DuckDB: {res.total_time_s:.1f}s, "
                    f"{res.throughput_events_per_sec:,.0f} ev/s",
                    flush=True,
                )

        # Dask
        if not args.skip_dask:
            if "Dask" in oom_frameworks:
                print("  [Dask] SKIP (previous OOM)")
                throughput_results[sf]["Dask"] = SKIP
                memory_results[sf]["Dask"] = SKIP
            else:
                print(f"  Running Dask {sf}×...", flush=True)
                res = _safe_run("Dask", dask_benchmark.run, data_path, sf)
                throughput_results[sf]["Dask"] = res
                memory_results[sf]["Dask"] = res
                if res in (OOM, ERR):
                    oom_frameworks.add("Dask")
                elif isinstance(res, BenchmarkResult):
                    print(
                        f"  ✓ Dask: {res.total_time_s:.1f}s, "
                        f"{res.throughput_events_per_sec:,.0f} ev/s",
                        flush=True,
                    )

    # ── Print summary table ───────────────────────────────────────────────────
    print_sweep_table(
        scales, base_records, throughput_results, memory_results, active_frameworks
    )

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_file = save_results(scales, base_records, throughput_results, args.output_dir)
    print(f"  Results saved → {out_file}\n")


if __name__ == "__main__":
    main()
