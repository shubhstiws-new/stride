"""
cuDF Standalone GPU runner.

Mirrors the polars_benchmark.py pattern but uses cuDF (GPU DataFrame library).
Used by operation modules when framework == "cudf".

Standalone diagnostic usage:
    python benchmarks/frameworks/cudf_standalone.py --data-path ./data/behavior1k/filesize=1mb
"""

import argparse
import sys
import time
from pathlib import Path


def get_gpu_memory_used_mb(device_index: int = 0) -> float:
    """Return used GPU memory in MB via pynvml."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / 1024 / 1024
    except Exception:
        return 0.0


def run_io_scan(data_path: str) -> dict:
    """cuDF IO scan: read parquet + count rows."""
    import cudf

    t0 = time.perf_counter()
    df = cudf.read_parquet(str(Path(data_path) / "*.parquet"))
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    n = len(df)
    compute_time = time.perf_counter() - t1

    return {
        "operation": "io_scan",
        "total_records": n,
        "load_time_s": load_time,
        "compute_time_s": compute_time,
        "gpu_memory_mb": get_gpu_memory_used_mb(),
    }


def run_filter_agg(data_path: str, task_filter: list[str] | None = None) -> dict:
    """cuDF filter + group-by aggregation."""
    import cudf

    if task_filter is None:
        task_filter = ["0", "1", "2", "3", "4"]

    t0 = time.perf_counter()
    df = cudf.read_parquet(str(Path(data_path) / "*.parquet"))
    n = len(df)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    filtered = df[df["task_id"].isin(task_filter)]
    result = (
        filtered.groupby("episode_id")["signal_value"]
        .agg(["mean", "std", "count"])
    )
    result_rows = len(result)
    compute_time = time.perf_counter() - t1

    return {
        "operation": "filter_agg",
        "total_records": n,
        "result_rows": result_rows,
        "load_time_s": load_time,
        "compute_time_s": compute_time,
        "gpu_memory_mb": get_gpu_memory_used_mb(),
    }


def run_window_hsm(data_path: str, gap_threshold_s: float = 5.0) -> dict:
    """cuDF gap-based session detection."""
    import cudf

    t0 = time.perf_counter()
    df = cudf.read_parquet(str(Path(data_path) / "*.parquet"))
    n = len(df)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    df = df.sort_values(["episode_id", "timestamp"])
    df["_gap"] = df.groupby("episode_id")["timestamp"].diff()
    df["_is_new"] = ((df["_gap"].isna()) | (df["_gap"] > gap_threshold_s)).astype("int32")
    df["_sess_num"] = df.groupby("episode_id")["_is_new"].cumsum()
    df["session_id"] = df["episode_id"] + "_sess_" + df["_sess_num"].astype(str)

    summary = df.groupby("session_id")["timestamp"].agg(["min", "max", "count"])
    result_rows = len(summary)
    compute_time = time.perf_counter() - t1

    return {
        "operation": "window_hsm",
        "total_records": n,
        "result_rows": result_rows,
        "load_time_s": load_time,
        "compute_time_s": compute_time,
        "gpu_memory_mb": get_gpu_memory_used_mb(),
    }


def run_join(data_path: str) -> dict:
    """cuDF self-join on task_id."""
    import cudf

    t0 = time.perf_counter()
    df = cudf.read_parquet(str(Path(data_path) / "*.parquet"))
    n = len(df)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    pairs = df[["task_id", "episode_id"]].drop_duplicates()
    joined = pairs.merge(pairs, on="task_id", suffixes=("_a", "_b"))
    joined = joined[joined["episode_id_a"] != joined["episode_id_b"]]
    result_rows = len(joined)
    compute_time = time.perf_counter() - t1

    return {
        "operation": "join",
        "total_records": n,
        "result_rows": result_rows,
        "load_time_s": load_time,
        "compute_time_s": compute_time,
        "gpu_memory_mb": get_gpu_memory_used_mb(),
    }


# ── Diagnostic CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="cuDF standalone diagnostic — run all 4 operations on a dataset"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Directory containing Parquet files (e.g., ./data/behavior1k/filesize=1mb)",
    )
    parser.add_argument(
        "--operations",
        default="io_scan,filter_agg,window_hsm,join",
        help="Comma-separated operations to run",
    )
    args = parser.parse_args()

    try:
        import cudf  # noqa: F401
    except ImportError:
        print("ERROR: cuDF not installed. Install via RAPIDS conda:")
        print("  conda create -n rapids -c rapidsai -c nvidia cudf=25.02 cuda-toolkit=12.0")
        sys.exit(1)

    ops = [o.strip() for o in args.operations.split(",")]
    data_path = args.data_path

    print(f"\ncuDF Standalone Diagnostic — {data_path}")
    print("=" * 60)

    gpu_mb_start = get_gpu_memory_used_mb()
    print(f"GPU memory at start: {gpu_mb_start:.0f} MB")

    op_map = {
        "io_scan": run_io_scan,
        "filter_agg": run_filter_agg,
        "window_hsm": run_window_hsm,
        "join": run_join,
    }

    for op in ops:
        if op not in op_map:
            print(f"  Unknown operation: {op}")
            continue
        try:
            result = op_map[op](data_path)
            n = result.get("total_records", 0)
            compute_t = result.get("compute_time_s", 0.0)
            throughput = n / max(compute_t, 1e-9)
            result_rows = result.get("result_rows", n)
            gpu_mb = result.get("gpu_memory_mb", 0.0)
            print(
                f"  {op:<15} {n:>10,} rows  "
                f"{compute_t:.3f}s  {throughput:>12,.0f} rec/s  "
                f"GPU: {gpu_mb:.0f} MB  out: {result_rows:,}"
            )
        except Exception as e:
            print(f"  {op:<15} ERROR: {e}")

    print()


if __name__ == "__main__":
    main()
