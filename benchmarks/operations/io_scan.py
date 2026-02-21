"""
Operation 1: IO Scan — pure read + COUNT(*).

Measures Parquet decode and file-open overhead with no compute beyond counting.
Framework dispatch via `framework` argument.
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

from benchmarks.operations import (
    MatrixResult, make_error_result, glob_pattern, get_data_info, build_spark_session,
    is_s3_path, configure_duckdb_s3, read_parquet_pandas, read_parquet_dask,
)


def _run_polars(data_path: str) -> tuple[int, float, float, float]:
    """Returns (total_records, load_time_s, compute_time_s, peak_memory_mb)."""
    import polars as pl

    tracemalloc.start()
    t0 = time.perf_counter()
    df = pl.read_parquet(glob_pattern(data_path))
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    n = len(df)
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return n, load_time, compute_time, peak / 1024 / 1024


def _run_duckdb(data_path: str) -> tuple[int, float, float, float]:
    import duckdb

    tracemalloc.start()
    con = duckdb.connect(":memory:")
    if is_s3_path(data_path):
        configure_duckdb_s3(con)
    pattern = glob_pattern(data_path)

    t0 = time.perf_counter()
    result = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pattern}')").fetchone()
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    n = result[0]
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    con.close()
    return n, load_time, compute_time, peak / 1024 / 1024


def _run_pandas(data_path: str) -> tuple[int, float, float, float]:
    tracemalloc.start()

    t0 = time.perf_counter()
    df = read_parquet_pandas(data_path)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    n = len(df)
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return n, load_time, compute_time, peak / 1024 / 1024


def _run_dask(data_path: str) -> tuple[int, float, float, float]:
    tracemalloc.start()

    t0 = time.perf_counter()
    ddf = read_parquet_dask(data_path)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    n = len(ddf)  # triggers compute
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return n, load_time, compute_time, peak / 1024 / 1024


def _run_pyspark(data_path: str, hardware_cfg: dict) -> tuple[int, float, float, float]:
    spark, proc, mem_before = build_spark_session("matrix-io-scan", data_path, hardware_cfg)

    t0 = time.perf_counter()
    df = spark.read.parquet(data_path)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    n = df.count()
    compute_time = time.perf_counter() - t1

    spark.stop()
    mem_after = proc.memory_info().rss
    peak_mb = max(0.0, (mem_after - mem_before) / 1024 / 1024)
    return n, load_time, compute_time, peak_mb


def _run_pyspark_rapids(data_path: str, hardware_cfg: dict) -> tuple[int, float, float, float]:
    spark, proc, mem_before = build_spark_session(
        "matrix-io-scan-rapids", data_path, hardware_cfg, rapids=True,
    )

    t0 = time.perf_counter()
    df = spark.read.parquet(data_path)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    n = df.count()
    compute_time = time.perf_counter() - t1

    spark.stop()
    mem_after = proc.memory_info().rss
    peak_mb = max(0.0, (mem_after - mem_before) / 1024 / 1024)
    return n, load_time, compute_time, peak_mb


def _run_cudf(data_path: str) -> tuple[int, float, float, float]:
    import cudf

    t0 = time.perf_counter()
    df = cudf.read_parquet(glob_pattern(data_path))
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    n = len(df)
    compute_time = time.perf_counter() - t1

    # GPU memory via pynvml
    gpu_mb = 0.0
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        gpu_mb = info.used / 1024 / 1024
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return n, load_time, compute_time, gpu_mb


def run(
    data_path: str,
    framework: str,
    hardware_cfg: dict,
    dataset: str,
    filesize_target: str,
    hardware: str,
) -> MatrixResult:
    """Run IO scan benchmark for the given framework."""
    num_files, actual_file_size_mb = get_data_info(data_path)
    gpu_memory_mb = 0.0
    t_wall = time.perf_counter()

    try:
        if framework == "polars":
            n, load_t, compute_t, peak_mb = _run_polars(data_path)
        elif framework == "duckdb":
            n, load_t, compute_t, peak_mb = _run_duckdb(data_path)
        elif framework == "pandas":
            n, load_t, compute_t, peak_mb = _run_pandas(data_path)
        elif framework == "dask":
            n, load_t, compute_t, peak_mb = _run_dask(data_path)
        elif framework == "pyspark":
            n, load_t, compute_t, peak_mb = _run_pyspark(data_path, hardware_cfg)
        elif framework == "pyspark-rapids":
            n, load_t, compute_t, peak_mb = _run_pyspark_rapids(data_path, hardware_cfg)
        elif framework == "cudf":
            n, load_t, compute_t, peak_mb = _run_cudf(data_path)
            gpu_memory_mb = peak_mb
            peak_mb = 0.0
        else:
            return make_error_result(
                hardware, framework, dataset, filesize_target, "io_scan", status="skip"
            )

        total_time = time.perf_counter() - t_wall
        throughput = n / max(compute_t, 1e-9)

        return MatrixResult(
            hardware=hardware,
            framework=framework,
            dataset=dataset,
            filesize_target=filesize_target,
            operation="io_scan",
            total_records=n,
            num_files=num_files,
            actual_file_size_mb=actual_file_size_mb,
            load_time_s=load_t,
            compute_time_s=compute_t,
            total_time_s=total_time,
            throughput_records_per_sec=throughput,
            peak_memory_mb=peak_mb,
            gpu_memory_mb=gpu_memory_mb,
            result_rows=1,  # just the count
            status="ok",
        )

    except MemoryError:
        return make_error_result(
            hardware, framework, dataset, filesize_target, "io_scan",
            status="oom", num_files=num_files, actual_file_size_mb=actual_file_size_mb,
        )
    except Exception as e:
        oom_keywords = ("OutOfMemoryError", "OutOfMemoryException", "MemoryError", "Out of Memory")
        status = "oom" if any(kw in str(e) for kw in oom_keywords) else "err"
        return make_error_result(
            hardware, framework, dataset, filesize_target, "io_scan",
            status=status, num_files=num_files, actual_file_size_mb=actual_file_size_mb,
        )
