"""
Operation 1: IO Scan — pure read + COUNT(*).

Measures Parquet decode and file-open overhead with no compute beyond counting.
Framework dispatch via `framework` argument.
"""

from __future__ import annotations

import glob
import time
import tracemalloc
from pathlib import Path

from benchmarks.operations import MatrixResult, make_error_result


def _get_data_info(data_path: str) -> tuple[int, float]:
    """Return (num_files, total_size_mb)."""
    files = list(Path(data_path).glob("*.parquet"))
    n = len(files)
    total_bytes = sum(f.stat().st_size for f in files)
    return n, total_bytes / 1024 / 1024


def _run_polars(data_path: str) -> tuple[int, float, float, float]:
    """Returns (total_records, load_time_s, compute_time_s, peak_memory_mb)."""
    import polars as pl

    tracemalloc.start()
    t0 = time.perf_counter()
    df = pl.read_parquet(str(Path(data_path) / "*.parquet"))
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
    pattern = str(Path(data_path) / "*.parquet")

    t0 = time.perf_counter()
    # DuckDB reads lazily; force materialization via COUNT(*)
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
    import pandas as pd

    tracemalloc.start()
    files = sorted(Path(data_path).glob("*.parquet"))

    t0 = time.perf_counter()
    frames = [pd.read_parquet(str(f)) for f in files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    n = len(df)
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return n, load_time, compute_time, peak / 1024 / 1024


def _run_dask(data_path: str) -> tuple[int, float, float, float]:
    import dask.dataframe as dd

    tracemalloc.start()
    pattern = str(Path(data_path) / "*.parquet")

    t0 = time.perf_counter()
    ddf = dd.read_parquet(pattern)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    n = len(ddf)  # triggers compute
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return n, load_time, compute_time, peak / 1024 / 1024


def _run_pyspark(data_path: str, hardware_cfg: dict) -> tuple[int, float, float, float]:
    import psutil
    from pyspark.sql import SparkSession

    proc = psutil.Process()
    mem_before = proc.memory_info().rss

    spark_cfg = hardware_cfg.get("spark_config", {})
    builder = (
        SparkSession.builder
        .appName("matrix-io-scan")
        .master(spark_cfg.get("master", "local[*]"))
        .config("spark.driver.memory", spark_cfg.get("driver_memory", "8g"))
        .config("spark.sql.shuffle.partitions", str(spark_cfg.get("shuffle_partitions", 8)))
        .config("spark.ui.showConsoleProgress", "false")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

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
    """PySpark with RAPIDS GPU plugin."""
    import psutil
    from pyspark.sql import SparkSession

    rapids_jar = hardware_cfg.get("rapids_jar", "")
    if not rapids_jar:
        raise RuntimeError("rapids_jar path not set in hardware_cfg")

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

    proc = psutil.Process()
    mem_before = proc.memory_info().rss

    spark_cfg = hardware_cfg.get("spark_config", {})
    spark = (
        SparkSession.builder
        .appName("matrix-io-scan-rapids")
        .master(spark_cfg.get("master", "local[*]"))
        .config("spark.driver.memory", spark_cfg.get("driver_memory", "16g"))
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .config("spark.rapids.sql.enabled", "true")
        .config("spark.rapids.memory.gpu.minAllocFraction", "0")
        .config("spark.rapids.memory.gpu.allocFraction", "0.7")
        .config("spark.rapids.memory.gpu.maxAllocFraction", "0.8")
        .config("spark.rapids.sql.concurrentGpuTasks", "2")
        .config("spark.jars", rapids_jar)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

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
    df = cudf.read_parquet(str(Path(data_path) / "*.parquet"))
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
    num_files, actual_file_size_mb = _get_data_info(data_path)
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
