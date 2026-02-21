"""
Operation 2: Filter + Aggregation.

WHERE task_id IN ['0','1','2','3','4']
→ GROUP BY episode_id
→ AVG(signal_value), STDDEV(signal_value), COUNT(*)

Measures typical ETL aggregation performance.
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

from benchmarks.operations import MatrixResult, make_error_result

_TASK_FILTER = ["0", "1", "2", "3", "4"]

# Column name candidates in priority order (unified schema first, demo fallbacks second)
_TASK_COL_CANDIDATES = ["task_id", "_task_id", "task_index", "robot_id"]
_GROUP_COL_CANDIDATES = ["episode_id", "robot_id", "_connection_id", "signal_type"]


def _resolve_col(columns: list[str], candidates: list[str]) -> str:
    """Return the first candidate column that exists in columns."""
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {columns}")


def _get_data_info(data_path: str) -> tuple[int, float]:
    files = list(Path(data_path).glob("*.parquet"))
    return len(files), sum(f.stat().st_size for f in files) / 1024 / 1024


def _run_polars(data_path: str) -> tuple[int, float, float, float, int]:
    import polars as pl

    tracemalloc.start()
    t0 = time.perf_counter()
    df = pl.read_parquet(str(Path(data_path) / "*.parquet"))
    n = len(df)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    task_col = _resolve_col(df.columns, _TASK_COL_CANDIDATES)
    group_col = _resolve_col(df.columns, _GROUP_COL_CANDIDATES)
    result = (
        df.filter(pl.col(task_col).cast(pl.Utf8).is_in(_TASK_FILTER))
        .group_by(group_col)
        .agg(
            pl.col("signal_value").mean().alias("avg_value"),
            pl.col("signal_value").std().alias("std_value"),
            pl.len().alias("count"),
        )
    )
    result_rows = len(result)
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return n, load_time, compute_time, peak / 1024 / 1024, result_rows


def _run_duckdb(data_path: str) -> tuple[int, float, float, float, int]:
    import duckdb

    tracemalloc.start()
    con = duckdb.connect(":memory:")
    pattern = str(Path(data_path) / "*.parquet")

    t0 = time.perf_counter()
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pattern}')").fetchone()[0]
    load_time = time.perf_counter() - t0

    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pattern}') LIMIT 0").fetchall()]
    task_col = _resolve_col(cols, _TASK_COL_CANDIDATES)
    group_col = _resolve_col(cols, _GROUP_COL_CANDIDATES)
    task_list = ", ".join(f"'{t}'" for t in _TASK_FILTER)
    t1 = time.perf_counter()
    result = con.execute(f"""
        SELECT {group_col},
               AVG(signal_value)    AS avg_value,
               STDDEV(signal_value) AS std_value,
               COUNT(*)             AS cnt
        FROM read_parquet('{pattern}')
        WHERE CAST({task_col} AS VARCHAR) IN ({task_list})
        GROUP BY {group_col}
    """).df()
    result_rows = len(result)
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    con.close()
    return n, load_time, compute_time, peak / 1024 / 1024, result_rows


def _run_pandas(data_path: str) -> tuple[int, float, float, float, int]:
    import pandas as pd

    tracemalloc.start()
    files = sorted(Path(data_path).glob("*.parquet"))

    t0 = time.perf_counter()
    df = pd.concat([pd.read_parquet(str(f)) for f in files], ignore_index=True)
    n = len(df)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    task_col = _resolve_col(list(df.columns), _TASK_COL_CANDIDATES)
    group_col = _resolve_col(list(df.columns), _GROUP_COL_CANDIDATES)
    filtered = df[df[task_col].astype(str).isin(_TASK_FILTER)]
    result = filtered.groupby(group_col)["signal_value"].agg(["mean", "std", "count"])
    result_rows = len(result)
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return n, load_time, compute_time, peak / 1024 / 1024, result_rows


def _run_dask(data_path: str) -> tuple[int, float, float, float, int]:
    import dask.dataframe as dd

    tracemalloc.start()
    pattern = str(Path(data_path) / "*.parquet")

    t0 = time.perf_counter()
    ddf = dd.read_parquet(pattern)
    n = len(ddf)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    task_col = _resolve_col(list(ddf.columns), _TASK_COL_CANDIDATES)
    group_col = _resolve_col(list(ddf.columns), _GROUP_COL_CANDIDATES)
    filtered = ddf[ddf[task_col].astype(str).isin(_TASK_FILTER)]
    result = (
        filtered.groupby(group_col)["signal_value"]
        .agg(["mean", "std", "count"])
        .compute()
    )
    result_rows = len(result)
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return n, load_time, compute_time, peak / 1024 / 1024, result_rows


def _run_pyspark(data_path: str, hardware_cfg: dict) -> tuple[int, float, float, float, int]:
    import psutil
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    spark_cfg = hardware_cfg.get("spark_config", {})

    spark = (
        SparkSession.builder
        .appName("matrix-filter-agg")
        .master(spark_cfg.get("master", "local[*]"))
        .config("spark.driver.memory", spark_cfg.get("driver_memory", "8g"))
        .config("spark.sql.shuffle.partitions", str(spark_cfg.get("shuffle_partitions", 8)))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    t0 = time.perf_counter()
    df = spark.read.parquet(data_path)
    n = df.count()
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    task_col = _resolve_col(df.columns, _TASK_COL_CANDIDATES)
    group_col = _resolve_col(df.columns, _GROUP_COL_CANDIDATES)
    result = (
        df.filter(F.col(task_col).cast("string").isin(_TASK_FILTER))
        .groupBy(group_col)
        .agg(
            F.avg("signal_value").alias("avg_value"),
            F.stddev("signal_value").alias("std_value"),
            F.count("*").alias("cnt"),
        )
    )
    result_rows = result.count()
    compute_time = time.perf_counter() - t1

    spark.stop()
    mem_after = proc.memory_info().rss
    peak_mb = max(0.0, (mem_after - mem_before) / 1024 / 1024)
    return n, load_time, compute_time, peak_mb, result_rows


def _run_pyspark_rapids(data_path: str, hardware_cfg: dict) -> tuple[int, float, float, float, int]:
    import psutil
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

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
        .appName("matrix-filter-agg-rapids")
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
    n = df.count()
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    task_col = _resolve_col(df.columns, _TASK_COL_CANDIDATES)
    group_col = _resolve_col(df.columns, _GROUP_COL_CANDIDATES)
    result = (
        df.filter(F.col(task_col).cast("string").isin(_TASK_FILTER))
        .groupBy(group_col)
        .agg(
            F.avg("signal_value").alias("avg_value"),
            F.stddev("signal_value").alias("std_value"),
            F.count("*").alias("cnt"),
        )
    )
    result_rows = result.count()
    compute_time = time.perf_counter() - t1

    spark.stop()
    mem_after = proc.memory_info().rss
    peak_mb = max(0.0, (mem_after - mem_before) / 1024 / 1024)
    return n, load_time, compute_time, peak_mb, result_rows


def _run_cudf(data_path: str) -> tuple[int, float, float, float, int]:
    import cudf

    t0 = time.perf_counter()
    df = cudf.read_parquet(str(Path(data_path) / "*.parquet"))
    n = len(df)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    task_col = _resolve_col(list(df.columns), _TASK_COL_CANDIDATES)
    group_col = _resolve_col(list(df.columns), _GROUP_COL_CANDIDATES)
    filtered = df[df[task_col].astype(str).isin(_TASK_FILTER)]
    result = (
        filtered.groupby(group_col)["signal_value"]
        .agg(["mean", "std", "count"])
    )
    result_rows = len(result)
    compute_time = time.perf_counter() - t1

    gpu_mb = 0.0
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_mb = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024 / 1024
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return n, load_time, compute_time, gpu_mb, result_rows


def run(
    data_path: str,
    framework: str,
    hardware_cfg: dict,
    dataset: str,
    filesize_target: str,
    hardware: str,
) -> MatrixResult:
    """Run filter + aggregation benchmark for the given framework."""
    num_files, actual_file_size_mb = _get_data_info(data_path)
    gpu_memory_mb = 0.0
    t_wall = time.perf_counter()

    try:
        if framework == "polars":
            n, load_t, compute_t, peak_mb, result_rows = _run_polars(data_path)
        elif framework == "duckdb":
            n, load_t, compute_t, peak_mb, result_rows = _run_duckdb(data_path)
        elif framework == "pandas":
            n, load_t, compute_t, peak_mb, result_rows = _run_pandas(data_path)
        elif framework == "dask":
            n, load_t, compute_t, peak_mb, result_rows = _run_dask(data_path)
        elif framework == "pyspark":
            n, load_t, compute_t, peak_mb, result_rows = _run_pyspark(data_path, hardware_cfg)
        elif framework == "pyspark-rapids":
            n, load_t, compute_t, peak_mb, result_rows = _run_pyspark_rapids(data_path, hardware_cfg)
        elif framework == "cudf":
            n, load_t, compute_t, gpu_mb, result_rows = _run_cudf(data_path)
            gpu_memory_mb = gpu_mb
            peak_mb = 0.0
        else:
            return make_error_result(
                hardware, framework, dataset, filesize_target, "filter_agg", status="skip"
            )

        total_time = time.perf_counter() - t_wall
        throughput = n / max(compute_t, 1e-9)

        return MatrixResult(
            hardware=hardware,
            framework=framework,
            dataset=dataset,
            filesize_target=filesize_target,
            operation="filter_agg",
            total_records=n,
            num_files=num_files,
            actual_file_size_mb=actual_file_size_mb,
            load_time_s=load_t,
            compute_time_s=compute_t,
            total_time_s=total_time,
            throughput_records_per_sec=throughput,
            peak_memory_mb=peak_mb,
            gpu_memory_mb=gpu_memory_mb,
            result_rows=result_rows,
            status="ok",
        )

    except MemoryError:
        return make_error_result(
            hardware, framework, dataset, filesize_target, "filter_agg",
            status="oom", num_files=num_files, actual_file_size_mb=actual_file_size_mb,
        )
    except Exception as e:
        oom_keywords = ("OutOfMemoryError", "OutOfMemoryException", "MemoryError", "Out of Memory")
        status = "oom" if any(kw in str(e) for kw in oom_keywords) else "err"
        return make_error_result(
            hardware, framework, dataset, filesize_target, "filter_agg",
            status=status, num_files=num_files, actual_file_size_mb=actual_file_size_mb,
        )
