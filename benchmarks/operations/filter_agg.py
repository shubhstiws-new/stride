"""
Operation 2: Filter + Aggregation.

WHERE task_id IN (first 5 distinct task_ids)
→ GROUP BY episode_id
→ AVG(signal_value), STDDEV(signal_value), COUNT(*)

Measures typical ETL aggregation performance.
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

from benchmarks.operations import (
    MatrixResult, make_error_result, glob_pattern, get_data_info, build_spark_session,
)

_NUM_TASK_FILTER = 5  # number of distinct task_id values to filter on

# Column name candidates in priority order (unified schema first, demo fallbacks second)
_TASK_COL_CANDIDATES = ["task_id", "_task_id", "task_index", "robot_id"]
_GROUP_COL_CANDIDATES = ["episode_id", "robot_id", "_connection_id", "signal_type"]


def _resolve_col(columns: list[str], candidates: list[str]) -> str:
    """Return the first candidate column that exists in columns."""
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {columns}")


def _run_polars(data_path: str) -> tuple[int, float, float, float, int]:
    import polars as pl

    tracemalloc.start()
    t0 = time.perf_counter()
    df = pl.read_parquet(glob_pattern(data_path))
    n = len(df)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    task_col = _resolve_col(df.columns, _TASK_COL_CANDIDATES)
    group_col = _resolve_col(df.columns, _GROUP_COL_CANDIDATES)
    task_filter = (
        df.select(pl.col(task_col).cast(pl.Utf8))
        .unique()
        .sort(task_col)
        .head(_NUM_TASK_FILTER)[task_col]
        .to_list()
    )
    result = (
        df.filter(pl.col(task_col).cast(pl.Utf8).is_in(task_filter))
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
    pattern = glob_pattern(data_path)

    t0 = time.perf_counter()
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pattern}')").fetchone()[0]
    load_time = time.perf_counter() - t0

    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pattern}') LIMIT 0").fetchall()]
    task_col = _resolve_col(cols, _TASK_COL_CANDIDATES)
    group_col = _resolve_col(cols, _GROUP_COL_CANDIDATES)
    task_vals = con.execute(f"""
        SELECT DISTINCT CAST({task_col} AS VARCHAR) AS tid
        FROM read_parquet('{pattern}')
        ORDER BY tid
        LIMIT {_NUM_TASK_FILTER}
    """).fetchall()
    task_list = ", ".join(f"'{r[0]}'" for r in task_vals)
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
    pattern = glob_pattern(data_path)

    t0 = time.perf_counter()
    df = pd.read_parquet(pattern)
    n = len(df)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    task_col = _resolve_col(list(df.columns), _TASK_COL_CANDIDATES)
    group_col = _resolve_col(list(df.columns), _GROUP_COL_CANDIDATES)
    task_filter = sorted(df[task_col].astype(str).unique())[:_NUM_TASK_FILTER]
    filtered = df[df[task_col].astype(str).isin(task_filter)]
    result = filtered.groupby(group_col)["signal_value"].agg(["mean", "std", "count"])
    result_rows = len(result)
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return n, load_time, compute_time, peak / 1024 / 1024, result_rows


def _run_dask(data_path: str) -> tuple[int, float, float, float, int]:
    import dask.dataframe as dd

    tracemalloc.start()
    pattern = glob_pattern(data_path)

    t0 = time.perf_counter()
    ddf = dd.read_parquet(pattern)
    n = len(ddf)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    task_col = _resolve_col(list(ddf.columns), _TASK_COL_CANDIDATES)
    group_col = _resolve_col(list(ddf.columns), _GROUP_COL_CANDIDATES)
    task_filter = sorted(ddf[task_col].astype(str).unique().compute().tolist())[:_NUM_TASK_FILTER]
    filtered = ddf[ddf[task_col].astype(str).isin(task_filter)]
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
    from pyspark.sql import functions as F

    spark, proc, mem_before = build_spark_session("matrix-filter-agg", data_path, hardware_cfg)

    t0 = time.perf_counter()
    df = spark.read.parquet(data_path)
    n = df.count()
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    task_col = _resolve_col(df.columns, _TASK_COL_CANDIDATES)
    group_col = _resolve_col(df.columns, _GROUP_COL_CANDIDATES)
    task_filter = [
        row[0] for row in
        df.select(F.col(task_col).cast("string")).distinct().orderBy(task_col).limit(_NUM_TASK_FILTER).collect()
    ]
    result = (
        df.filter(F.col(task_col).cast("string").isin(task_filter))
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
    from pyspark.sql import functions as F

    spark, proc, mem_before = build_spark_session(
        "matrix-filter-agg-rapids", data_path, hardware_cfg, rapids=True,
    )

    t0 = time.perf_counter()
    df = spark.read.parquet(data_path)
    n = df.count()
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    task_col = _resolve_col(df.columns, _TASK_COL_CANDIDATES)
    group_col = _resolve_col(df.columns, _GROUP_COL_CANDIDATES)
    task_filter = [
        row[0] for row in
        df.select(F.col(task_col).cast("string")).distinct().orderBy(task_col).limit(_NUM_TASK_FILTER).collect()
    ]
    result = (
        df.filter(F.col(task_col).cast("string").isin(task_filter))
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
    df = cudf.read_parquet(glob_pattern(data_path))
    n = len(df)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    task_col = _resolve_col(list(df.columns), _TASK_COL_CANDIDATES)
    group_col = _resolve_col(list(df.columns), _GROUP_COL_CANDIDATES)
    task_filter = sorted(df[task_col].astype(str).unique().to_pandas().tolist())[:_NUM_TASK_FILTER]
    filtered = df[df[task_col].astype(str).isin(task_filter)]
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
    num_files, actual_file_size_mb = get_data_info(data_path)
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
