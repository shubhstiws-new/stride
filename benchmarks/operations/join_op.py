"""
Operation 4: Join — Self-join on task_id, count pairs.

df a JOIN df b ON a.task_id = b.task_id AND a.episode_id != b.episode_id
→ COUNT pairs

Simulates cross-robot cross-dataset analysis, measures shuffle cost.
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

from benchmarks.operations import (
    MatrixResult, make_error_result, glob_pattern, get_data_info, build_spark_session,
)

# Column name candidates in priority order (unified schema first, demo fallbacks second)
_JOIN_COL_CANDIDATES = ["task_id", "_task_id", "task_index", "robot_id"]
_PAIR_COL_CANDIDATES = ["episode_id", "robot_id", "_connection_id", "signal_type"]


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
    join_col = _resolve_col(df.columns, _JOIN_COL_CANDIDATES)
    pair_col = _resolve_col(df.columns, _PAIR_COL_CANDIDATES)
    pairs = df.select([join_col, pair_col]).unique()
    joined = pairs.join(pairs, on=join_col, how="inner", suffix="_b").filter(
        pl.col(pair_col) != pl.col(f"{pair_col}_b")
    )
    result_rows = len(joined)
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
    join_col = _resolve_col(cols, _JOIN_COL_CANDIDATES)
    pair_col = _resolve_col(cols, _PAIR_COL_CANDIDATES)

    t1 = time.perf_counter()
    result = con.execute(f"""
        WITH pairs AS (
            SELECT DISTINCT {join_col}, {pair_col}
            FROM read_parquet('{pattern}')
        )
        SELECT COUNT(*) AS pair_count
        FROM pairs a
        JOIN pairs b ON a.{join_col} = b.{join_col}
        WHERE a.{pair_col} != b.{pair_col}
    """).fetchone()
    result_rows = result[0]
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
    join_col = _resolve_col(list(df.columns), _JOIN_COL_CANDIDATES)
    pair_col = _resolve_col(list(df.columns), _PAIR_COL_CANDIDATES)
    pairs = df[[join_col, pair_col]].drop_duplicates()
    joined = pairs.merge(pairs, on=join_col, suffixes=("_a", "_b"))
    joined = joined[joined[f"{pair_col}_a"] != joined[f"{pair_col}_b"]]
    result_rows = len(joined)
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
    cols = list(ddf.columns)
    join_col = _resolve_col(cols, _JOIN_COL_CANDIDATES)
    pair_col = _resolve_col(cols, _PAIR_COL_CANDIDATES)
    df = ddf[[join_col, pair_col]].drop_duplicates().compute()
    n_approx = len(ddf)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    joined = df.merge(df, on=join_col, suffixes=("_a", "_b"))
    joined = joined[joined[f"{pair_col}_a"] != joined[f"{pair_col}_b"]]
    result_rows = len(joined)
    n = len(dd.read_parquet(pattern).compute())
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return n, load_time, compute_time, peak / 1024 / 1024, result_rows


def _run_pyspark(data_path: str, hardware_cfg: dict) -> tuple[int, float, float, float, int]:
    from pyspark.sql import functions as F

    spark, proc, mem_before = build_spark_session("matrix-join", data_path, hardware_cfg)

    t0 = time.perf_counter()
    df = spark.read.parquet(data_path)
    n = df.count()
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    join_col = _resolve_col(df.columns, _JOIN_COL_CANDIDATES)
    pair_col = _resolve_col(df.columns, _PAIR_COL_CANDIDATES)
    pairs = df.select(join_col, pair_col).distinct()
    a = pairs.alias("a")
    b = pairs.alias("b")
    joined = a.join(b, on=join_col).filter(F.col(f"a.{pair_col}") != F.col(f"b.{pair_col}"))
    result_rows = joined.count()
    compute_time = time.perf_counter() - t1

    spark.stop()
    mem_after = proc.memory_info().rss
    peak_mb = max(0.0, (mem_after - mem_before) / 1024 / 1024)
    return n, load_time, compute_time, peak_mb, result_rows


def _run_pyspark_rapids(data_path: str, hardware_cfg: dict) -> tuple[int, float, float, float, int]:
    from pyspark.sql import functions as F

    spark, proc, mem_before = build_spark_session(
        "matrix-join-rapids", data_path, hardware_cfg, rapids=True,
    )

    t0 = time.perf_counter()
    df = spark.read.parquet(data_path)
    n = df.count()
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    join_col = _resolve_col(df.columns, _JOIN_COL_CANDIDATES)
    pair_col = _resolve_col(df.columns, _PAIR_COL_CANDIDATES)
    pairs = df.select(join_col, pair_col).distinct()
    a = pairs.alias("a")
    b = pairs.alias("b")
    joined = a.join(b, on=join_col).filter(F.col(f"a.{pair_col}") != F.col(f"b.{pair_col}"))
    result_rows = joined.count()
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
    join_col = _resolve_col(list(df.columns), _JOIN_COL_CANDIDATES)
    pair_col = _resolve_col(list(df.columns), _PAIR_COL_CANDIDATES)
    pairs = df[[join_col, pair_col]].drop_duplicates()
    joined = pairs.merge(pairs, on=join_col, suffixes=("_a", "_b"))
    joined = joined[joined[f"{pair_col}_a"] != joined[f"{pair_col}_b"]]
    result_rows = len(joined)
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
    """Run self-join benchmark for the given framework."""
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
                hardware, framework, dataset, filesize_target, "join", status="skip"
            )

        total_time = time.perf_counter() - t_wall
        throughput = n / max(compute_t, 1e-9)

        return MatrixResult(
            hardware=hardware,
            framework=framework,
            dataset=dataset,
            filesize_target=filesize_target,
            operation="join",
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
            hardware, framework, dataset, filesize_target, "join",
            status="oom", num_files=num_files, actual_file_size_mb=actual_file_size_mb,
        )
    except Exception as e:
        oom_keywords = ("OutOfMemoryError", "OutOfMemoryException", "MemoryError", "Out of Memory")
        status = "oom" if any(kw in str(e) for kw in oom_keywords) else "err"
        return make_error_result(
            hardware, framework, dataset, filesize_target, "join",
            status=status, num_files=num_files, actual_file_size_mb=actual_file_size_mb,
        )
