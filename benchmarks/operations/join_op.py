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

from benchmarks.operations import MatrixResult, make_error_result


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
    # Deduplicate to (task_id, episode_id) pairs to keep join tractable
    pairs = df.select(["task_id", "episode_id"]).unique()
    joined = pairs.join(pairs, on="task_id", how="inner", suffix="_b").filter(
        pl.col("episode_id") != pl.col("episode_id_b")
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
    pattern = str(Path(data_path) / "*.parquet")

    t0 = time.perf_counter()
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pattern}')").fetchone()[0]
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = con.execute(f"""
        WITH pairs AS (
            SELECT DISTINCT task_id, episode_id
            FROM read_parquet('{pattern}')
        )
        SELECT COUNT(*) AS pair_count
        FROM pairs a
        JOIN pairs b ON a.task_id = b.task_id
        WHERE a.episode_id != b.episode_id
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
    files = sorted(Path(data_path).glob("*.parquet"))

    t0 = time.perf_counter()
    df = pd.concat([pd.read_parquet(str(f)) for f in files], ignore_index=True)
    n = len(df)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    pairs = df[["task_id", "episode_id"]].drop_duplicates()
    joined = pairs.merge(pairs, on="task_id", suffixes=("_a", "_b"))
    joined = joined[joined["episode_id_a"] != joined["episode_id_b"]]
    result_rows = len(joined)
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
    df = ddf[["task_id", "episode_id"]].drop_duplicates().compute()
    n_approx = len(ddf)  # note: may trigger extra compute; use len of full df
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    import pandas as pd
    joined = df.merge(df, on="task_id", suffixes=("_a", "_b"))
    joined = joined[joined["episode_id_a"] != joined["episode_id_b"]]
    result_rows = len(joined)
    # Reload to get true n
    n = len(dd.read_parquet(pattern).compute())
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
        .appName("matrix-join")
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
    pairs = df.select("task_id", "episode_id").distinct()
    a = pairs.alias("a")
    b = pairs.alias("b")
    joined = a.join(b, on="task_id").filter(F.col("a.episode_id") != F.col("b.episode_id"))
    result_rows = joined.count()
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

    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    spark_cfg = hardware_cfg.get("spark_config", {})

    spark = (
        SparkSession.builder
        .appName("matrix-join-rapids")
        .master(spark_cfg.get("master", "local[*]"))
        .config("spark.driver.memory", spark_cfg.get("driver_memory", "16g"))
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .config("spark.rapids.sql.enabled", "true")
        .config("spark.executor.resource.gpu.amount", "1")
        .config("spark.task.resource.gpu.amount", "1")
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
    pairs = df.select("task_id", "episode_id").distinct()
    a = pairs.alias("a")
    b = pairs.alias("b")
    joined = a.join(b, on="task_id").filter(F.col("a.episode_id") != F.col("b.episode_id"))
    result_rows = joined.count()
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
    pairs = df[["task_id", "episode_id"]].drop_duplicates()
    joined = pairs.merge(pairs, on="task_id", suffixes=("_a", "_b"))
    joined = joined[joined["episode_id_a"] != joined["episode_id_b"]]
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
