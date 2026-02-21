"""
Operation 3: Window / HSM — Gap-based session detection.

Uses 5-second gap threshold to detect sessions per episode_id.
Reuses session_detector.py for PySpark path directly.
Supports both local filesystem paths and S3/MinIO URIs.
"""

import os
import time
import tracemalloc
from pathlib import Path

from benchmarks.operations import MatrixResult, make_error_result

_GAP_THRESHOLD_S = 5.0
_US_PER_S = 1_000_000


# ── S3 / local path helpers ───────────────────────────────────────────────────

def _is_s3_path(path: str) -> bool:
    return path.startswith("s3://") or path.startswith("s3a://")


def _glob_pattern(data_path: str) -> str:
    """Return parquet glob pattern for local or S3 paths."""
    if _is_s3_path(data_path):
        return data_path.rstrip("/") + "/*.parquet"
    return str(Path(data_path) / "*.parquet")


def _get_data_info(data_path: str) -> tuple[int, float]:
    if _is_s3_path(data_path):
        try:
            import boto3
            from urllib.parse import urlparse
            parsed = urlparse(data_path)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")
            endpoint = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:9000")
            s3 = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
            )
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            objects = [o for o in resp.get("Contents", []) if o["Key"].endswith(".parquet")]
            return len(objects), sum(o["Size"] for o in objects) / 1024 / 1024
        except Exception:
            return 0, 0.0
    files = list(Path(data_path).glob("*.parquet"))
    return len(files), sum(f.stat().st_size for f in files) / 1024 / 1024


def _configure_duckdb_minio(con) -> None:
    """Configure DuckDB httpfs for MinIO-compatible S3 endpoint."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:9000")
    host = endpoint.replace("https://", "").replace("http://", "")
    use_ssl = endpoint.startswith("https://")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET s3_endpoint='{host}';")
    con.execute(f"SET s3_use_ssl={'true' if use_ssl else 'false'};")
    con.execute("SET s3_url_style='path';")
    con.execute(f"SET s3_access_key_id='{access_key}';")
    con.execute(f"SET s3_secret_access_key='{secret_key}';")
    con.execute("SET s3_region='us-east-1';")


def _s3_storage_opts() -> dict:
    """Build s3fs/fsspec storage options for pandas/dask."""
    return {
        "endpoint_url": os.environ.get("AWS_ENDPOINT_URL", "http://localhost:9000"),
        "key": os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        "secret": os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    }


# ── Framework runners ─────────────────────────────────────────────────────────

def _run_polars(data_path: str) -> tuple[int, float, float, float, int]:
    import polars as pl

    tracemalloc.start()
    t0 = time.perf_counter()
    # Polars reads s3:// natively via object_store; respects AWS_ENDPOINT_URL
    df = pl.read_parquet(_glob_pattern(data_path))
    n = len(df)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    df = df.sort(["episode_id", "timestamp"])
    df = (
        df.with_columns(
            pl.col("timestamp").diff().over("episode_id").alias("_gap_s")
        )
        .with_columns(
            (pl.col("_gap_s").is_null() | (pl.col("_gap_s") > _GAP_THRESHOLD_S))
            .cast(pl.UInt32)
            .alias("_is_new")
        )
        .with_columns(
            pl.col("_is_new").cum_sum().over("episode_id").alias("_sess_num")
        )
        .with_columns(
            (pl.col("episode_id") + "_sess_" + pl.col("_sess_num").cast(pl.Utf8))
            .alias("session_id")
        )
        .drop(["_gap_s", "_is_new", "_sess_num"])
    )

    summary = (
        df.group_by("session_id")
        .agg(
            pl.col("timestamp").min().alias("session_start"),
            pl.col("timestamp").max().alias("session_end"),
            pl.len().alias("event_count"),
        )
    )
    result_rows = len(summary)
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return n, load_time, compute_time, peak / 1024 / 1024, result_rows


def _run_duckdb(data_path: str) -> tuple[int, float, float, float, int]:
    import duckdb

    tracemalloc.start()
    con = duckdb.connect(":memory:")
    pattern = _glob_pattern(data_path)

    if _is_s3_path(data_path):
        _configure_duckdb_minio(con)

    t0 = time.perf_counter()
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pattern}')").fetchone()[0]
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = con.execute(f"""
        WITH ordered AS (
            SELECT *,
                   timestamp - LAG(timestamp) OVER (
                       PARTITION BY episode_id ORDER BY timestamp
                   ) AS gap_s
            FROM read_parquet('{pattern}')
        ),
        flagged AS (
            SELECT *,
                   SUM(CASE WHEN gap_s IS NULL OR gap_s > {_GAP_THRESHOLD_S} THEN 1 ELSE 0 END)
                       OVER (PARTITION BY episode_id ORDER BY timestamp) AS sess_num
            FROM ordered
        )
        SELECT episode_id || '_sess_' || CAST(sess_num AS VARCHAR) AS session_id,
               MIN(timestamp) AS session_start,
               MAX(timestamp) AS session_end,
               COUNT(*)       AS event_count
        FROM flagged
        GROUP BY episode_id, sess_num
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

    if _is_s3_path(data_path):
        import s3fs
        opts = _s3_storage_opts()
        fs = s3fs.S3FileSystem(
            endpoint_url=opts["endpoint_url"],
            key=opts["key"],
            secret=opts["secret"],
        )
        # fs.glob returns paths without s3:// prefix
        s3_glob = _glob_pattern(data_path).replace("s3://", "")
        raw_paths = fs.glob(s3_glob)
        file_paths = ["s3://" + p for p in raw_paths]

        t0 = time.perf_counter()
        df = pd.concat(
            [pd.read_parquet(f, storage_options=opts) for f in file_paths],
            ignore_index=True,
        )
    else:
        files = sorted(Path(data_path).glob("*.parquet"))
        t0 = time.perf_counter()
        df = pd.concat([pd.read_parquet(str(f)) for f in files], ignore_index=True)

    n = len(df)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    df = df.sort_values(["episode_id", "timestamp"])
    df["_gap"] = df.groupby("episode_id")["timestamp"].diff()
    df["_is_new"] = (df["_gap"].isna() | (df["_gap"] > _GAP_THRESHOLD_S)).astype(int)
    df["_sess_num"] = df.groupby("episode_id")["_is_new"].cumsum()
    df["session_id"] = df["episode_id"] + "_sess_" + df["_sess_num"].astype(str)

    summary = (
        df.groupby("session_id")["timestamp"]
        .agg(session_start="min", session_end="max", event_count="count")
    )
    result_rows = len(summary)
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return n, load_time, compute_time, peak / 1024 / 1024, result_rows


def _run_dask(data_path: str) -> tuple[int, float, float, float, int]:
    import dask.dataframe as dd

    tracemalloc.start()
    pattern = _glob_pattern(data_path)

    t0 = time.perf_counter()
    if _is_s3_path(data_path):
        opts = _s3_storage_opts()
        ddf = dd.read_parquet(pattern, storage_options=opts)
    else:
        ddf = dd.read_parquet(pattern)

    # Dask window functions require sorted partitions; materialize for HSM
    df = ddf.compute()
    n = len(df)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    df = df.sort_values(["episode_id", "timestamp"])
    df["_gap"] = df.groupby("episode_id")["timestamp"].diff()
    df["_is_new"] = (df["_gap"].isna() | (df["_gap"] > _GAP_THRESHOLD_S)).astype(int)
    df["_sess_num"] = df.groupby("episode_id")["_is_new"].cumsum()
    df["session_id"] = df["episode_id"] + "_sess_" + df["_sess_num"].astype(str)

    summary = df.groupby("session_id")["timestamp"].agg(["min", "max", "count"])
    result_rows = len(summary)
    compute_time = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return n, load_time, compute_time, peak / 1024 / 1024, result_rows


def _run_pyspark(data_path: str, hardware_cfg: dict) -> tuple[int, float, float, float, int]:
    """PySpark path — reuses session_detector.py."""
    import psutil
    from pyspark.sql import SparkSession
    import sys

    repo_root = str(Path(__file__).parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from hsm_pyspark.spark.session_detector import SessionDetector, SessionConfig

    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    spark_cfg = hardware_cfg.get("spark_config", {})

    builder = (
        SparkSession.builder
        .appName("matrix-window-hsm")
        .master(spark_cfg.get("master", "local[*]"))
        .config("spark.driver.memory", spark_cfg.get("driver_memory", "8g"))
        .config("spark.sql.shuffle.partitions", str(spark_cfg.get("shuffle_partitions", 8)))
        .config("spark.ui.showConsoleProgress", "false")
    )

    # S3A / MinIO config for Spark
    if data_path.startswith("s3a://"):
        endpoint = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:9000")
        access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
        builder = (
            builder
            .config("spark.hadoop.fs.s3a.endpoint", endpoint)
            .config("spark.hadoop.fs.s3a.access.key", access_key)
            .config("spark.hadoop.fs.s3a.secret.key", secret_key)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
            )
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    t0 = time.perf_counter()
    df = spark.read.parquet(data_path)
    n = df.count()
    load_time = time.perf_counter() - t0

    config = SessionConfig(
        entity_key="episode_id",
        timestamp_col="timestamp",
        signal_col="dim_name",
        value_col="signal_value",
    )
    detector = SessionDetector(config)

    t1 = time.perf_counter()
    df_sessions = detector.detect_gap_sessions(df, gap_threshold_seconds=_GAP_THRESHOLD_S)
    summary = detector.summarize_sessions(df_sessions, "activity_session_id")
    result_rows = summary.count()
    compute_time = time.perf_counter() - t1

    spark.stop()
    mem_after = proc.memory_info().rss
    peak_mb = max(0.0, (mem_after - mem_before) / 1024 / 1024)
    return n, load_time, compute_time, peak_mb, result_rows


def _run_pyspark_rapids(data_path: str, hardware_cfg: dict) -> tuple[int, float, float, float, int]:
    import psutil
    from pyspark.sql import SparkSession
    import sys

    repo_root = str(Path(__file__).parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from hsm_pyspark.spark.session_detector import SessionDetector, SessionConfig

    rapids_jar = hardware_cfg.get("rapids_jar", "")
    if not rapids_jar:
        raise RuntimeError("rapids_jar path not set in hardware_cfg")

    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    spark_cfg = hardware_cfg.get("spark_config", {})

    spark = (
        SparkSession.builder
        .appName("matrix-window-hsm-rapids")
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

    config = SessionConfig(
        entity_key="episode_id",
        timestamp_col="timestamp",
        signal_col="dim_name",
        value_col="signal_value",
    )
    detector = SessionDetector(config)

    t1 = time.perf_counter()
    df_sessions = detector.detect_gap_sessions(df, gap_threshold_seconds=_GAP_THRESHOLD_S)
    summary = detector.summarize_sessions(df_sessions, "activity_session_id")
    result_rows = summary.count()
    compute_time = time.perf_counter() - t1

    spark.stop()
    mem_after = proc.memory_info().rss
    peak_mb = max(0.0, (mem_after - mem_before) / 1024 / 1024)
    return n, load_time, compute_time, peak_mb, result_rows


def _run_cudf(data_path: str) -> tuple[int, float, float, float, int]:
    import cudf

    t0 = time.perf_counter()
    df = cudf.read_parquet(_glob_pattern(data_path))
    n = len(df)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    df = df.sort_values(["episode_id", "timestamp"])
    df["_gap"] = df.groupby("episode_id")["timestamp"].diff()
    df["_is_new"] = ((df["_gap"].isna()) | (df["_gap"] > _GAP_THRESHOLD_S)).astype("int32")
    df["_sess_num"] = df.groupby("episode_id")["_is_new"].cumsum()
    df["session_id"] = df["episode_id"] + "_sess_" + df["_sess_num"].astype(str)

    summary = (
        df.groupby("session_id")["timestamp"]
        .agg(["min", "max", "count"])
    )
    result_rows = len(summary)
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
    """Run window/HSM gap-detection benchmark for the given framework."""
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
                hardware, framework, dataset, filesize_target, "window_hsm", status="skip"
            )

        total_time = time.perf_counter() - t_wall
        throughput = n / max(compute_t, 1e-9)

        return MatrixResult(
            hardware=hardware,
            framework=framework,
            dataset=dataset,
            filesize_target=filesize_target,
            operation="window_hsm",
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
            hardware, framework, dataset, filesize_target, "window_hsm",
            status="oom", num_files=num_files, actual_file_size_mb=actual_file_size_mb,
        )
    except Exception as e:
        oom_keywords = ("OutOfMemoryError", "OutOfMemoryException", "MemoryError", "Out of Memory")
        status = "oom" if any(kw in str(e) for kw in oom_keywords) else "err"
        return make_error_result(
            hardware, framework, dataset, filesize_target, "window_hsm",
            status=status, num_files=num_files, actual_file_size_mb=actual_file_size_mb,
        )
