"""
Operations layer for the benchmark matrix.

Each operation module exposes: run(data_path, framework, hardware_cfg) -> MatrixResult
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3/MinIO URI."""
    return path.startswith("s3://") or path.startswith("s3a://")


def glob_pattern(data_path: str) -> str:
    """Return parquet glob pattern that works for both S3 and local paths."""
    if is_s3_path(data_path):
        return data_path.rstrip("/") + "/*.parquet"
    return str(Path(data_path) / "*.parquet")


def get_data_info(data_path: str) -> tuple[int, float]:
    """Return (num_files, total_size_mb) — works for local paths, returns (0, 0) for S3."""
    if is_s3_path(data_path):
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
            files = [o for o in resp.get("Contents", []) if o["Key"].endswith(".parquet")]
            total_bytes = sum(o["Size"] for o in files)
            return len(files), total_bytes / 1024 / 1024
        except Exception:
            return 0, 0.0
    files = list(Path(data_path).glob("*.parquet"))
    return len(files), sum(f.stat().st_size for f in files) / 1024 / 1024


def build_spark_session(
    app_name: str,
    data_path: str,
    hardware_cfg: dict,
    rapids: bool = False,
):
    """
    Build a SparkSession with S3A and k8s config when needed.

    Returns (spark, proc, mem_before) tuple for memory tracking.
    """
    import psutil
    from pyspark.sql import SparkSession

    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    spark_cfg = hardware_cfg.get("spark_config", {})
    master = spark_cfg.get("master", "local[*]")
    is_k8s = master.startswith("k8s://")

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", spark_cfg.get("driver_memory", "8g"))
        .config("spark.sql.shuffle.partitions", str(spark_cfg.get("shuffle_partitions", 8)))
        .config("spark.ui.showConsoleProgress", "false")
    )

    # RAPIDS GPU plugin
    if rapids:
        rapids_jar = hardware_cfg.get("rapids_jar", "")
        if not rapids_jar:
            raise RuntimeError("rapids_jar path not set in hardware_cfg")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
        builder = (
            builder
            .config("spark.driver.memory", spark_cfg.get("driver_memory", "16g"))
            .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
            .config("spark.rapids.sql.enabled", "true")
            .config("spark.rapids.memory.gpu.minAllocFraction", "0")
            .config("spark.rapids.memory.gpu.allocFraction", "0.7")
            .config("spark.rapids.memory.gpu.maxAllocFraction", "0.8")
            .config("spark.rapids.sql.concurrentGpuTasks", "2")
            .config("spark.executor.resource.gpu.amount", "1")
            .config("spark.task.resource.gpu.amount", "1")
            .config("spark.jars", rapids_jar)
        )

    # k8s executor config
    if is_k8s:
        executor_image = spark_cfg.get("executor_image", "")
        k8s_namespace = spark_cfg.get("kubernetes_namespace", "hsm-bench")
        k8s_sa = spark_cfg.get("kubernetes_service_account", "spark")
        builder = (
            builder
            .config("spark.executor.instances", str(spark_cfg.get("executor_instances", 4)))
            .config("spark.executor.cores", str(spark_cfg.get("executor_cores", 1)))
            .config("spark.executor.memory", spark_cfg.get("executor_memory", "1g"))
            .config("spark.kubernetes.namespace", k8s_namespace)
            .config("spark.kubernetes.authenticate.serviceAccountName", k8s_sa)
            .config("spark.kubernetes.authenticate.driver.serviceAccountName", k8s_sa)
        )
        if executor_image:
            builder = builder.config("spark.kubernetes.container.image", executor_image)

    # S3A / MinIO config
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
    return spark, proc, mem_before


@dataclass
class MatrixResult:
    """Standardised result for a single benchmark matrix cell."""

    hardware: str           # "mac-cpu" | "linux-cpu" | "linux-gpu"
    framework: str          # "pyspark" | "polars" | "pyspark-rapids" | "cudf" | etc.
    dataset: str            # "behavior1k" | "droid" | "oxe" | "nvidia_physicalai"
    filesize_target: str    # "1mb" | "10mb" | "100mb" | "1gb"
    operation: str          # "io_scan" | "filter_agg" | "window_hsm" | "join"
    total_records: int
    num_files: int
    actual_file_size_mb: float
    load_time_s: float
    compute_time_s: float
    total_time_s: float
    throughput_records_per_sec: float
    peak_memory_mb: float
    gpu_memory_mb: float    # 0.0 if not GPU
    result_rows: int        # output row count (sessions, groups, etc.)
    status: str             # "ok" | "oom" | "err" | "skip"

    def to_dict(self) -> dict:
        return asdict(self)


def make_error_result(
    hardware: str,
    framework: str,
    dataset: str,
    filesize_target: str,
    operation: str,
    status: str = "err",
    num_files: int = 0,
    actual_file_size_mb: float = 0.0,
) -> MatrixResult:
    """Return a MatrixResult with zeroed metrics and given status."""
    return MatrixResult(
        hardware=hardware,
        framework=framework,
        dataset=dataset,
        filesize_target=filesize_target,
        operation=operation,
        total_records=0,
        num_files=num_files,
        actual_file_size_mb=actual_file_size_mb,
        load_time_s=0.0,
        compute_time_s=0.0,
        total_time_s=0.0,
        throughput_records_per_sec=0.0,
        peak_memory_mb=0.0,
        gpu_memory_mb=0.0,
        result_rows=0,
        status=status,
    )
