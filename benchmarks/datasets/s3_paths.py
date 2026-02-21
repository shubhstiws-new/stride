"""
S3 / MinIO path resolution for multi-framework benchmarks.

Translates a local data_path to an S3 URI when benchmarking in S3 / MinIO
storage mode, or returns the local path unchanged for local-mode benchmarks.

Framework S3 support:
  - pyspark / pyspark-rapids: native S3A via hadoop-aws (s3a:// URI)
  - polars:    native s3:// via polars[all] / object_store
  - duckdb:    native s3:// via httpfs extension
  - pandas:    s3:// transparently via s3fs
  - dask:      s3:// transparently via s3fs
  - cudf:      s3:// via cudf.read_parquet (uses s3fs under the hood)

Storage modes:
  "local"  — use local filesystem paths (no S3 env vars set)
  "s3"     — real AWS S3 (uses default AWS credential chain)
  "minio"  — local MinIO (sets AWS_* env vars to point at MinIO server)
"""

import os
from pathlib import Path

# All frameworks that can read s3:// / s3a:// natively (no local fallback needed)
S3_NATIVE_FRAMEWORKS = frozenset(
    {"pyspark", "pyspark-rapids", "polars", "duckdb", "pandas", "dask", "cudf"}
)

# Spark uses s3a://, everything else uses s3://
_SPARK_FRAMEWORKS = frozenset({"pyspark", "pyspark-rapids"})


def local_to_s3_uri(
    local_path: str,
    s3_bucket: str,
    s3_prefix: str = "benchmarks",
    spark: bool = False,
) -> str:
    """
    Convert a local data directory path to an S3 URI.

    Example:
        local_path = "/data/behavior1k/filesize=10mb"
        → "s3://robotics-bench/benchmarks/behavior1k/filesize=10mb"
        (or "s3a://..." for Spark frameworks)
    """
    p = Path(local_path)
    # Two-level relative key: {dataset}/{filesize=label}
    rel = f"{p.parent.name}/{p.name}"
    scheme = "s3a" if spark else "s3"
    return f"{scheme}://{s3_bucket}/{s3_prefix}/{rel}"


def configure_minio_env(
    endpoint: str = "http://localhost:9000",
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
) -> None:
    """
    Set AWS_* environment variables so boto3, s3fs, and DuckDB all point at MinIO.
    Call this once before any S3-reading framework is invoked.
    """
    os.environ["AWS_ACCESS_KEY_ID"] = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
    os.environ["AWS_ENDPOINT_URL"] = endpoint
    os.environ["AWS_ENDPOINT_URL_S3"] = endpoint
    # Disable SSL verification for plain-HTTP MinIO
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


def configure_spark_for_minio(spark, endpoint: str = "http://localhost:9000") -> None:
    """
    Configure an active SparkSession to read from MinIO via s3a://.

    Must be called after SparkSession is created but before any read.parquet("s3a://...").
    """
    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3a.endpoint", endpoint)
    hadoop_conf.set("fs.s3a.access.key", os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"))
    hadoop_conf.set("fs.s3a.secret.key", os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"))
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")


def resolve_data_path(
    local_path: str,
    framework: str,
    storage_mode: str = "local",
    s3_bucket: str = "robotics-bench",
    s3_prefix: str = "benchmarks",
    minio_endpoint: str = "http://localhost:9000",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin",
) -> str:
    """
    Return the data path appropriate for the given framework and storage mode.

    Args:
        local_path:    Absolute local filesystem path (always provided as baseline).
        framework:     One of the benchmark framework names.
        storage_mode:  "local" | "s3" | "minio"
        s3_bucket:     S3 / MinIO bucket name.
        s3_prefix:     Key prefix inside the bucket.
        minio_endpoint: MinIO HTTP endpoint (for storage_mode="minio").
        minio_access_key / minio_secret_key: MinIO credentials.

    Returns:
        Data path string — either local absolute path or s3[a]://... URI.
    """
    if storage_mode == "local":
        return local_path

    if framework not in S3_NATIVE_FRAMEWORKS:
        # Framework doesn't support S3; fall back to local
        return local_path

    if storage_mode == "minio":
        configure_minio_env(minio_endpoint, minio_access_key, minio_secret_key)

    use_s3a = framework in _SPARK_FRAMEWORKS
    return local_to_s3_uri(local_path, s3_bucket, s3_prefix, spark=use_s3a)
