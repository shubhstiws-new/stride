"""
PySpark + RAPIDS GPU Plugin runner configuration.

Provides a helper to build a RAPIDS-enabled SparkSession and track GPU memory.
Used by individual operation modules when framework == "pyspark-rapids".

Standalone usage (diagnostic):
    python benchmarks/frameworks/rapids_spark.py --jar ./jars/rapids-4-spark_2.12-25.02.0.jar
"""

import argparse
import sys
import time
from pathlib import Path


def build_rapids_spark_session(
    app_name: str,
    rapids_jar: str,
    driver_memory: str = "16g",
    master: str = "local[*]",
    shuffle_partitions: int = 16,
    gpu_max_alloc_fraction: float = 0.8,
    concurrent_gpu_tasks: int = 2,
):
    """
    Build and return a RAPIDS GPU-enabled SparkSession.

    Args:
        app_name: Spark application name.
        rapids_jar: Absolute path to rapids-4-spark_*.jar
        driver_memory: Spark driver memory (e.g. "16g").
        master: Spark master URL (default: local[*]).
        shuffle_partitions: spark.sql.shuffle.partitions value.
        gpu_max_alloc_fraction: Fraction of GPU memory RAPIDS may use (0.0–1.0).
        concurrent_gpu_tasks: Number of concurrent GPU tasks per executor.

    Returns:
        Active SparkSession configured with RAPIDS plugin.

    Raises:
        FileNotFoundError: If rapids_jar does not exist.
        RuntimeError: If the RAPIDS plugin fails to load.
    """
    if not Path(rapids_jar).exists():
        raise FileNotFoundError(
            f"RAPIDS JAR not found: {rapids_jar}\n"
            "Run: bash setup/gpu_setup.sh  to download it."
        )

    import os
    # GPU 0 (3090 Ti) may be used by other processes (ComfyUI etc).
    # GPU 1 (RTX 3090, 24 GB) is dedicated for benchmarks.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        # RAPIDS plugin
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .config("spark.rapids.sql.enabled", "true")
        # GPU resource allocation
        .config("spark.executor.resource.gpu.amount", "1")
        .config("spark.task.resource.gpu.amount", "1")
        # Memory — target GPU 1 (RTX 3090, 24 GB free).
        # minAllocFraction=0 disables the lower-bound check.
        # allocFraction=0.7 ensures pool < maxAllocFraction ceiling.
        .config("spark.rapids.memory.gpu.minAllocFraction", "0")
        .config("spark.rapids.memory.gpu.allocFraction", "0.7")
        .config("spark.rapids.memory.gpu.maxAllocFraction", str(gpu_max_alloc_fraction))
        .config("spark.rapids.sql.concurrentGpuTasks", str(concurrent_gpu_tasks))
        # JAR
        .config("spark.jars", rapids_jar)
        # Suppress console spam
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def get_gpu_memory_mb(device_index: int = 0) -> float:
    """Return currently used GPU memory in MB for the specified device."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / 1024 / 1024
    except Exception:
        return 0.0


def get_gpu_memory_delta_mb(device_index: int = 0):
    """
    Context manager that captures GPU memory used before and after a block.

    Usage:
        with get_gpu_memory_delta_mb() as gpu_mem:
            run_gpu_work()
        print(f"GPU delta: {gpu_mem.delta_mb:.1f} MB")
    """

    class _GPUMemContext:
        def __init__(self, idx: int):
            self.idx = idx
            self.before_mb = 0.0
            self.after_mb = 0.0
            self.delta_mb = 0.0

        def __enter__(self):
            self.before_mb = get_gpu_memory_mb(self.idx)
            return self

        def __exit__(self, *_):
            self.after_mb = get_gpu_memory_mb(self.idx)
            self.delta_mb = max(0.0, self.after_mb - self.before_mb)

    return _GPUMemContext(device_index)


# ── Diagnostic CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAPIDS Spark session diagnostic — verifies plugin loads correctly"
    )
    parser.add_argument(
        "--jar",
        required=True,
        help="Path to rapids-4-spark_*.jar",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional: Parquet path to run a quick COUNT(*) via RAPIDS",
    )
    args = parser.parse_args()

    print(f"Building RAPIDS SparkSession with JAR: {args.jar}")
    t0 = time.perf_counter()
    spark = build_rapids_spark_session(
        app_name="RAPIDS-Diagnostic",
        rapids_jar=args.jar,
    )
    print(f"SparkSession started in {time.perf_counter() - t0:.1f}s")

    # Confirm RAPIDS plugin is active
    plugins = spark.conf.get("spark.plugins", "")
    rapids_enabled = spark.conf.get("spark.rapids.sql.enabled", "false")
    print(f"  spark.plugins          = {plugins}")
    print(f"  spark.rapids.sql.enabled = {rapids_enabled}")

    gpu_mb_before = get_gpu_memory_mb()
    print(f"  GPU memory before job  = {gpu_mb_before:.0f} MB")

    if args.data_path:
        print(f"\nRunning COUNT(*) on {args.data_path}...")
        t1 = time.perf_counter()
        df = spark.read.parquet(args.data_path)
        n = df.count()
        elapsed = time.perf_counter() - t1
        gpu_mb_after = get_gpu_memory_mb()
        print(f"  COUNT(*) = {n:,} rows in {elapsed:.2f}s")
        print(f"  GPU memory after job   = {gpu_mb_after:.0f} MB (delta {gpu_mb_after - gpu_mb_before:.0f} MB)")
    else:
        # Simple test query
        t1 = time.perf_counter()
        result = spark.range(1_000_000).selectExpr("SUM(id)").collect()
        elapsed = time.perf_counter() - t1
        print(f"\n  SUM(range(1M)) = {result[0][0]:,}  in {elapsed:.2f}s")

    spark.stop()
    print("\nRAP IDS diagnostic complete.")


if __name__ == "__main__":
    main()
