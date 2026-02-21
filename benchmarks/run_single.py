#!/usr/bin/env python3
"""
benchmarks/run_single.py — Pod entry point for Airflow KubernetesPodOperator tasks.

Each benchmark matrix cell (framework × dataset × filesize × operation) runs as a
separate k8s pod. This script:
  1. Resolves the S3/MinIO data path via s3_paths.py
  2. Dispatches to the correct operation module (operations/<op>.py)
  3. Writes the MatrixResult JSON to MinIO under:
       s3://robotics-bench/results/{run_id}/{cell_id}.json
  4. Exits 0 on success, 1 on any failure (Airflow marks the task failed)

Usage (inside pod, called by Dockerfile ENTRYPOINT):
    python3 benchmarks/run_single.py \\
        --framework pyspark \\
        --dataset behavior1k \\
        --filesize 10mb \\
        --operation window_hsm \\
        --run-id dag_run_2026_02_21 \\
        --minio-endpoint http://minio.hsm-bench.svc:9000 \\
        --spark-master k8s://https://kubernetes.default.svc:6443 \\
        --local-data-dir /data/behavior1k \\
        --hardware linux-k8s
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.datasets.s3_paths import configure_minio_env, local_to_s3_uri
from benchmarks.operations import MatrixResult


# ── Operations registry ───────────────────────────────────────────────────────
_OPERATIONS = ["io_scan", "filter_agg", "window_hsm", "join_op"]


def _load_operation(op_name: str):
    """Dynamically import and return the operation module."""
    import importlib
    module_name = op_name.rstrip("_op")  # normalise "join_op" → "join"
    # Try exact name first, then with _op suffix
    for name in [op_name, f"{op_name.rstrip('_op')}"]:
        try:
            return importlib.import_module(f"benchmarks.operations.{name}")
        except ModuleNotFoundError:
            continue
    raise ImportError(f"No operation module found for '{op_name}'")


# ── Hardware config builders ──────────────────────────────────────────────────

def _build_hardware_cfg(args) -> dict:
    """
    Build the hardware_cfg dict expected by each operation's run() function.
    Mirrors the structure used by matrix_benchmark.py but tuned for k8s pods.
    """
    spark_master = args.spark_master or "local[*]"
    is_k8s = spark_master.startswith("k8s://")

    spark_config: dict = {
        "master": spark_master,
        "driver_memory": args.driver_memory,
        "shuffle_partitions": args.shuffle_partitions,
    }

    if is_k8s:
        spark_config.update({
            "executor_instances": args.executor_instances,
            "executor_cores": 1,
            "executor_memory": "1g",
            "executor_image": args.executor_image or "",
            "kubernetes_namespace": args.k8s_namespace,
            "kubernetes_service_account": "spark",
        })

    cfg: dict = {
        "spark_config": spark_config,
    }

    # PySpark-RAPIDS: JAR mounted at /opt/rapids/ on GPU nodes
    if args.framework == "pyspark-rapids":
        rapids_jar = args.rapids_jar or "/opt/rapids/rapids-4-spark_2.12-25.02.0-cuda12.jar"
        cfg["rapids_jar"] = rapids_jar
        if is_k8s:
            spark_config.update({
                "executor_gpu": 1,
                "rapids_jar_local": rapids_jar,
            })

    return cfg


# ── S3 result writer ──────────────────────────────────────────────────────────

def _write_result_to_s3(result: MatrixResult, run_id: str, cell_id: str, endpoint: str) -> None:
    """Write result JSON to s3://robotics-bench/results/{run_id}/{cell_id}.json"""
    import boto3

    payload = json.dumps(result.to_dict(), indent=2)
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        region_name="us-east-1",
    )
    key = f"results/{run_id}/{cell_id}.json"
    s3.put_object(
        Bucket="robotics-bench",
        Key=key,
        Body=payload.encode("utf-8"),
        ContentType="application/json",
    )
    print(f"  Result written → s3://robotics-bench/{key}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a single benchmark matrix cell inside a k8s pod"
    )
    parser.add_argument("--framework", required=True,
                        choices=["pyspark", "pyspark-rapids", "polars", "duckdb",
                                 "pandas", "dask", "cudf"],
                        help="Benchmark framework")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g. behavior1k, lerobot, gr00t)")
    parser.add_argument("--filesize", required=True,
                        help="Filesize label (e.g. 1mb, 10mb)")
    parser.add_argument("--operation", required=True,
                        choices=["io_scan", "filter_agg", "window_hsm", "join_op"],
                        help="Benchmark operation")
    parser.add_argument("--run-id", default="default",
                        help="Airflow dag_run_id (used for S3 result path)")
    parser.add_argument("--minio-endpoint",
                        default=os.environ.get("AWS_ENDPOINT_URL", "http://minio.hsm-bench.svc:9000"),
                        help="MinIO endpoint URL")
    parser.add_argument("--minio-access-key",
                        default=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"))
    parser.add_argument("--minio-secret-key",
                        default=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"))
    parser.add_argument("--s3-bucket", default="robotics-bench")
    parser.add_argument("--s3-prefix", default="benchmarks")
    parser.add_argument("--local-data-dir",
                        default="/data",
                        help="Local data root (used to build S3 URI; actual data is in MinIO)")
    parser.add_argument("--hardware", default="linux-k8s",
                        help="Hardware label for result metadata")
    parser.add_argument("--spark-master",
                        default=None,
                        help="Spark master URL (e.g. k8s://https://kubernetes.default.svc:6443)")
    parser.add_argument("--driver-memory", default="3g")
    parser.add_argument("--shuffle-partitions", type=int, default=8)
    parser.add_argument("--executor-instances", type=int, default=4)
    parser.add_argument("--executor-image", default=None,
                        help="Docker image for Spark executor pods")
    parser.add_argument("--k8s-namespace", default="hsm-bench")
    parser.add_argument("--rapids-jar", default=None,
                        help="Path to RAPIDS JAR inside executor pod (pyspark-rapids only)")

    args = parser.parse_args()

    # ── Configure MinIO env vars ───────────────────────────────────────────────
    configure_minio_env(
        endpoint=args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
    )

    # ── Resolve S3 data path ───────────────────────────────────────────────────
    local_path = str(Path(args.local_data_dir) / args.dataset / f"filesize={args.filesize}")
    use_s3a = args.framework in {"pyspark", "pyspark-rapids"}
    data_path = local_to_s3_uri(local_path, args.s3_bucket, args.s3_prefix, spark=use_s3a)

    # ── Build hardware config ──────────────────────────────────────────────────
    hardware_cfg = _build_hardware_cfg(args)

    # ── Cell ID for result key ─────────────────────────────────────────────────
    cell_id = f"{args.framework}__{args.dataset}__{args.filesize}__{args.operation}"

    print(f"\n{'='*60}")
    print(f"  HSM-PySpark Benchmark Pod")
    print(f"{'='*60}")
    print(f"  Framework : {args.framework}")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Filesize  : {args.filesize}")
    print(f"  Operation : {args.operation}")
    print(f"  Data path : {data_path}")
    print(f"  MinIO     : {args.minio_endpoint}")
    print(f"  Run ID    : {args.run_id}")
    print(f"{'='*60}\n")

    t_start = time.perf_counter()

    try:
        op_module = _load_operation(args.operation)
        result: MatrixResult = op_module.run(
            data_path=data_path,
            framework=args.framework,
            hardware_cfg=hardware_cfg,
            dataset=args.dataset,
            filesize_target=args.filesize,
            hardware=args.hardware,
        )
    except Exception as e:
        print(f"ERROR: operation raised exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Write error result to S3 so collect_results.py can tally it
        from benchmarks.operations import make_error_result
        result = make_error_result(
            hardware=args.hardware,
            framework=args.framework,
            dataset=args.dataset,
            filesize_target=args.filesize,
            operation=args.operation,
            status="err",
        )

    elapsed = time.perf_counter() - t_start
    print(f"\n  Status    : {result.status}")
    print(f"  Records   : {result.total_records:,}")
    if result.status == "ok":
        tp = result.throughput_records_per_sec
        if tp >= 1_000_000:
            print(f"  Throughput: {tp/1_000_000:.2f}M rec/s")
        else:
            print(f"  Throughput: {tp/1_000:.0f}K rec/s")
    print(f"  Wall time : {elapsed:.1f}s")

    # ── Write result to S3 ────────────────────────────────────────────────────
    try:
        _write_result_to_s3(result, args.run_id, cell_id, args.minio_endpoint)
    except Exception as e:
        # Print to stdout so Airflow logs capture it; don't fail the pod
        print(f"WARNING: could not write result to S3: {e}", file=sys.stderr)
        # Fallback: print JSON to stdout so Airflow can capture it
        print("RESULT_JSON:" + json.dumps(result.to_dict()))

    return 0 if result.status in ("ok", "oom") else 1


if __name__ == "__main__":
    sys.exit(main())
