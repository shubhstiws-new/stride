"""
airflow/dags/benchmark_matrix_dag.py

Parametric benchmark matrix DAG.
One KubernetesPodOperator task per (framework × dataset × filesize × operation).
GPU frameworks get the GPU image + nvidia.com/gpu: 1 resource request.
PySpark tasks use the spark ServiceAccount + larger pod resources.
Final task collects results from MinIO and pivots them.

Trigger via Airflow UI or CLI:
    airflow dags trigger benchmark_matrix --conf '{"run_note": "sprint3b"}'
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup

try:
    from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
    from kubernetes.client import models as k8s
except ImportError:
    raise ImportError(
        "Install: pip install apache-airflow-providers-cncf-kubernetes kubernetes"
    )

# ── Configuration ─────────────────────────────────────────────────────────────

# Docker Hub image names (set as Airflow Variables or hard-code here)
CPU_IMAGE = Variable.get("hsm_bench_cpu_image", default_var="shubhstiwsdocker/hsm-bench-worker:latest")
GPU_IMAGE = Variable.get("hsm_bench_gpu_image", default_var="shubhstiwsdocker/hsm-bench-worker-gpu:latest")

MINIO_ENDPOINT = "http://minio.hsm-bench.svc:9000"
SPARK_MASTER   = "k8s://https://kubernetes.default.svc:6443"
NAMESPACE      = "hsm-bench"
BUCKET         = "robotics-bench"

DATASETS  = ["behavior1k", "droid", "oxe"]
FILESIZES = ["1mb", "10mb"]
OPERATIONS = ["io_scan", "filter_agg", "window_hsm", "join_op"]

# Frameworks and their execution profile
# profile: "cpu" | "spark" | "spark-rapids" | "gpu"
FRAMEWORKS = [
    {"name": "polars",         "profile": "cpu"},
    {"name": "duckdb",         "profile": "cpu"},
    {"name": "pandas",         "profile": "cpu"},
    {"name": "dask",           "profile": "cpu"},
    {"name": "pyspark",        "profile": "spark"},
    {"name": "pyspark-rapids", "profile": "spark-rapids"},
    {"name": "cudf",           "profile": "gpu"},
]

# ── Resource definitions ───────────────────────────────────────────────────────

def _cpu_resources() -> k8s.V1ResourceRequirements:
    return k8s.V1ResourceRequirements(
        requests={"cpu": "1", "memory": "1Gi"},
        limits={"cpu": "1",   "memory": "1Gi"},
    )


def _spark_driver_resources() -> k8s.V1ResourceRequirements:
    """Spark driver pod: 2 cores + 3 GB (driver_memory=2g + overhead)."""
    return k8s.V1ResourceRequirements(
        requests={"cpu": "2", "memory": "3Gi"},
        limits={"cpu": "2",   "memory": "3Gi"},
    )


def _gpu_resources() -> k8s.V1ResourceRequirements:
    return k8s.V1ResourceRequirements(
        requests={"cpu": "2", "memory": "8Gi", "nvidia.com/gpu": "1"},
        limits={"cpu": "2",   "memory": "8Gi", "nvidia.com/gpu": "1"},
    )


def _env_vars() -> list[k8s.V1EnvVar]:
    return [
        k8s.V1EnvVar(name="AWS_ENDPOINT_URL",       value=MINIO_ENDPOINT),
        k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID",       value="minioadmin"),
        k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY",   value="minioadmin"),
        k8s.V1EnvVar(name="AWS_DEFAULT_REGION",      value="us-east-1"),
        k8s.V1EnvVar(name="PYTHONUNBUFFERED",        value="1"),
    ]


def _rapids_host_path_volumes() -> tuple[list[k8s.V1Volume], list[k8s.V1VolumeMount]]:
    """Mount /opt/rapids from host node into the pod for the RAPIDS JAR."""
    volumes = [
        k8s.V1Volume(
            name="rapids-jar",
            host_path=k8s.V1HostPathVolumeSource(path="/opt/rapids", type="Directory"),
        )
    ]
    mounts = [
        k8s.V1VolumeMount(name="rapids-jar", mount_path="/opt/rapids", read_only=True)
    ]
    return volumes, mounts


# ── Task factory ──────────────────────────────────────────────────────────────

def make_task(
    dag: DAG,
    framework: str,
    profile: str,
    dataset: str,
    filesize: str,
    operation: str,
    run_id: str,
) -> KubernetesPodOperator:
    """Create a KubernetesPodOperator for one benchmark matrix cell."""

    task_id = f"{framework}__{dataset}__{filesize}__{operation}"

    # Base CLI args for run_single.py
    cmd_args = [
        "--framework", framework,
        "--dataset",   dataset,
        "--filesize",  filesize,
        "--operation", operation,
        "--run-id",    run_id,
        "--minio-endpoint", MINIO_ENDPOINT,
        "--hardware", "linux-k8s",
    ]

    if profile in ("spark", "spark-rapids"):
        cmd_args += [
            "--spark-master", SPARK_MASTER,
            "--executor-instances", "4",
            "--executor-image", CPU_IMAGE,
            "--k8s-namespace", NAMESPACE,
            "--driver-memory", "2g",
        ]
    if profile == "spark-rapids":
        cmd_args += ["--rapids-jar", "/opt/rapids/rapids-4-spark_2.12-25.02.0-cuda12.jar"]

    # Choose image and resources
    if profile == "gpu":
        image     = GPU_IMAGE
        resources = _gpu_resources()
        volumes, mounts = [], []
    elif profile == "spark-rapids":
        image     = CPU_IMAGE   # driver uses CPU image; RAPIDS JAR on hostPath
        resources = _spark_driver_resources()
        volumes, mounts = _rapids_host_path_volumes()
    elif profile == "spark":
        image     = CPU_IMAGE
        resources = _spark_driver_resources()
        volumes, mounts = [], []
    else:
        image     = CPU_IMAGE
        resources = _cpu_resources()
        volumes, mounts = [], []

    sa_name = "spark" if profile in ("spark", "spark-rapids") else "default"

    # GPU pods need the nvidia RuntimeClass to access GPUs
    extra_kwargs = {}
    if profile in ("gpu", "spark-rapids"):
        extra_kwargs["pod_override"] = k8s.V1Pod(
            spec=k8s.V1PodSpec(
                runtime_class_name="nvidia",
                containers=[k8s.V1Container(name="base")],
            )
        )

    return KubernetesPodOperator(
        task_id=task_id,
        name=f"bench-{task_id.replace('_', '-').replace('--', '-')}",
        namespace=NAMESPACE,
        image=image,
        arguments=cmd_args,
        env_vars=_env_vars(),
        container_resources=resources,
        volumes=volumes,
        volume_mounts=mounts,
        service_account_name=sa_name,
        is_delete_operator_pod=True,
        get_logs=True,
        log_events_on_failure=True,
        startup_timeout_seconds=300,
        # Allow GPU / Spark tasks to take longer
        execution_timeout=timedelta(minutes=60 if profile in ("spark", "spark-rapids", "gpu") else 30),
        # Retry once on transient failures
        retries=1,
        retry_delay=timedelta(minutes=2),
        dag=dag,
        # Don't fail the whole DAG if one cell fails
        trigger_rule="all_done",
        **extra_kwargs,
    )


# ── Collect results task ──────────────────────────────────────────────────────

def make_collect_task(dag: DAG, run_id: str) -> KubernetesPodOperator:
    """Final task: merge all per-cell JSON results into a combined pivot table."""
    return KubernetesPodOperator(
        task_id="collect_results",
        name="bench-collect-results",
        namespace=NAMESPACE,
        image=CPU_IMAGE,
        cmds=["python3"],
        arguments=[
            "/app/benchmarks/collect_results_s3.py",
            "--run-id", run_id,
            "--minio-endpoint", MINIO_ENDPOINT,
            "--bucket", BUCKET,
        ],
        env_vars=_env_vars(),
        container_resources=_cpu_resources(),
        is_delete_operator_pod=True,
        get_logs=True,
        startup_timeout_seconds=120,
        execution_timeout=timedelta(minutes=10),
        dag=dag,
        trigger_rule="all_done",
    )


# ── DAG definition ─────────────────────────────────────────────────────────────

default_args = {
    "owner": "hsm-bench",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 0,
}

with DAG(
    dag_id="benchmark_matrix",
    description="Full benchmark matrix: frameworks × datasets × filesizes × operations",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,       # manual trigger only
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,   # sequential execution for clean, unbiased results
    tags=["benchmark", "sprint3b"],
) as dag:

    # Use dag_run.run_id as the S3 results sub-prefix
    _run_id = "{{ run_id }}"

    all_cell_tasks = []

    for fw_def in FRAMEWORKS:
        fw_name    = fw_def["name"]
        fw_profile = fw_def["profile"]

        with TaskGroup(group_id=f"fw_{fw_name.replace('-', '_')}") as fw_group:
            for dataset in DATASETS:
                for filesize in FILESIZES:
                    for operation in OPERATIONS:
                        t = make_task(
                            dag=dag,
                            framework=fw_name,
                            profile=fw_profile,
                            dataset=dataset,
                            filesize=filesize,
                            operation=operation,
                            run_id=_run_id,
                        )
                        all_cell_tasks.append(t)

    collect = make_collect_task(dag=dag, run_id=_run_id)

    # All cell tasks must complete (succeed or fail) before collecting results
    for t in all_cell_tasks:
        t >> collect
