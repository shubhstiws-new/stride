"""
airflow/dags/gpu_scale_dag.py

GPU scale sweep DAG.
Runs the window_hsm operation across increasing scale factors using cuDF and
pyspark-rapids frameworks. Each (scale × framework) = its own k8s pod.

Scale factors mirror the CPU scale sweep but capped at GPU VRAM limits:
  cuDF:           1×, 10×, 50×  (OOM expected at 100×+ on RTX 3090 24 GB)
  pyspark-rapids: 1×, 10×, 50×, 100×, 500×

Trigger via Airflow UI or CLI:
    airflow dags trigger gpu_scale_sweep
"""

from __future__ import annotations

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

CPU_IMAGE = Variable.get("hsm_bench_cpu_image", default_var="shubhstiwsdocker/hsm-bench-worker:latest")
GPU_IMAGE = Variable.get("hsm_bench_gpu_image", default_var="shubhstiwsdocker/hsm-bench-worker-gpu:latest")

MINIO_ENDPOINT = "http://minio.hsm-bench.svc:9000"
SPARK_MASTER   = "k8s://https://kubernetes.default.svc:6443"
NAMESPACE      = "hsm-bench"
BUCKET         = "robotics-bench"

# Scale factors and which frameworks run at each scale
_SCALE_PLANS = [
    # (scale_factor, frameworks_to_run)
    (1,   ["cudf", "pyspark-rapids"]),
    (10,  ["cudf", "pyspark-rapids"]),
    (50,  ["cudf", "pyspark-rapids"]),
    (100, ["pyspark-rapids"]),          # cudf OOM expected here
    (500, ["pyspark-rapids"]),
]

# Dataset for scale sweep (behavior1k 1mb files, replicated N× in MinIO)
_SWEEP_DATASET = "behavior1k"
_SWEEP_FILESIZE = "1mb"


# ── Resource helpers ──────────────────────────────────────────────────────────

def _cudf_resources() -> k8s.V1ResourceRequirements:
    """cuDF pod: 2 cores + 8 GB RAM + 1 GPU."""
    return k8s.V1ResourceRequirements(
        requests={"cpu": "2", "memory": "8Gi", "nvidia.com/gpu": "1"},
        limits={"cpu": "2",   "memory": "8Gi", "nvidia.com/gpu": "1"},
    )


def _spark_driver_resources() -> k8s.V1ResourceRequirements:
    return k8s.V1ResourceRequirements(
        requests={"cpu": "2", "memory": "3Gi"},
        limits={"cpu": "2",   "memory": "3Gi"},
    )


def _env_vars() -> list[k8s.V1EnvVar]:
    return [
        k8s.V1EnvVar(name="AWS_ENDPOINT_URL",     value=MINIO_ENDPOINT),
        k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID",     value="minioadmin"),
        k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value="minioadmin"),
        k8s.V1EnvVar(name="AWS_DEFAULT_REGION",    value="us-east-1"),
        k8s.V1EnvVar(name="PYTHONUNBUFFERED",      value="1"),
    ]


def _rapids_volumes() -> tuple[list[k8s.V1Volume], list[k8s.V1VolumeMount]]:
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

def make_scale_task(
    dag: DAG,
    scale: int,
    framework: str,
    run_id: str,
) -> KubernetesPodOperator:
    task_id = f"{framework}__scale_{scale}x"

    cmd_args = [
        "--framework", framework,
        "--dataset",   _SWEEP_DATASET,
        "--filesize",  _SWEEP_FILESIZE,
        "--operation", "window_hsm",
        "--run-id",    f"{run_id}__scale",
        "--minio-endpoint", MINIO_ENDPOINT,
        "--hardware",  "linux-k8s-gpu",
        "--s3-prefix", f"scale_sweep/scale_{scale}x",
    ]

    if framework == "pyspark-rapids":
        cmd_args += [
            "--spark-master", SPARK_MASTER,
            "--executor-instances", "2",  # 2 GPU executors (one per RTX 3090)
            "--executor-image", CPU_IMAGE,
            "--k8s-namespace", NAMESPACE,
            "--driver-memory", "3g",
            "--rapids-jar", "/opt/rapids/rapids-4-spark_2.12-25.02.0-cuda12.jar",
        ]
        image     = CPU_IMAGE
        resources = _spark_driver_resources()
        volumes, mounts = _rapids_volumes()
        sa_name = "spark"
    else:
        image     = GPU_IMAGE
        resources = _cudf_resources()
        volumes, mounts = [], []
        sa_name = "default"

    return KubernetesPodOperator(
        task_id=task_id,
        name=f"scale-{framework.replace('-', '')}-{scale}x",
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
        execution_timeout=timedelta(minutes=90),
        retries=1,
        retry_delay=timedelta(minutes=5),
        dag=dag,
        trigger_rule="all_done",
    )


def make_collect_task(dag: DAG, run_id: str) -> KubernetesPodOperator:
    return KubernetesPodOperator(
        task_id="collect_scale_results",
        name="scale-collect-results",
        namespace=NAMESPACE,
        image=CPU_IMAGE,
        arguments=[
            "python3", "benchmarks/collect_results_s3.py",
            "--run-id", f"{run_id}__scale",
            "--minio-endpoint", MINIO_ENDPOINT,
            "--bucket", BUCKET,
        ],
        env_vars=_env_vars(),
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "512Mi"},
            limits={"cpu": "1",     "memory": "1Gi"},
        ),
        is_delete_operator_pod=True,
        get_logs=True,
        startup_timeout_seconds=120,
        execution_timeout=timedelta(minutes=10),
        dag=dag,
        trigger_rule="all_done",
    )


# ── DAG definition ────────────────────────────────────────────────────────────

default_args = {
    "owner": "hsm-bench",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 0,
}

with DAG(
    dag_id="gpu_scale_sweep",
    description="GPU scale sweep: cuDF + PySpark-RAPIDS across 1×–500× scale factors",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    max_active_tasks=4,   # 2 GPUs → max 2 concurrent GPU pods; plus some Spark pods
    tags=["benchmark", "gpu", "sprint3b"],
) as dag:

    _run_id = "{{ run_id }}"

    all_tasks = []
    prev_scale_tasks: list = []

    for scale, frameworks in _SCALE_PLANS:
        with TaskGroup(group_id=f"scale_{scale}x") as scale_group:
            scale_tasks = []
            for fw in frameworks:
                t = make_scale_task(dag=dag, scale=scale, framework=fw, run_id=_run_id)
                scale_tasks.append(t)
                all_tasks.append(t)

        # Run scales sequentially (each scale × framework can run in parallel within the group)
        if prev_scale_tasks:
            for prev in prev_scale_tasks:
                prev >> scale_group  # type: ignore[operator]
        prev_scale_tasks = scale_tasks

    collect = make_collect_task(dag=dag, run_id=_run_id)
    for t in all_tasks:
        t >> collect
