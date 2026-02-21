#!/usr/bin/env bash
# setup/deploy_infra.sh
#
# Deploys MinIO + Airflow via Helm into the hsm-bench namespace,
# creates the robotics-bench bucket, and seeds data from HuggingFace.
#
# Usage:
#   bash setup/deploy_infra.sh [--skip-data-seed]
#
# Prerequisites:
#   - k3s running (setup/k3s_setup.sh completed)
#   - Helm installed
#   - Python env with hsm-pyspark[benchmark] installed (for dataset_manager.py)

set -euo pipefail

NAMESPACE="hsm-bench"
MINIO_RELEASE="minio"
AIRFLOW_RELEASE="airflow"
BUCKET_NAME="robotics-bench"
MINIO_ENDPOINT="http://localhost:9000"          # host-level port-forward during seeding
MINIO_K8S_ENDPOINT="http://minio.${NAMESPACE}.svc:9000"  # in-cluster
MINIO_USER="minioadmin"
MINIO_PASS="minioadmin"

SKIP_SEED=false
for arg in "$@"; do
    [[ "$arg" == "--skip-data-seed" ]] && SKIP_SEED=true
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

# ── 1. Deploy MinIO ───────────────────────────────────────────────────────────
echo "==> Deploying MinIO..."
helm upgrade --install "${MINIO_RELEASE}" minio/minio \
    --namespace "${NAMESPACE}" \
    --values "${SCRIPT_DIR}/helm_values/minio-values.yaml" \
    --wait --timeout 5m

echo "    MinIO deployed."

# ── 2. Port-forward MinIO for local access during seeding ─────────────────────
echo "==> Setting up MinIO port-forward (background)..."
kubectl port-forward svc/minio 9000:9000 9001:9001 -n "${NAMESPACE}" &
PF_PID=$!
trap "kill ${PF_PID} 2>/dev/null || true" EXIT
sleep 5  # wait for port-forward to be ready

# ── 3. Create bucket via mc (MinIO client) ────────────────────────────────────
echo "==> Creating bucket '${BUCKET_NAME}'..."
if command -v mc &>/dev/null; then
    mc alias set local "${MINIO_ENDPOINT}" "${MINIO_USER}" "${MINIO_PASS}" --api S3v4 2>/dev/null || true
    mc mb --ignore-existing "local/${BUCKET_NAME}" || true
    echo "    Bucket created via mc."
else
    # Fallback: use Python boto3
    python3 - <<PYEOF
import boto3, botocore
s3 = boto3.client(
    "s3",
    endpoint_url="${MINIO_ENDPOINT}",
    aws_access_key_id="${MINIO_USER}",
    aws_secret_access_key="${MINIO_PASS}",
    region_name="us-east-1",
)
try:
    s3.head_bucket(Bucket="${BUCKET_NAME}")
    print("    Bucket already exists.")
except botocore.exceptions.ClientError:
    s3.create_bucket(Bucket="${BUCKET_NAME}")
    print("    Bucket '${BUCKET_NAME}' created.")
PYEOF
fi

# ── 4. Deploy Airflow ─────────────────────────────────────────────────────────
echo "==> Deploying Airflow..."
helm upgrade --install "${AIRFLOW_RELEASE}" apache-airflow/airflow \
    --namespace "${NAMESPACE}" \
    --values "${SCRIPT_DIR}/helm_values/airflow-values.yaml" \
    --wait --timeout 10m

echo "    Airflow deployed."

# ── 5. Seed data from HuggingFace (optional) ─────────────────────────────────
if [[ "${SKIP_SEED}" == "false" ]]; then
    echo "==> Seeding robotics datasets from HuggingFace into MinIO..."
    echo "    This downloads ~10 GB of data. Use --skip-data-seed to skip."
    cd "${REPO_ROOT}"
    python3 benchmarks/datasets/dataset_manager.py \
        --datasets behavior1k lerobot gr00t \
        --filesizes 1mb 10mb \
        --output-dir ./data \
        --minio-endpoint "${MINIO_ENDPOINT}" \
        --minio-access-key "${MINIO_USER}" \
        --minio-secret-key "${MINIO_PASS}" \
        --bucket "${BUCKET_NAME}"
    echo "    Data seeding complete."
else
    echo "    Skipping data seed (--skip-data-seed)."
fi

# ── 6. Print access info ──────────────────────────────────────────────────────
echo ""
echo "==> Infrastructure deployed successfully."
echo ""
echo "  MinIO console  : kubectl port-forward svc/minio 9001:9001 -n ${NAMESPACE}"
echo "                   → http://localhost:9001  (${MINIO_USER} / ${MINIO_PASS})"
echo ""
echo "  Airflow UI     : kubectl port-forward svc/airflow-webserver 8080:8080 -n ${NAMESPACE}"
echo "                   → http://localhost:8080  (admin / admin)"
echo ""
echo "  Trigger DAG    : Open Airflow UI → benchmark_matrix → trigger"
echo ""
echo "  Watch pods     : kubectl get pods -n ${NAMESPACE} -w"
