#!/usr/bin/env bash
# setup/k3s_setup.sh
#
# Installs k3s + Helm + NVIDIA GPU device plugin on the benchmark Linux machine.
# Creates the hsm-bench namespace and RBAC for Spark-on-k8s.
#
# Usage:
#   bash setup/k3s_setup.sh
#
# Requirements:
#   - Ubuntu 20.04/22.04 with NVIDIA drivers already installed (nvidia-smi works)
#   - Internet access (downloads k3s, helm, RAPIDS JAR)
#   - Run as a user with sudo privileges

set -euo pipefail

NAMESPACE="hsm-bench"
RAPIDS_JAR_DIR="/opt/rapids"
RAPIDS_JAR_NAME="rapids-4-spark_2.12-25.02.0-cuda12.jar"
# Official RAPIDS 25.02 artifact (Spark 3.5, CUDA 12)
RAPIDS_JAR_URL="https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/25.02.0/rapids-4-spark_2.12-25.02.0.jar"

# ── 1. Install k3s (no traefik, keep kube-proxy) ─────────────────────────────
echo "==> Installing k3s..."
if command -v k3s &>/dev/null; then
    echo "    k3s already installed: $(k3s --version | head -1)"
else
    curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server --disable=traefik" sh -
    echo "    k3s installed."
fi

# Export kubeconfig for current user
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
echo "    KUBECONFIG=/etc/rancher/k3s/k3s.yaml"
# Make it readable by non-root (helm needs it)
sudo chmod 644 /etc/rancher/k3s/k3s.yaml

# Wait for k3s to be ready
echo "==> Waiting for k3s node to be Ready..."
kubectl wait --for=condition=Ready node --all --timeout=120s

# ── 2. Install Helm ───────────────────────────────────────────────────────────
echo "==> Installing Helm..."
if command -v helm &>/dev/null; then
    echo "    Helm already installed: $(helm version --short)"
else
    curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi

# ── 3. Add Helm repos ─────────────────────────────────────────────────────────
echo "==> Adding Helm repos..."
helm repo add minio https://charts.min.io/ 2>/dev/null || true
helm repo add apache-airflow https://airflow.apache.org 2>/dev/null || true
helm repo update

# ── 4. NVIDIA GPU device plugin ───────────────────────────────────────────────
echo "==> Installing NVIDIA GPU device plugin..."

# Label nodes with NVIDIA GPU (k3s auto-detects, but label explicitly)
for node in $(kubectl get nodes -o jsonpath='{.items[*].metadata.name}'); do
    kubectl label node "$node" nvidia.com/gpu=present --overwrite 2>/dev/null || true
done

# Apply the official NVIDIA device plugin DaemonSet
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml

echo "==> Waiting for GPU device plugin to be ready..."
kubectl -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout=120s || true

# ── 5. Create namespace ───────────────────────────────────────────────────────
echo "==> Creating namespace ${NAMESPACE}..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kubectl apply -f "${SCRIPT_DIR}/k8s/namespace.yaml"

# ── 6. Apply RBAC ─────────────────────────────────────────────────────────────
echo "==> Applying RBAC..."
kubectl apply -f "${SCRIPT_DIR}/k8s/rbac.yaml"

# ── 7. Download RAPIDS JAR to host ───────────────────────────────────────────
echo "==> Downloading RAPIDS JAR..."
sudo mkdir -p "${RAPIDS_JAR_DIR}"
if [[ -f "${RAPIDS_JAR_DIR}/${RAPIDS_JAR_NAME}" ]]; then
    echo "    Already present: ${RAPIDS_JAR_DIR}/${RAPIDS_JAR_NAME}"
else
    echo "    Downloading from Maven Central..."
    sudo curl -L --progress-bar -o "${RAPIDS_JAR_DIR}/${RAPIDS_JAR_NAME}" "${RAPIDS_JAR_URL}"
    echo "    Downloaded: $(du -sh ${RAPIDS_JAR_DIR}/${RAPIDS_JAR_NAME})"
fi
sudo chmod 644 "${RAPIDS_JAR_DIR}/${RAPIDS_JAR_NAME}"

# ── 8. Verify GPU resources visible to k8s ───────────────────────────────────
echo ""
echo "==> Cluster status:"
kubectl get nodes -o wide
echo ""
echo "==> GPU resource capacity:"
kubectl get nodes -o json | python3 -c "
import json, sys
nodes = json.load(sys.stdin)['items']
for n in nodes:
    cap = n['status'].get('capacity', {})
    gpu = cap.get('nvidia.com/gpu', '0')
    name = n['metadata']['name']
    print(f'  {name}: nvidia.com/gpu={gpu}')
" 2>/dev/null || kubectl get nodes -o custom-columns="NAME:.metadata.name,GPU:.status.capacity.nvidia\\.com/gpu"

echo ""
echo "==> k3s setup complete."
echo "    Namespace      : ${NAMESPACE}"
echo "    RAPIDS JAR     : ${RAPIDS_JAR_DIR}/${RAPIDS_JAR_NAME}"
echo "    Next: run setup/deploy_infra.sh"
