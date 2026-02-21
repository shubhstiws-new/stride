#!/usr/bin/env bash
# ============================================================
# GPU Machine One-Time Setup for HSM-PySpark RAPIDS Benchmarks
# ============================================================
# Requirements: CUDA 12, dual RTX 3090, Linux, conda installed
# Usage: bash setup/gpu_setup.sh [--repo-url <git-url>]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="rapids-spark"
RAPIDS_VERSION="25.02"
CUDA_VERSION="12.0"
PYSPARK_VERSION="4.0.0"
JAR_DIR="$REPO_ROOT/jars"
JAR_NAME="rapids-4-spark_2.12-25.02.0.jar"
JAR_URL="https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/25.02.0/rapids-4-spark_2.12-25.02.0.jar"

echo "============================================================"
echo " HSM-PySpark GPU Setup"
echo "============================================================"

# ── Step 1: Verify CUDA + dual RTX 3090 ─────────────────────
echo ""
echo "[1/5] Verifying CUDA and GPU hardware..."
nvidia-smi || { echo "ERROR: nvidia-smi failed. Check NVIDIA driver."; exit 1; }
nvcc --version || { echo "ERROR: nvcc not found. Install CUDA toolkit."; exit 1; }

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c "RTX 3090" || true)
echo "  Found ${GPU_COUNT}× RTX 3090"
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "  WARNING: Expected 2× RTX 3090, found ${GPU_COUNT}. Proceeding anyway."
fi

# ── Step 2: Create RAPIDS conda environment ──────────────────
echo ""
echo "[2/5] Creating conda environment '${CONDA_ENV}'..."
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  Environment '${CONDA_ENV}' already exists. Skipping creation."
    echo "  To recreate: conda env remove -n ${CONDA_ENV} && bash setup/gpu_setup.sh"
else
    conda create -n "${CONDA_ENV}" -y \
        -c rapidsai -c nvidia -c conda-forge \
        "cudf=${RAPIDS_VERSION}" \
        rmm \
        "cuda-toolkit=${CUDA_VERSION}" \
        python=3.11 \
        openjdk=17 \
        pandas \
        polars \
        duckdb \
        pyarrow \
        psutil \
        pynvml \
        huggingface_hub \
        datasets \
        paramiko
    echo "  Conda environment created."
fi

# Install pyspark + other pip packages into the env
conda run -n "${CONDA_ENV}" pip install \
    "pyspark>=${PYSPARK_VERSION}" \
    "dask[dataframe]>=2024.1" \
    tabulate \
    pyyaml \
    pydantic \
    --quiet
echo "  pip packages installed."

# ── Step 3: Download RAPIDS Spark JAR ───────────────────────
echo ""
echo "[3/5] Downloading RAPIDS Spark JAR..."
mkdir -p "$JAR_DIR"
JAR_PATH="$JAR_DIR/$JAR_NAME"
if [ -f "$JAR_PATH" ]; then
    echo "  JAR already exists: $JAR_PATH"
else
    echo "  Downloading from Maven Central..."
    wget -q --show-progress -O "$JAR_PATH" "$JAR_URL"
    echo "  Downloaded: $JAR_PATH"
fi
JAR_SIZE=$(du -sh "$JAR_PATH" | cut -f1)
echo "  JAR size: $JAR_SIZE"

# ── Step 4: Verify repo structure ───────────────────────────
echo ""
echo "[4/5] Verifying repository structure..."
for required_path in \
    "benchmarks/matrix_benchmark.py" \
    "benchmarks/datasets/dataset_manager.py" \
    "benchmarks/operations/__init__.py" \
    "setup/hardware_check.py"
do
    full_path="$REPO_ROOT/$required_path"
    if [ -f "$full_path" ]; then
        echo "  OK: $required_path"
    else
        echo "  MISSING: $required_path — run from the repo root after git pull"
    fi
done

# ── Step 5: Preflight hardware check ────────────────────────
echo ""
echo "[5/5] Running preflight hardware check..."
conda run -n "${CONDA_ENV}" python "$SCRIPT_DIR/hardware_check.py" || true

echo ""
echo "============================================================"
echo " Setup complete."
echo " To run benchmarks:"
echo "   conda activate ${CONDA_ENV}"
echo "   python benchmarks/matrix_benchmark.py \\"
echo "     --hardware linux-gpu \\"
echo "     --rapids-jar $JAR_PATH \\"
echo "     --datasets behavior1k,droid,oxe,nvidia_physicalai \\"
echo "     --filesizes 1mb,10mb,100mb,1gb"
echo "============================================================"
