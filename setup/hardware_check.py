#!/usr/bin/env python3
"""
Hardware Preflight Check for RAPIDS GPU Benchmarks.

Verifies:
- nvidia-smi lists 2× RTX 3090 (24 GB VRAM each)
- cuDF import succeeds
- RAPIDS JAR exists
- Available GPU VRAM >= 20 GB per card
- PySpark import succeeds with correct Java
- Java 17 available

Usage:
    python setup/hardware_check.py [--jar-dir ./jars]
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Color helpers ─────────────────────────────────────────────────────────────
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"

PASS = f"{_GREEN}PASS{_RESET}"
FAIL = f"{_RED}FAIL{_RESET}"
WARN = f"{_YELLOW}WARN{_RESET}"

results: list[tuple[str, str, str]] = []  # (check, status, detail)


def check(name: str, passed: bool, detail: str = "", warn_only: bool = False) -> bool:
    status = PASS if passed else (WARN if warn_only else FAIL)
    results.append((name, status, detail))
    return passed


# ── Individual checks ─────────────────────────────────────────────────────────

def check_nvidia_smi() -> bool:
    if not shutil.which("nvidia-smi"):
        return check("nvidia-smi", False, "nvidia-smi not found in PATH")

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip()
    except subprocess.CalledProcessError as e:
        return check("nvidia-smi", False, str(e))

    gpus = [line.strip() for line in out.splitlines() if line.strip()]
    gpu_info = "; ".join(gpus)

    rtx3090_count = sum(1 for g in gpus if "RTX 3090" in g)
    ok = rtx3090_count >= 2
    check(
        "GPU count (expect 2× RTX 3090)",
        ok,
        f"Found {rtx3090_count}× RTX 3090 — {gpu_info}",
        warn_only=(rtx3090_count == 1),
    )
    return ok


def check_gpu_vram() -> bool:
    try:
        import pynvml

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        min_free_mb = float("inf")
        details = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            free_gb = mem.free / 1024 ** 3
            total_gb = mem.total / 1024 ** 3
            details.append(f"GPU{i} {name}: {free_gb:.1f}/{total_gb:.1f} GB free")
            min_free_mb = min(min_free_mb, mem.free / 1024 ** 2)
        pynvml.nvmlShutdown()
        ok = min_free_mb >= 20 * 1024  # 20 GB
        return check("GPU VRAM >= 20 GB/card", ok, "; ".join(details))
    except ImportError:
        return check("GPU VRAM check", False, "pynvml not installed — pip install nvidia-ml-py3")
    except Exception as e:
        return check("GPU VRAM check", False, str(e))


def check_cudf() -> bool:
    try:
        import cudf  # noqa: F401

        ver = cudf.__version__
        return check("cuDF import", True, f"version {ver}")
    except ImportError as e:
        return check("cuDF import", False, str(e))


def check_rapids_jar(jar_dir: str) -> bool:
    pattern = str(Path(jar_dir) / "rapids-4-spark_*.jar")
    matches = glob.glob(pattern)
    ok = len(matches) > 0
    detail = matches[0] if ok else f"No JAR found at {pattern}"
    return check("RAPIDS Spark JAR", ok, detail)


def check_pyspark() -> bool:
    try:
        import pyspark

        ver = pyspark.__version__
        return check("PySpark import", True, f"version {ver}")
    except ImportError as e:
        return check("PySpark import", False, str(e))


def check_java() -> bool:
    java = shutil.which("java")
    if not java:
        return check("Java 17", False, "java not found in PATH")
    try:
        out = subprocess.check_output(
            ["java", "-version"], stderr=subprocess.STDOUT, text=True, timeout=10
        )
        ok = "17" in out or "openjdk 17" in out.lower()
        return check("Java 17", ok, out.strip().splitlines()[0] if out.strip() else "")
    except subprocess.CalledProcessError as e:
        return check("Java 17", False, str(e))


def check_polars() -> bool:
    try:
        import polars

        return check("Polars import", True, f"version {polars.__version__}")
    except ImportError as e:
        return check("Polars import", False, str(e))


def check_duckdb() -> bool:
    try:
        import duckdb

        return check("DuckDB import", True, f"version {duckdb.__version__}")
    except ImportError as e:
        return check("DuckDB import", False, str(e))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAPIDS GPU hardware preflight check")
    parser.add_argument(
        "--jar-dir",
        default=str(Path(__file__).parent.parent / "jars"),
        help="Directory to look for RAPIDS Spark JAR",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  HSM-PySpark — GPU Hardware Preflight Check")
    print("=" * 60)

    check_nvidia_smi()
    check_gpu_vram()
    check_cudf()
    check_rapids_jar(args.jar_dir)
    check_pyspark()
    check_java()
    check_polars()
    check_duckdb()

    # Print results table
    print()
    col_w = max(len(r[0]) for r in results) + 2
    for name, status, detail in results:
        detail_str = f"  [{detail}]" if detail else ""
        print(f"  {name:<{col_w}} {status}{detail_str}")

    print()
    failed = [r for r in results if "FAIL" in r[1]]
    warned = [r for r in results if "WARN" in r[1]]

    if not failed:
        print(f"  {_GREEN}All checks passed!{_RESET}", end="")
        if warned:
            print(f"  ({len(warned)} warning(s))", end="")
        print()
        sys.exit(0)
    else:
        print(f"  {_RED}{len(failed)} check(s) FAILED.{_RESET} Resolve before running GPU benchmarks.")
        sys.exit(1)


if __name__ == "__main__":
    main()
