#!/usr/bin/env python3
"""
Matrix Benchmark Orchestrator — sweeps all 5 dimensions.

Dimensions: Hardware × Framework × Dataset × File Size × Operation

Usage:
    # Mac CPU (no GPU)
    python benchmarks/matrix_benchmark.py \\
        --data-dir ./data \\
        --datasets behavior1k,droid,oxe,nvidia_physicalai \\
        --filesizes 1mb,10mb,100mb,1gb \\
        --operations io_scan,filter_agg,window_hsm,join \\
        --frameworks pyspark,polars,duckdb,dask,pandas \\
        --hardware mac-cpu \\
        --output-dir ./benchmarks/results

    # Linux GPU machine
    python benchmarks/matrix_benchmark.py \\
        --frameworks pyspark-rapids,cudf,pyspark,polars,duckdb \\
        --hardware linux-gpu \\
        --rapids-jar ~/hsm-pyspark/jars/rapids-4-spark_2.12-25.02.0.jar

    # Smoke test (no Spark, fast)
    python benchmarks/matrix_benchmark.py \\
        --datasets behavior1k --filesizes 1mb \\
        --operations io_scan,filter_agg \\
        --frameworks polars,duckdb \\
        --hardware mac-cpu
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.operations import MatrixResult, make_error_result

# ── Constants ─────────────────────────────────────────────────────────────────

ALL_DATASETS = ["behavior1k", "droid", "oxe", "nvidia_physicalai"]
ALL_FILESIZES = ["1mb", "10mb", "100mb", "1gb"]
ALL_OPERATIONS = ["io_scan", "filter_agg", "window_hsm", "join"]
ALL_FRAMEWORKS_CPU = ["pyspark", "polars", "duckdb", "dask", "pandas"]
ALL_FRAMEWORKS_GPU = ["pyspark-rapids", "cudf", "pyspark", "polars", "duckdb"]

_OOM_KEYWORDS = (
    "OutOfMemoryError", "OutOfMemoryException",
    "MemoryError", "Out of Memory",
)


# ── Subprocess isolation (mirrors simulate_scale._safe_run) ───────────────────

def _worker_fn(op_module_name: str, kwargs: dict, result_queue) -> None:
    """Subprocess target: import the op module, call run(**kwargs), put result."""
    try:
        import importlib
        mod = importlib.import_module(f"benchmarks.operations.{op_module_name}")
        result = mod.run(**kwargs)
        result_queue.put(("ok", result))
    except MemoryError:
        result_queue.put(("oom", "MemoryError"))
    except Exception as exc:
        exc_str = str(exc)
        if any(kw in exc_str for kw in _OOM_KEYWORDS):
            result_queue.put(("oom", exc_str[:300]))
        else:
            result_queue.put(("err", f"{type(exc).__name__}: {exc_str[:300]}"))


def _safe_run(
    label: str,
    op_module_name: str,
    kwargs: dict,
    timeout_s: int = 600,
) -> MatrixResult | str:
    """
    Run an operation in an isolated subprocess.

    Returns MatrixResult on success, or "oom"/"err" string on failure.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker_fn, args=(op_module_name, kwargs, q))
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        p.kill()
        p.join()
        print(f"    !! {label}: TIMEOUT after {timeout_s}s", flush=True)
        return "err"

    if p.exitcode == 0:
        try:
            status, payload = q.get_nowait()
        except Exception:
            print(f"    !! {label}: subprocess returned no result", flush=True)
            return "err"
        if status == "ok":
            return payload
        if status == "oom":
            print(f"    !! {label}: OOM — {str(payload)[:120]}", flush=True)
            return "oom"
        print(f"    !! {label}: {payload}", flush=True)
        return "err"

    if p.exitcode == -9:
        print(f"    !! {label}: killed by OS OOM (SIGKILL)", flush=True)
        return "oom"

    print(f"    !! {label}: subprocess exited with code {p.exitcode}", flush=True)
    return "err"


# ── Display helpers ───────────────────────────────────────────────────────────

def _fmt_result(result: MatrixResult | str, framework: str) -> str:
    if isinstance(result, str):
        icon = "!!" if result in ("oom", "err") else "  "
        return f"  {icon} {framework:<18} {result.upper()}"

    status_icon = "✓" if result.status == "ok" else "!!"
    throughput = result.throughput_records_per_sec
    if throughput >= 1_000_000:
        tp_str = f"{throughput/1_000_000:.1f}M rec/s"
    elif throughput >= 1_000:
        tp_str = f"{throughput/1_000:.0f}K rec/s"
    else:
        tp_str = f"{throughput:.0f} rec/s"

    mem_str = (
        f"< 1 MB" if result.peak_memory_mb < 0.5
        else f"{result.peak_memory_mb:>5.0f} MB"
    )
    gpu_str = (
        f"GPU: {result.gpu_memory_mb:.1f} GB"
        if result.gpu_memory_mb > 0
        else "GPU:  0 MB"
    )

    return (
        f"  {status_icon} {framework:<18} "
        f"{result.compute_time_s:>6.2f}s  "
        f"{tp_str:>12}  "
        f"{mem_str:>8}  "
        f"{gpu_str}"
    )


# ── Incremental JSON save ─────────────────────────────────────────────────────

def _save_results(
    results: list[dict],
    output_dir: Path,
    hardware: str,
    timestamp: str,
) -> Path:
    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"matrix_{hardware}_{timestamp}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "results": results,
    }
    out_file.write_text(json.dumps(payload, indent=2))
    return out_file


# ── Main orchestration ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HSM-PySpark Matrix Benchmark — 5-dimensional sweep"
    )
    parser.add_argument("--data-dir", default="./data", help="Root data directory")
    parser.add_argument(
        "--datasets",
        default=",".join(ALL_DATASETS),
        help=f"Comma-separated datasets (default: {','.join(ALL_DATASETS)})",
    )
    parser.add_argument(
        "--filesizes",
        default=",".join(ALL_FILESIZES),
        help=f"Comma-separated file sizes (default: {','.join(ALL_FILESIZES)})",
    )
    parser.add_argument(
        "--operations",
        default=",".join(ALL_OPERATIONS),
        help=f"Comma-separated operations (default: {','.join(ALL_OPERATIONS)})",
    )
    parser.add_argument(
        "--frameworks",
        default=",".join(ALL_FRAMEWORKS_CPU),
        help=f"Comma-separated frameworks (default: {','.join(ALL_FRAMEWORKS_CPU)})",
    )
    parser.add_argument(
        "--hardware",
        default="mac-cpu",
        choices=["mac-cpu", "linux-cpu", "linux-gpu"],
        help="Hardware label for results (default: mac-cpu)",
    )
    parser.add_argument(
        "--rapids-jar",
        default="",
        help="Path to rapids-4-spark_*.jar (required for pyspark-rapids)",
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmarks/results",
        help="Output directory for result JSON files",
    )
    parser.add_argument(
        "--spark-master",
        default="local[*]",
        help="Spark master URL (default: local[*])",
    )
    parser.add_argument(
        "--spark-driver-memory",
        default="8g",
        help="Spark driver memory (default: 8g)",
    )
    parser.add_argument(
        "--spark-shuffle-partitions",
        type=int,
        default=8,
        help="spark.sql.shuffle.partitions (default: 8)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-cell timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing partial results file (skip completed cells)",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    filesizes = [f.strip() for f in args.filesizes.split(",")]
    operations = [o.strip() for o in args.operations.split(",")]
    frameworks = [fw.strip() for fw in args.frameworks.split(",")]
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    hardware_cfg = {
        "rapids_jar": args.rapids_jar,
        "spark_config": {
            "master": args.spark_master,
            "driver_memory": args.spark_driver_memory,
            "shuffle_partitions": args.spark_shuffle_partitions,
        },
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results: list[dict] = []

    # Track previously completed cells for resume
    completed_cells: set[tuple] = set()
    if args.resume:
        existing = list(output_dir.glob(f"matrix_{args.hardware}_*.json"))
        for f in sorted(existing):
            try:
                data = json.loads(f.read_text())
                for r in data.get("results", []):
                    key = (r["dataset"], r["filesize_target"], r["operation"], r["framework"])
                    if r["status"] == "ok":
                        completed_cells.add(key)
                        results.append(r)
            except Exception:
                pass
        if completed_cells:
            print(f"  Resume: found {len(completed_cells)} completed cells")

    # Track OOM'd frameworks to skip at higher sizes
    oom_frameworks: set[str] = set()

    print("\n" + "=" * 70)
    print("HSM-PySpark: Matrix Benchmark")
    print("=" * 70)
    print(f"Hardware   : {args.hardware}")
    print(f"Datasets   : {datasets}")
    print(f"File sizes : {filesizes}")
    print(f"Operations : {operations}")
    print(f"Frameworks : {frameworks}")
    print(f"Data dir   : {data_dir}")
    print(f"Output dir : {output_dir}")
    print("=" * 70)

    op_module_map = {
        "io_scan": "io_scan",
        "filter_agg": "filter_agg",
        "window_hsm": "window_hsm",
        "join": "join_op",
    }

    for dataset in datasets:
        for filesize in filesizes:
            data_path = data_dir / dataset / f"filesize={filesize}"

            if not data_path.exists():
                print(f"\n[{dataset}/{filesize}] SKIP — data not found: {data_path}")
                print(f"  Run: python benchmarks/datasets/dataset_manager.py --datasets {dataset}")
                continue

            for operation in operations:
                print(f"\n[{dataset} / {filesize} / {operation}]")

                op_module = op_module_map.get(operation)
                if not op_module:
                    print(f"  Unknown operation: {operation}")
                    continue

                for framework in frameworks:
                    cell_key = (dataset, filesize, operation, framework)

                    # Skip previously completed (resume mode)
                    if cell_key in completed_cells:
                        print(f"  - {framework:<18} (already done, skipped)")
                        continue

                    # Skip if this framework OOM'd at a smaller size
                    oom_key = f"{framework}:{operation}"
                    if oom_key in oom_frameworks:
                        print(f"  - {framework:<18} SKIP (OOM at smaller size)")
                        skip_result = make_error_result(
                            args.hardware, framework, dataset, filesize, operation,
                            status="skip",
                        )
                        results.append(skip_result.to_dict())
                        continue

                    # Build kwargs for the worker
                    kwargs = {
                        "data_path": str(data_path),
                        "framework": framework,
                        "hardware_cfg": hardware_cfg,
                        "dataset": dataset,
                        "filesize_target": filesize,
                        "hardware": args.hardware,
                    }

                    label = f"{framework}/{dataset}/{filesize}/{operation}"
                    t_start = time.perf_counter()
                    result = _safe_run(label, op_module, kwargs, timeout_s=args.timeout)
                    elapsed = time.perf_counter() - t_start

                    if isinstance(result, str):
                        # oom or err from subprocess
                        status = result
                        if status == "oom":
                            oom_frameworks.add(oom_key)
                        mat_result = make_error_result(
                            args.hardware, framework, dataset, filesize, operation,
                            status=status,
                        )
                        results.append(mat_result.to_dict())
                        print(_fmt_result(status, framework))
                    else:
                        if result.status == "oom":
                            oom_frameworks.add(oom_key)
                        results.append(result.to_dict())
                        print(_fmt_result(result, framework))

                # Save incrementally after each (dataset, filesize, operation) triple
                out_file = _save_results(results, output_dir, args.hardware, timestamp)

        # Reset OOM tracking per dataset (different datasets have different sizes)
        oom_frameworks.clear()

    # Final save
    out_file = _save_results(results, output_dir, args.hardware, timestamp)
    print(f"\n{'=' * 70}")
    print(f"Matrix benchmark complete.")
    print(f"Results → {out_file}")
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    total = len(results)
    print(f"Cells: {ok_count}/{total} succeeded")
    print("=" * 70)


if __name__ == "__main__":
    main()
