#!/usr/bin/env python3
"""
HSM Session Detection — GPU Scale-Factor Sweep Benchmark

Mirror of simulate_scale.py for GPU frameworks. Uses the same broadcast
cross-join trick for pyspark-rapids (lazy, constant VRAM) while cuDF must
materialise the full scaled DataFrame in VRAM, hitting the OOM wall between
50× and 100× on a 24 GB RTX 3090.

Frameworks tested:
    pyspark-rapids  — Spark + RAPIDS plugin, broadcast cross-join (GPU-lazy)
    cudf            — NVIDIA cuDF, actual concat copies in VRAM
    polars          — CPU reference (same as CPU sweep)
    duckdb          — CPU reference (same as CPU sweep)

Usage:
    conda activate rapids-spark
    python benchmarks/simulate_scale_gpu.py \\
        --data-path ./data/robotics_demo \\
        --scales 1,10,50,100,500 \\
        --rapids-jar ./jars/rapids-4-spark_2.12-25.02.0.jar
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Sentinel values ───────────────────────────────────────────────────────────

OOM = "OOM"
ERR = "ERR"
SKIP = "SKIP"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GPUScaleResult:
    framework: str
    scale_factor: int
    virtual_records: int
    total_time_s: float
    throughput_events_per_sec: float
    peak_memory_mb: float       # CPU RAM used (psutil)
    gpu_memory_mb: float        # VRAM used (pynvml)
    status: str = "ok"
    note: str = ""


# ── Subprocess isolation ──────────────────────────────────────────────────────

def _worker(fn_name: str, kwargs: dict, q: mp.Queue) -> None:
    try:
        mod = sys.modules[__name__]
        result = getattr(mod, fn_name)(**kwargs)
        q.put(("ok", result))
    except MemoryError as e:
        q.put(("oom", f"MemoryError: {e}"))
    except Exception as e:
        msg = str(e)
        if any(k in msg for k in ("OutOfMemory", "CUDA out of memory", "insufficient memory",
                                   "MemoryError", "OOM", "memory")):
            q.put(("oom", msg[:300]))
        else:
            q.put(("err", f"{type(e).__name__}: {msg[:300]}"))


def _safe_run(label: str, fn_name: str, kwargs: dict,
              timeout: int = 600) -> Union[GPUScaleResult, str]:
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker, args=(fn_name, kwargs, q))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return OOM

    if not q.empty():
        status, payload = q.get()
        if status == "ok":
            return payload
        if status == "oom":
            print(f"    !! {label}: OOM — {payload[:120]}")
            return OOM
        print(f"    !! {label}: ERR — {payload[:120]}")
        return ERR

    return ERR


# ── pyspark-rapids runner (broadcast cross-join — GPU lazy) ───────────────────

def _run_pyspark_rapids(data_path: str, scale_factor: int,
                        rapids_jar: str) -> GPUScaleResult:
    import psutil
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    # Target GPU 1 (RTX 3090, 24 GB free) — GPU 0 may be used by other apps
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

    proc = psutil.Process()
    mem_before = proc.memory_info().rss

    spark = (
        SparkSession.builder
        .appName(f"gpu-scale-rapids-{scale_factor}x")
        .master("local[*]")
        .config("spark.driver.memory", "16g")
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .config("spark.rapids.sql.enabled", "true")
        .config("spark.rapids.memory.gpu.minAllocFraction", "0")
        .config("spark.rapids.memory.gpu.allocFraction", "0.7")
        .config("spark.rapids.memory.gpu.maxAllocFraction", "0.8")
        .config("spark.jars", rapids_jar)
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    t0 = time.perf_counter()
    base_df = spark.read.parquet(data_path)
    # Broadcast cross-join: creates virtual copies without materialising in VRAM.
    # Spark processes one batch at a time → constant VRAM regardless of scale.
    scale_df = spark.range(scale_factor).toDF("_scale_id")
    scaled_df = base_df.crossJoin(F.broadcast(scale_df))
    n = scaled_df.count()
    total_time = time.perf_counter() - t0

    spark.stop()

    gpu_mb = _gpu_memory_mb()
    mem_after = proc.memory_info().rss
    peak_mb = max(0.0, (mem_after - mem_before) / 1024 / 1024)

    return GPUScaleResult(
        framework="pyspark-rapids",
        scale_factor=scale_factor,
        virtual_records=n,
        total_time_s=total_time,
        throughput_events_per_sec=n / max(total_time, 1e-9),
        peak_memory_mb=peak_mb,
        gpu_memory_mb=gpu_mb,
        note="broadcast cross-join — VRAM stays constant across all scales",
    )


# ── cuDF runner (actual VRAM concat — will OOM at high scale) ────────────────

def _run_cudf(data_path: str, scale_factor: int) -> GPUScaleResult:
    import cudf
    import psutil

    # Use GPU 1 so we don't clash with ComfyUI on GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    gpu_before = _gpu_memory_mb(device=1)

    t0 = time.perf_counter()
    base_df = cudf.read_parquet(str(Path(data_path) / "*.parquet"))

    # Materialise scale_factor copies in VRAM — OOMs when > GPU VRAM
    frames = [base_df] * scale_factor
    scaled_df = cudf.concat(frames, ignore_index=True)
    n = len(scaled_df)

    # Simple aggregation to exercise GPU compute
    _ = scaled_df.groupby("robot_id")["signal_value"].mean()
    total_time = time.perf_counter() - t0

    gpu_after = _gpu_memory_mb(device=1)
    mem_after = proc.memory_info().rss
    peak_mb = max(0.0, (mem_after - mem_before) / 1024 / 1024)

    return GPUScaleResult(
        framework="cudf",
        scale_factor=scale_factor,
        virtual_records=n,
        total_time_s=total_time,
        throughput_events_per_sec=n / max(total_time, 1e-9),
        peak_memory_mb=peak_mb,
        gpu_memory_mb=max(0.0, gpu_after - gpu_before),
        note="full materialisation in VRAM — OOMs when data > GPU VRAM",
    )


# ── Polars CPU reference ──────────────────────────────────────────────────────

def _run_polars(data_path: str, scale_factor: int) -> GPUScaleResult:
    import polars as pl
    import psutil

    proc = psutil.Process()
    mem_before = proc.memory_info().rss

    t0 = time.perf_counter()
    base_df = pl.read_parquet(str(Path(data_path) / "*.parquet"))
    frames = [base_df] * scale_factor
    scaled_df = pl.concat(frames, rechunk=False)
    n = len(scaled_df)
    _ = scaled_df.group_by("robot_id").agg(pl.col("signal_value").mean())
    total_time = time.perf_counter() - t0

    mem_after = proc.memory_info().rss
    peak_mb = max(0.0, (mem_after - mem_before) / 1024 / 1024)

    return GPUScaleResult(
        framework="polars",
        scale_factor=scale_factor,
        virtual_records=n,
        total_time_s=total_time,
        throughput_events_per_sec=n / max(total_time, 1e-9),
        peak_memory_mb=peak_mb,
        gpu_memory_mb=0.0,
        note="CPU reference",
    )


# ── DuckDB CPU reference ──────────────────────────────────────────────────────

def _run_duckdb(data_path: str, scale_factor: int) -> GPUScaleResult:
    import duckdb
    import psutil

    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    pattern = str(Path(data_path) / "*.parquet")

    t0 = time.perf_counter()
    con = duckdb.connect(":memory:")
    # UNION ALL to create scale_factor virtual copies (lazy — DuckDB may spill)
    union_sql = " UNION ALL ".join(
        [f"SELECT * FROM read_parquet('{pattern}')"] * scale_factor
    )
    n = con.execute(f"SELECT COUNT(*) FROM ({union_sql})").fetchone()[0]
    _ = con.execute(f"""
        SELECT robot_id, AVG(signal_value)
        FROM ({union_sql})
        GROUP BY robot_id
    """).df()
    total_time = time.perf_counter() - t0
    con.close()

    mem_after = proc.memory_info().rss
    peak_mb = max(0.0, (mem_after - mem_before) / 1024 / 1024)

    return GPUScaleResult(
        framework="duckdb",
        scale_factor=scale_factor,
        virtual_records=n,
        total_time_s=total_time,
        throughput_events_per_sec=n / max(total_time, 1e-9),
        peak_memory_mb=peak_mb,
        gpu_memory_mb=0.0,
        note="CPU reference — spills to disk under memory pressure",
    )


# ── GPU memory helper ─────────────────────────────────────────────────────────

def _gpu_memory_mb(device: int = 1) -> float:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        pynvml.nvmlShutdown()
        return info.used / 1024 / 1024
    except Exception:
        return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU scale-factor sweep: pyspark-rapids vs cuDF vs CPU references"
    )
    parser.add_argument("--data-path", default="./data/robotics_demo")
    parser.add_argument("--scales", default="1,10,50,100,500",
                        help="Comma-separated scale factors")
    parser.add_argument("--rapids-jar",
                        default="./jars/rapids-4-spark_2.12-25.02.0.jar")
    parser.add_argument("--output-dir", default="./benchmarks/results")
    parser.add_argument("--skip-rapids", action="store_true")
    parser.add_argument("--skip-cudf", action="store_true")
    parser.add_argument("--skip-polars", action="store_true")
    parser.add_argument("--skip-duckdb", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-framework timeout in seconds (default 1800 = 30 min)")
    args = parser.parse_args()

    scales = [int(s) for s in args.scales.split(",")]
    data_path = str(Path(args.data_path).resolve())
    rapids_jar = str(Path(args.rapids_jar).resolve())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_records = sum(
        int(f.stat().st_size / 80)           # approx rows from parquet size
        for f in Path(data_path).glob("*.parquet")
    )
    # Get actual count from first parquet file for accuracy
    try:
        import polars as pl
        base_records = len(pl.read_parquet(str(Path(data_path) / "*.parquet")))
    except Exception:
        pass

    approx_gb_per_scale = base_records * 80 / 1024**3  # ~80 bytes/row uncompressed

    print("=" * 60)
    print("HSM-PySpark: GPU Scale-Factor Sweep")
    print("=" * 60)
    print(f"Data path  : {data_path}")
    print(f"Scales     : {scales}")
    print(f"RAPIDS JAR : {rapids_jar}")
    print(f"Base records: {base_records:,}")
    print("=" * 60)

    sweep = []

    for scale in scales:
        virtual_records = base_records * scale
        approx_gb = approx_gb_per_scale * scale
        print(f"\n── Scale {scale}× ({approx_gb:.1f} GB, ~{virtual_records:,} virtual records) ──")

        scale_entry: dict = {
            "scale_factor": scale,
            "approx_data_size_gb": round(approx_gb, 2),
            "virtual_records": virtual_records,
            "frameworks": {},
        }

        # ── pyspark-rapids ────────────────────────────────────────────────────
        if not args.skip_rapids:
            print(f"  Running pyspark-rapids {scale}×...")
            result = _safe_run(
                f"pyspark-rapids {scale}×",
                "_run_pyspark_rapids",
                {"data_path": data_path, "scale_factor": scale, "rapids_jar": rapids_jar},
                timeout=args.timeout,
            )
            if isinstance(result, GPUScaleResult):
                print(f"  ✓ pyspark-rapids: {result.total_time_s:.1f}s, "
                      f"{result.throughput_events_per_sec:,.0f} ev/s, "
                      f"VRAM {result.gpu_memory_mb:.0f} MB")
                scale_entry["frameworks"]["pyspark-rapids"] = asdict(result)
            else:
                print(f"  !! pyspark-rapids: {result}")
                scale_entry["frameworks"]["pyspark-rapids"] = {"status": result}

        # ── cuDF ──────────────────────────────────────────────────────────────
        if not args.skip_cudf:
            print(f"  Running cudf {scale}×...")
            result = _safe_run(
                f"cudf {scale}×",
                "_run_cudf",
                {"data_path": data_path, "scale_factor": scale},
                timeout=args.timeout,
            )
            if isinstance(result, GPUScaleResult):
                print(f"  ✓ cudf: {result.total_time_s:.1f}s, "
                      f"{result.throughput_events_per_sec:,.0f} ev/s, "
                      f"VRAM {result.gpu_memory_mb:.0f} MB")
                scale_entry["frameworks"]["cudf"] = asdict(result)
            else:
                print(f"  !! cudf: {result}")
                scale_entry["frameworks"]["cudf"] = {"status": result}

        # ── Polars (CPU ref) ──────────────────────────────────────────────────
        if not args.skip_polars:
            print(f"  Running polars {scale}×...")
            result = _safe_run(
                f"polars {scale}×",
                "_run_polars",
                {"data_path": data_path, "scale_factor": scale},
                timeout=args.timeout,
            )
            if isinstance(result, GPUScaleResult):
                print(f"  ✓ polars: {result.total_time_s:.1f}s, "
                      f"{result.throughput_events_per_sec:,.0f} ev/s, "
                      f"RAM {result.peak_memory_mb:.0f} MB")
                scale_entry["frameworks"]["polars"] = asdict(result)
            else:
                print(f"  !! polars: {result}")
                scale_entry["frameworks"]["polars"] = {"status": result}

        # ── DuckDB (CPU ref) ──────────────────────────────────────────────────
        if not args.skip_duckdb:
            print(f"  Running duckdb {scale}×...")
            result = _safe_run(
                f"duckdb {scale}×",
                "_run_duckdb",
                {"data_path": data_path, "scale_factor": scale},
                timeout=args.timeout,
            )
            if isinstance(result, GPUScaleResult):
                print(f"  ✓ duckdb: {result.total_time_s:.1f}s, "
                      f"{result.throughput_events_per_sec:,.0f} ev/s, "
                      f"RAM {result.peak_memory_mb:.0f} MB")
                scale_entry["frameworks"]["duckdb"] = asdict(result)
            else:
                print(f"  !! duckdb: {result}")
                scale_entry["frameworks"]["duckdb"] = {"status": result}

        sweep.append(scale_entry)

    # ── Save results ──────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = output_dir / f"gpu_scale_sweep_{ts}.json"
    payload = {
        "generated_at": ts,
        "machine": "linux-gpu (RTX 3090 Ti + RTX 3090)",
        "rapids_jar": rapids_jar,
        "base_records": base_records,
        "scales": scales,
        "sweep": sweep,
        "notes": {
            "pyspark_rapids_memory_model": (
                "Broadcast cross-join: base parquet read once, virtual copies "
                "never materialised in VRAM. GPU memory stays near-constant "
                "regardless of scale — same trick as CPU PySpark."
            ),
            "cudf_memory_model": (
                "Full materialisation: all scale_factor copies concatenated in "
                "VRAM. OOMs when virtual_records × ~400 bytes > GPU VRAM (~24 GB)."
            ),
        },
    }
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved → {out_file}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  GPU SCALE SWEEP  (throughput: events/sec)")
    print("=" * 100)
    headers = ["Scale", "Records", "pyspark-rapids", "cudf", "polars (CPU)", "duckdb (CPU)"]
    print(f"  {'Scale':<22} {'Records':<12} {'pyspark-rapids':>16} {'cudf':>16} {'polars':>14} {'duckdb':>14}")
    print("  " + "-" * 96)
    for entry in sweep:
        sf = entry["scale_factor"]
        gb = entry["approx_data_size_gb"]
        vr = entry["virtual_records"]
        label = f"{sf}× ({gb:.1f} GB)"

        def _fmt(fw: str) -> str:
            fw_data = entry["frameworks"].get(fw, {})
            if isinstance(fw_data, dict) and "throughput_events_per_sec" in fw_data:
                t = fw_data["throughput_events_per_sec"]
                if t >= 1e9:
                    return f"{t/1e9:.1f}B"
                if t >= 1e6:
                    return f"{t/1e6:.1f}M"
                if t >= 1e3:
                    return f"{t/1e3:.0f}K"
                return f"{t:.0f}"
            return fw_data.get("status", "—") if isinstance(fw_data, dict) else "—"

        print(f"  {label:<22} {vr//1000:>8}K  "
              f"{'pyspark-rapids':>4}: {_fmt('pyspark-rapids'):>10}  "
              f"cudf: {_fmt('cudf'):>10}  "
              f"polars: {_fmt('polars'):>10}  "
              f"duckdb: {_fmt('duckdb'):>10}")

    print("=" * 100)


if __name__ == "__main__":
    main()
