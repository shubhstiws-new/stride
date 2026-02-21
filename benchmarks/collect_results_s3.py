#!/usr/bin/env python3
"""
benchmarks/collect_results_s3.py — Collect + pivot per-cell JSON results from MinIO.

Called by the final Airflow task in both benchmark_matrix_dag and gpu_scale_dag.
Reads all s3://robotics-bench/results/{run_id}/*.json files, merges them, and
prints a pivot table to stdout (captured in Airflow logs).

Usage:
    python3 benchmarks/collect_results_s3.py \\
        --run-id dag_run_2026_02_21 \\
        --minio-endpoint http://minio.hsm-bench.svc:9000 \\
        --bucket robotics-bench
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_results_from_s3(run_id: str, endpoint: str, bucket: str) -> list[dict]:
    import boto3, os

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        region_name="us-east-1",
    )
    prefix = f"results/{run_id}/"
    paginator = s3.get_paginator("list_objects_v2")
    results = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                try:
                    results.append(json.loads(body))
                except json.JSONDecodeError as e:
                    print(f"  WARN: could not parse {obj['Key']}: {e}")
    return results


FILESIZE_ORDER = ["1mb", "10mb", "100mb", "1gb"]
OPERATION_ORDER = ["io_scan", "filter_agg", "window_hsm", "join_op"]


def build_pivot(results: list[dict]) -> dict:
    pivot: dict = defaultdict(lambda: defaultdict(dict))
    for r in results:
        if r.get("status") != "ok":
            continue
        op, fs, fw = r.get("operation", ""), r.get("filesize_target", ""), r.get("framework", "")
        tp = r.get("throughput_records_per_sec", 0.0)
        if op and fs and fw:
            existing = pivot[op][fs].get(fw, 0.0)
            if tp > existing:
                pivot[op][fs][fw] = tp
    return pivot


def print_pivot(pivot: dict) -> None:
    col_w = 18
    pad = lambda s, w: str(s)[:w].center(w)

    ops = [o for o in OPERATION_ORDER if o in pivot]
    filesizes = [f for f in FILESIZE_ORDER if any(f in pivot[o] for o in ops)]

    print()
    print("=" * 80)
    print("  THROUGHPUT WINNERS  (operation × file size)")
    print("=" * 80)
    print("  " + " | ".join(pad(h, col_w) for h in ["File Size"] + ops))
    print("  " + "-" * (col_w * (len(ops) + 1) + 3 * len(ops) + 2))

    for fs in filesizes:
        row = [fs]
        for op in ops:
            fw_tp = pivot.get(op, {}).get(fs, {})
            if not fw_tp:
                row.append("-")
            else:
                winner = max(fw_tp, key=lambda fw: fw_tp[fw])
                tp = fw_tp[winner]
                tp_str = (
                    f"{tp/1_000_000:.1f}M/s" if tp >= 1_000_000
                    else f"{tp/1_000:.0f}K/s" if tp >= 1_000
                    else f"{tp:.0f}/s"
                )
                row.append(f"{winner} {tp_str}")
        print("  " + " | ".join(pad(c, col_w) for c in row))
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--minio-endpoint", default="http://minio.hsm-bench.svc:9000")
    parser.add_argument("--bucket", default="robotics-bench")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  HSM-PySpark: Collect Results")
    print(f"  Run ID  : {args.run_id}")
    print(f"  MinIO   : {args.minio_endpoint}")
    print(f"{'='*60}\n")

    results = load_results_from_s3(args.run_id, args.minio_endpoint, args.bucket)
    print(f"  Loaded {len(results)} result(s)")

    if not results:
        print("  No results found — check that benchmark pods completed.")
        sys.exit(0)

    ok   = sum(1 for r in results if r.get("status") == "ok")
    oom  = sum(1 for r in results if r.get("status") == "oom")
    err  = sum(1 for r in results if r.get("status") == "err")
    skip = sum(1 for r in results if r.get("status") == "skip")
    print(f"  OK={ok}  OOM={oom}  ERR={err}  SKIP={skip}")

    pivot = build_pivot(results)
    print_pivot(pivot)

    print("  Done.")


if __name__ == "__main__":
    main()
