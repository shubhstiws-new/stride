#!/usr/bin/env python3
"""
Results Merger — Combine Mac + GPU result JSONs into a unified matrix.

Reads all matrix_*.json files, merges on (hardware, framework, dataset, filesize, operation),
and prints a pivot table showing throughput winners per operation × filesize.

Usage:
    python benchmarks/collect_results.py \\
        --results-dir ./benchmarks/results \\
        --output ./benchmarks/results/matrix_combined.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


# ── Load helpers ─────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> list[dict]:
    """Load all matrix_*.json files, return flat list of result dicts."""
    json_files = sorted(results_dir.glob("matrix_*.json"))
    if not json_files:
        print(f"No matrix_*.json files found in {results_dir}")
        return []

    all_results: list[dict] = []
    for f in json_files:
        try:
            data = json.loads(f.read_text())
            file_results = data.get("results", [])
            print(f"  Loaded {len(file_results):>4} results from {f.name}")
            all_results.extend(file_results)
        except Exception as e:
            print(f"  WARNING: Could not parse {f.name}: {e}")

    return all_results


def deduplicate(results: list[dict]) -> list[dict]:
    """
    Deduplicate on (hardware, framework, dataset, filesize_target, operation).
    Keeps the latest result if duplicates exist (last wins).
    """
    seen: dict[tuple, dict] = {}
    for r in results:
        key = (
            r.get("hardware", ""),
            r.get("framework", ""),
            r.get("dataset", ""),
            r.get("filesize_target", ""),
            r.get("operation", ""),
        )
        seen[key] = r
    return list(seen.values())


# ── Pivot table ───────────────────────────────────────────────────────────────

FILESIZE_ORDER = ["1mb", "10mb", "100mb", "1gb"]
OPERATION_ORDER = ["io_scan", "filter_agg", "window_hsm", "join"]


def build_pivot(results: list[dict]) -> dict:
    """
    Build {operation: {filesize: {framework: throughput}}} from results.
    Only includes results with status == "ok".
    """
    pivot: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in results:
        if r.get("status") != "ok":
            continue
        op = r.get("operation", "")
        fs = r.get("filesize_target", "")
        fw = r.get("framework", "")
        tp = r.get("throughput_records_per_sec", 0.0)
        if op and fs and fw:
            # Keep best across hardware variants
            existing = pivot[op][fs].get(fw, 0.0)
            if tp > existing:
                pivot[op][fs][fw] = tp
    return pivot


def print_pivot_table(pivot: dict) -> None:
    """Print winner-per-cell pivot table."""
    col_w = 16

    def pad(s: str, w: int) -> str:
        return str(s)[:w].center(w)

    print()
    print("=" * 80)
    print("  THROUGHPUT WINNERS  (operation × file size)")
    print("=" * 80)

    ops = [o for o in OPERATION_ORDER if o in pivot]
    filesizes = [fs for fs in FILESIZE_ORDER if any(fs in pivot[op] for op in ops)]

    # Header
    hdr = ["File Size"] + ops
    print("  " + " | ".join(pad(h, col_w) for h in hdr))
    print("  " + "-" * (col_w * len(hdr) + 3 * (len(hdr) - 1) + 2))

    for fs in filesizes:
        row = [fs]
        for op in ops:
            fw_tp = pivot.get(op, {}).get(fs, {})
            if not fw_tp:
                row.append("-")
            else:
                winner = max(fw_tp, key=lambda fw: fw_tp[fw])
                tp = fw_tp[winner]
                if tp >= 1_000_000:
                    tp_str = f"{tp/1_000_000:.1f}M/s"
                elif tp >= 1_000:
                    tp_str = f"{tp/1_000:.0f}K/s"
                else:
                    tp_str = f"{tp:.0f}/s"
                row.append(f"{winner} {tp_str}")
        print("  " + " | ".join(pad(c, col_w) for c in row))

    print()


def print_full_comparison(results: list[dict]) -> None:
    """Print per-framework throughput for each (op, filesize) cell."""
    # Group by (operation, filesize)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        if r.get("status") != "ok":
            continue
        key = (r.get("operation", ""), r.get("filesize_target", ""))
        grouped[key].append(r)

    print()
    print("=" * 90)
    print("  FULL COMPARISON  (sorted by throughput within each cell)")
    print("=" * 90)

    for op in OPERATION_ORDER:
        for fs in FILESIZE_ORDER:
            cell = grouped.get((op, fs), [])
            if not cell:
                continue
            cell_sorted = sorted(
                cell, key=lambda r: r.get("throughput_records_per_sec", 0), reverse=True
            )
            print(f"\n  [{op} / {fs}]")
            for r in cell_sorted:
                fw = r.get("framework", "?")
                hw = r.get("hardware", "?")
                tp = r.get("throughput_records_per_sec", 0)
                compute_t = r.get("compute_time_s", 0)
                mem_mb = r.get("peak_memory_mb", 0)
                gpu_mb = r.get("gpu_memory_mb", 0)
                n = r.get("total_records", 0)
                ds = r.get("dataset", "?")

                if tp >= 1_000_000:
                    tp_str = f"{tp/1_000_000:.1f}M rec/s"
                elif tp >= 1_000:
                    tp_str = f"{tp/1_000:.0f}K rec/s"
                else:
                    tp_str = f"{tp:.0f} rec/s"

                gpu_str = f"GPU {gpu_mb:.0f}MB" if gpu_mb > 0 else ""
                print(
                    f"    {fw:<18} {hw:<12} {ds:<16} "
                    f"{compute_t:>7.2f}s  {tp_str:>12}  "
                    f"mem {mem_mb:>5.0f}MB  {gpu_str}"
                )

    print()


# ── Save combined ─────────────────────────────────────────────────────────────

def save_combined(results: list[dict], output_path: Path) -> None:
    from datetime import datetime, timezone

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_results": len(results),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"  Combined results saved → {output_path}")


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_summary_stats(results: list[dict]) -> None:
    ok = [r for r in results if r.get("status") == "ok"]
    oom = [r for r in results if r.get("status") == "oom"]
    err = [r for r in results if r.get("status") == "err"]
    skip = [r for r in results if r.get("status") == "skip"]

    hardwares = sorted({r.get("hardware", "?") for r in results})
    frameworks = sorted({r.get("framework", "?") for r in results})
    datasets = sorted({r.get("dataset", "?") for r in results})

    print()
    print("=" * 60)
    print("  RESULT SUMMARY")
    print("=" * 60)
    print(f"  Total results : {len(results)}")
    print(f"  OK            : {len(ok)}")
    print(f"  OOM           : {len(oom)}")
    print(f"  Error         : {len(err)}")
    print(f"  Skipped       : {len(skip)}")
    print(f"  Hardware      : {hardwares}")
    print(f"  Frameworks    : {frameworks}")
    print(f"  Datasets      : {datasets}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Mac + GPU benchmark results into a combined matrix"
    )
    parser.add_argument(
        "--results-dir",
        default="./benchmarks/results",
        help="Directory containing matrix_*.json files (default: ./benchmarks/results)",
    )
    parser.add_argument(
        "--output",
        default="./benchmarks/results/matrix_combined.json",
        help="Output path for combined JSON (default: ./benchmarks/results/matrix_combined.json)",
    )
    parser.add_argument(
        "--no-pivot",
        action="store_true",
        help="Skip printing the winner pivot table",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also print full per-framework comparison table",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_path = Path(args.output).resolve()

    print("\n" + "=" * 60)
    print("HSM-PySpark: Results Merger")
    print("=" * 60)
    print(f"Results dir: {results_dir}")
    print(f"Output     : {output_path}")
    print()

    # Load all results
    all_results = load_results(results_dir)
    if not all_results:
        print("No results to merge.")
        sys.exit(1)

    # Deduplicate
    merged = deduplicate(all_results)
    print(f"\n  {len(all_results)} total records → {len(merged)} after deduplication")

    # Print summary
    print_summary_stats(merged)

    # Pivot table
    if not args.no_pivot:
        pivot = build_pivot(merged)
        print_pivot_table(pivot)

    # Full comparison
    if args.full:
        print_full_comparison(merged)

    # Save
    save_combined(merged, output_path)

    print()


if __name__ == "__main__":
    main()
