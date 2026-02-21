#!/usr/bin/env python3
"""
Dataset Manager — Download, normalize, and repartition robotics datasets.

Downloads from HuggingFace Hub, normalizes to unified long-form schema,
then writes size-targeted Parquet partitions for benchmark use.

Usage:
    python benchmarks/datasets/dataset_manager.py \\
        --datasets behavior1k,droid,oxe,nvidia_physicalai \\
        --episodes 500 \\
        --filesizes 1mb,10mb,100mb,1gb \\
        --output-dir ./data
"""

import argparse
import math
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl

from benchmarks.datasets.schema import DATASET_REGISTRY, UNIFIED_COLUMNS

# ── File size targets (bytes) ──────────────────────────────────────────────────

FILE_SIZE_TARGETS: dict[str, int] = {
    "1mb":   1 * 1024 * 1024,
    "10mb":  10 * 1024 * 1024,
    "100mb": 100 * 1024 * 1024,
    "1gb":   1024 * 1024 * 1024,
}

# Approximate bytes per row in compressed Parquet (empirical estimate)
_BYTES_PER_ROW_PARQUET = 80


# ── Download helpers ──────────────────────────────────────────────────────────

def _download_dataset(
    dataset_name: str,
    hf_repo_id: str,
    n_episodes: int,
    cache_dir: Path,
) -> Path:
    """Download Parquet files from HuggingFace Hub, return local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )

    local_dir = cache_dir / dataset_name / "raw"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {dataset_name} from {hf_repo_id}...")
    print(f"  (targeting first ~{n_episodes} episodes, Parquet only)")

    try:
        snapshot_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=["*.parquet", "*.json", "*.jsonl"],
            ignore_patterns=["*.mp4", "*.avi", "*.mov", "*.jpg", "*.png"],
        )
    except Exception as e:
        print(f"  WARNING: Download failed or partial: {e}")
        print(f"  Checking for existing files in {local_dir}...")

    parquet_files = list(local_dir.rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(
            f"No Parquet files found in {local_dir}. "
            f"Download may have failed for {hf_repo_id}.\n"
            f"Check the HF repo ID and your network/auth."
        )

    print(f"  Found {len(parquet_files)} Parquet file(s)")
    return local_dir


def _read_and_subsample(raw_dir: Path, n_episodes: int) -> pl.DataFrame:
    """Read raw Parquet files and subsample to first n_episodes episodes."""
    parquet_files = sorted(raw_dir.rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No Parquet files in {raw_dir}")

    # Read incrementally until we have enough episodes
    frames = []
    episodes_seen: set[str] = set()

    for pf in parquet_files:
        chunk = pl.read_parquet(str(pf))
        if chunk.is_empty():
            continue

        # Detect episode column
        ep_col = next(
            (c for c in ["episode_id", "episode_index", "episode"] if c in chunk.columns),
            None,
        )
        if ep_col:
            available_episodes = chunk[ep_col].unique().to_list()
            needed = [e for e in available_episodes if str(e) not in episodes_seen]
            if not needed:
                continue
            take = needed[: max(0, n_episodes - len(episodes_seen))]
            chunk = chunk.filter(pl.col(ep_col).is_in(take))
            episodes_seen.update(str(e) for e in take)
        frames.append(chunk)

        if len(episodes_seen) >= n_episodes:
            break

    if not frames:
        raise RuntimeError("No data loaded after filtering")

    return pl.concat(frames, how="diagonal_relaxed")


# ── Repartition helpers ───────────────────────────────────────────────────────

def _write_filesize_partition(
    df: pl.DataFrame,
    out_dir: Path,
    target_bytes: int,
    size_label: str,
) -> None:
    """
    Write Parquet files sized to approximately target_bytes each.

    Adjusts number of row groups so each file ≈ target_bytes.
    Actual file sizes are verified; if off by >50%, logs a warning.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = len(df)

    # Estimate rows per partition
    rows_per_file = max(1000, target_bytes // _BYTES_PER_ROW_PARQUET)
    n_partitions = max(1, math.ceil(total_rows / rows_per_file))

    print(f"    Writing {size_label}: ~{rows_per_file:,} rows/file, {n_partitions} file(s)...")

    for i in range(n_partitions):
        start = i * rows_per_file
        end = min((i + 1) * rows_per_file, total_rows)
        chunk = df.slice(start, end - start)
        out_file = out_dir / f"part_{i:04d}.parquet"
        chunk.write_parquet(str(out_file), compression="snappy")

    # Verify sizes
    written_files = list(out_dir.glob("*.parquet"))
    if written_files:
        actual_sizes = [f.stat().st_size for f in written_files]
        avg_actual = sum(actual_sizes) / len(actual_sizes)
        ratio = avg_actual / target_bytes
        if ratio < 0.3 or ratio > 3.0:
            print(
                f"    WARNING: actual avg file size {avg_actual/1024:.0f} KB, "
                f"target {target_bytes/1024:.0f} KB (ratio {ratio:.1f}×). "
                "Consider adjusting _BYTES_PER_ROW_PARQUET."
            )
        else:
            print(
                f"    OK: {len(written_files)} file(s), "
                f"avg {avg_actual/1024:.0f} KB/file "
                f"(target {target_bytes/1024:.0f} KB)"
            )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_dataset(
    dataset_name: str,
    n_episodes: int,
    filesizes: list[str],
    output_dir: Path,
    cache_dir: Path,
    minio_endpoint: str = "",
    minio_bucket: str = "robotics-bench",
    minio_prefix: str = "benchmarks",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin",
    skip_upload: bool = False,
) -> None:
    hf_repo_id, normalize_fn = DATASET_REGISTRY[dataset_name]

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}  (HF: {hf_repo_id})")
    print(f"{'=' * 60}")

    # Step 1: Download
    t0 = time.perf_counter()
    raw_dir = _download_dataset(dataset_name, hf_repo_id, n_episodes, cache_dir)
    print(f"  Download: {time.perf_counter() - t0:.1f}s")

    # Step 2: Read + subsample
    t1 = time.perf_counter()
    raw_df = _read_and_subsample(raw_dir, n_episodes)
    print(f"  Raw rows: {len(raw_df):,}  ({time.perf_counter() - t1:.1f}s)")

    # Step 3: Normalize
    t2 = time.perf_counter()
    try:
        normalized = normalize_fn(raw_df)
    except Exception as e:
        print(f"  ERROR normalizing {dataset_name}: {e}")
        raise
    print(f"  Normalized rows: {len(normalized):,}  ({time.perf_counter() - t2:.1f}s)")
    print(f"  Columns: {normalized.columns}")

    # Step 4: Write normalized
    norm_dir = output_dir / dataset_name / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)
    normalized.write_parquet(str(norm_dir / "data.parquet"), compression="snappy")
    print(f"  Saved normalized → {norm_dir}/data.parquet")

    # Step 5: Write per-filesize partitions
    print("  Partitioning by file size...")
    for size_label in filesizes:
        if size_label not in FILE_SIZE_TARGETS:
            print(f"  WARNING: unknown file size '{size_label}', skipping")
            continue
        target_bytes = FILE_SIZE_TARGETS[size_label]
        size_dir = output_dir / dataset_name / f"filesize={size_label}"
        _write_filesize_partition(normalized, size_dir, target_bytes, size_label)

    # Step 6: Upload to MinIO (optional)
    if minio_endpoint and not skip_upload:
        print(f"\n  Uploading to MinIO ({minio_endpoint})...")
        try:
            from benchmarks.datasets.minio_uploader import (
                _get_s3_client,
                ensure_bucket,
                upload_partition_dir,
            )

            s3_client = _get_s3_client(minio_endpoint, minio_access_key, minio_secret_key)
            ensure_bucket(s3_client, minio_bucket)

            for size_label in filesizes:
                size_dir = output_dir / dataset_name / f"filesize={size_label}"
                if not size_dir.exists():
                    continue
                n_files, n_bytes, elapsed = upload_partition_dir(
                    s3_client, size_dir, output_dir, minio_bucket, minio_prefix
                )
                mb = n_bytes / 1024 / 1024
                print(f"    {size_label}: {n_files} files, {mb:.1f} MB in {elapsed:.1f}s")
        except Exception as e:
            print(f"  WARNING: MinIO upload failed: {e}")
            print("  Local files are still intact; re-run with --minio-endpoint to retry.")

    print(f"\n  Done: {dataset_name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and normalize robotics datasets for benchmarking"
    )
    parser.add_argument(
        "--datasets",
        default="behavior1k,droid,oxe,nvidia_physicalai",
        help="Comma-separated dataset names (default: all four)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of episodes to download per dataset (default: 500)",
    )
    parser.add_argument(
        "--filesizes",
        default="1mb,10mb,100mb,1gb",
        help="Comma-separated file size targets (default: 1mb,10mb,100mb,1gb)",
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Root output directory (default: ./data)",
    )
    parser.add_argument(
        "--cache-dir",
        default="./data/.cache",
        help="Directory for raw HF downloads (default: ./data/.cache)",
    )
    # MinIO upload (optional — skipped if --minio-endpoint not provided)
    parser.add_argument(
        "--minio-endpoint",
        default="",
        help="MinIO endpoint URL (e.g. http://localhost:9000). If set, uploads after partitioning.",
    )
    parser.add_argument("--minio-bucket", default="robotics-bench", help="MinIO S3 bucket name")
    parser.add_argument("--minio-prefix", default="benchmarks", help="MinIO S3 key prefix")
    parser.add_argument("--minio-access-key", default="minioadmin", help="MinIO access key")
    parser.add_argument("--minio-secret-key", default="minioadmin", help="MinIO secret key")
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip MinIO upload even if --minio-endpoint is set",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    filesizes = [f.strip() for f in args.filesizes.split(",")]
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()

    unknown = [d for d in datasets if d not in DATASET_REGISTRY]
    if unknown:
        print(f"ERROR: Unknown dataset(s): {unknown}")
        print(f"Available: {list(DATASET_REGISTRY.keys())}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("HSM-PySpark: Dataset Manager")
    print("=" * 60)
    print(f"Datasets  : {datasets}")
    print(f"Episodes  : {args.episodes}")
    print(f"File sizes: {filesizes}")
    print(f"Output    : {output_dir}")

    for dataset_name in datasets:
        try:
            process_dataset(
                dataset_name=dataset_name,
                n_episodes=args.episodes,
                filesizes=filesizes,
                output_dir=output_dir,
                cache_dir=cache_dir,
                minio_endpoint=args.minio_endpoint,
                minio_bucket=args.minio_bucket,
                minio_prefix=args.minio_prefix,
                minio_access_key=args.minio_access_key,
                minio_secret_key=args.minio_secret_key,
                skip_upload=args.skip_upload,
            )
        except Exception as e:
            print(f"\nERROR processing {dataset_name}: {e}")
            print("Continuing with next dataset...")

    print("\n" + "=" * 60)
    print("Dataset Manager complete.")
    print(f"Data written to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
