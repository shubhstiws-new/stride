#!/usr/bin/env python3
"""
MinIO uploader — sync local Parquet benchmark datasets to MinIO (S3-compatible storage).

Walks data_dir looking for directories matching:
    {dataset}/filesize={size_label}/*.parquet

Uploads each Parquet file to:
    s3://{bucket}/{prefix}/{dataset}/filesize={size_label}/{filename}

Prerequisites:
    - MinIO running (e.g. via docker-compose up minio)
    - pip install boto3  (already in benchmark extras)

Usage:
    python benchmarks/datasets/minio_uploader.py \\
        --data-dir ./data \\
        --endpoint http://localhost:9000 \\
        --bucket robotics-bench \\
        --prefix benchmarks

    # Dry run (just print what would be uploaded):
    python benchmarks/datasets/minio_uploader.py --data-dir ./data --dry-run

    # Upload only specific datasets / file sizes:
    python benchmarks/datasets/minio_uploader.py \\
        --data-dir ./data \\
        --datasets behavior1k,droid \\
        --filesizes 1mb,10mb
"""

import argparse
import sys
import time
from pathlib import Path


def _get_s3_client(endpoint: str, access_key: str, secret_key: str):
    """Build and return a boto3 S3 client configured for MinIO."""
    import boto3
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(s3_client, bucket: str) -> bool:
    """Create bucket if it doesn't exist. Returns True if created."""
    try:
        s3_client.head_bucket(Bucket=bucket)
        return False
    except Exception:
        s3_client.create_bucket(Bucket=bucket)
        print(f"  Created bucket: {bucket}")
        return True


def upload_partition_dir(
    s3_client,
    local_dir: Path,
    data_root: Path,
    bucket: str,
    prefix: str,
    dry_run: bool = False,
) -> tuple[int, int, float]:
    """
    Upload all *.parquet files in local_dir to MinIO.

    S3 key = {prefix}/{dataset}/filesize={size}/{filename}
    (relative path from data_root)

    Returns (files_uploaded, bytes_uploaded, elapsed_s).
    """
    parquet_files = sorted(local_dir.glob("*.parquet"))
    if not parquet_files:
        return 0, 0, 0.0

    t0 = time.perf_counter()
    uploaded_files = 0
    uploaded_bytes = 0

    for pf in parquet_files:
        # Build S3 key relative to data_root
        rel = pf.relative_to(data_root)
        s3_key = f"{prefix}/{rel}"
        file_size = pf.stat().st_size

        if dry_run:
            print(f"    [DRY RUN] s3://{bucket}/{s3_key}  ({file_size/1024:.0f} KB)")
        else:
            s3_client.upload_file(str(pf), bucket, s3_key)
            uploaded_bytes += file_size
            uploaded_files += 1

    elapsed = time.perf_counter() - t0
    return uploaded_files, uploaded_bytes, elapsed


def upload_dataset(
    dataset_dir: Path,
    data_root: Path,
    s3_client,
    bucket: str,
    prefix: str,
    filesize_filter: set[str] | None,
    dry_run: bool,
) -> tuple[int, int]:
    """Upload all filesize partitions for one dataset. Returns (total_files, total_bytes)."""
    total_files = 0
    total_bytes = 0

    size_dirs = sorted(dataset_dir.glob("filesize=*"))
    if not size_dirs:
        print(f"  No filesize= partitions found in {dataset_dir}")
        return 0, 0

    for size_dir in size_dirs:
        size_label = size_dir.name.replace("filesize=", "")
        if filesize_filter and size_label not in filesize_filter:
            continue

        parquet_count = len(list(size_dir.glob("*.parquet")))
        if parquet_count == 0:
            continue

        print(f"  {size_dir.name}: {parquet_count} file(s)...", end="", flush=True)
        n_files, n_bytes, elapsed = upload_partition_dir(
            s3_client, size_dir, data_root, bucket, prefix, dry_run
        )

        if dry_run:
            print()
        else:
            total_files += n_files
            total_bytes += n_bytes
            mb = n_bytes / 1024 / 1024
            throughput = mb / max(elapsed, 0.001)
            print(f"  {n_files} files, {mb:.1f} MB in {elapsed:.1f}s ({throughput:.0f} MB/s)")

    return total_files, total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload benchmark Parquet datasets to MinIO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Local data root directory (contains {dataset}/filesize=*/*.parquet)",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:9000",
        help="MinIO server endpoint URL",
    )
    parser.add_argument("--bucket", default="robotics-bench", help="S3 bucket name")
    parser.add_argument("--prefix", default="benchmarks", help="S3 key prefix")
    parser.add_argument("--access-key", default="minioadmin", help="MinIO access key")
    parser.add_argument("--secret-key", default="minioadmin", help="MinIO secret key")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset names to upload (default: all found)",
    )
    parser.add_argument(
        "--filesizes",
        default=None,
        help="Comma-separated file size labels to upload (default: all found)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"ERROR: --data-dir '{data_dir}' does not exist")
        sys.exit(1)

    dataset_filter = set(args.datasets.split(",")) if args.datasets else None
    filesize_filter = set(args.filesizes.split(",")) if args.filesizes else None

    if args.dry_run:
        print("DRY RUN — no files will be uploaded\n")
        s3_client = None
    else:
        print(f"Connecting to MinIO at {args.endpoint}...")
        try:
            s3_client = _get_s3_client(args.endpoint, args.access_key, args.secret_key)
            ensure_bucket(s3_client, args.bucket)
        except Exception as e:
            print(f"ERROR: Cannot connect to MinIO at {args.endpoint}: {e}")
            print("Is MinIO running? Try: docker-compose up -d minio")
            sys.exit(1)

    print(f"\nUploading to s3://{args.bucket}/{args.prefix}/")
    print(f"Source: {data_dir}\n")

    grand_total_files = 0
    grand_total_bytes = 0
    t_wall = time.perf_counter()

    dataset_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "results"
    )

    for dataset_dir in dataset_dirs:
        if dataset_filter and dataset_dir.name not in dataset_filter:
            continue

        print(f"Dataset: {dataset_dir.name}")
        n_files, n_bytes = upload_dataset(
            dataset_dir=dataset_dir,
            data_root=data_dir,
            s3_client=s3_client,
            bucket=args.bucket,
            prefix=args.prefix,
            filesize_filter=filesize_filter,
            dry_run=args.dry_run,
        )
        grand_total_files += n_files
        grand_total_bytes += n_bytes

    elapsed_total = time.perf_counter() - t_wall
    if not args.dry_run:
        print(
            f"\nDone: {grand_total_files} files, "
            f"{grand_total_bytes/1024/1024:.1f} MB in {elapsed_total:.1f}s"
        )
        if grand_total_files > 0:
            print(
                f"MinIO browser: {args.endpoint.rstrip('/')}/browser/{args.bucket}"
                .replace("http://", "http://")
                .replace(":9000", ":9001")  # console port
            )


if __name__ == "__main__":
    main()
