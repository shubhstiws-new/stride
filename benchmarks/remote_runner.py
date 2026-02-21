#!/usr/bin/env python3
"""
Remote Runner — SSH to GPU machine, run matrix benchmark, pull results.

Steps:
1. rsync local benchmarks/ and data/ to GPU machine
2. SSH: run matrix_benchmark.py with linux-gpu config
3. Stream stdout to local terminal in real time via paramiko
4. On completion: scp all matrix_linux-gpu_*.json back locally

Usage:
    python benchmarks/remote_runner.py \\
        --host 192.168.x.x \\
        --user shubh \\
        --key ~/.ssh/id_rsa \\
        --remote-dir ~/hsm-pyspark \\
        --datasets behavior1k,droid,oxe,nvidia_physicalai \\
        --filesizes 1mb,10mb,100mb,1gb \\
        --pull-results ./benchmarks/results
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ── rsync helper ──────────────────────────────────────────────────────────────

def rsync_to_remote(
    host: str,
    user: str,
    key: str,
    local_dir: str,
    remote_dir: str,
    extra_dirs: list[str] | None = None,
) -> None:
    """rsync local project files to the GPU machine."""
    ssh_opts = f"-e 'ssh -i {key} -o StrictHostKeyChecking=no'"
    remote_target = f"{user}@{host}:{remote_dir}"

    dirs_to_sync = [local_dir] + (extra_dirs or [])

    for local_path in dirs_to_sync:
        cmd = [
            "rsync", "-avz", "--progress",
            "-e", f"ssh -i {key} -o StrictHostKeyChecking=no",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=.git",
            "--exclude=*.egg-info",
            local_path + "/",
            f"{user}@{host}:{remote_dir}/",
        ]
        print(f"  rsync {local_path}/ → {remote_target}/")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"rsync failed for {local_path}")


# ── SSH stream helper ─────────────────────────────────────────────────────────

def ssh_run_streaming(
    host: str,
    user: str,
    key: str,
    command: str,
    timeout_s: int = 7200,
) -> int:
    """
    Run a command on a remote host via SSH, streaming stdout/stderr locally.

    Returns the remote exit code.
    """
    try:
        import paramiko
    except ImportError:
        raise RuntimeError(
            "paramiko not installed. Run: pip install paramiko"
        )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, key_filename=key, timeout=30)

    print(f"\n  [SSH] {user}@{host}: {command[:80]}...")
    transport = client.get_transport()
    channel = transport.open_session()
    channel.get_pty()
    channel.exec_command(command)

    t_start = time.perf_counter()
    while True:
        # Stream output
        if channel.recv_ready():
            data = channel.recv(4096)
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

        if channel.recv_stderr_ready():
            data = channel.recv_stderr(4096)
            sys.stderr.buffer.write(data)
            sys.stderr.buffer.flush()

        if channel.exit_status_ready():
            # Drain remaining output
            while channel.recv_ready():
                sys.stdout.buffer.write(channel.recv(4096))
                sys.stdout.buffer.flush()
            break

        if time.perf_counter() - t_start > timeout_s:
            print(f"\n  [SSH] TIMEOUT after {timeout_s}s")
            channel.close()
            client.close()
            return 1

        time.sleep(0.05)

    exit_code = channel.recv_exit_status()
    client.close()
    return exit_code


# ── scp pull helper ───────────────────────────────────────────────────────────

def scp_pull_results(
    host: str,
    user: str,
    key: str,
    remote_results_dir: str,
    local_results_dir: str,
    pattern: str = "matrix_linux-gpu_*.json",
) -> list[str]:
    """Download result JSON files from the GPU machine."""
    local_dir = Path(local_results_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_src = f"{user}@{host}:{remote_results_dir}/{pattern}"
    cmd = [
        "scp",
        "-i", key,
        "-o", "StrictHostKeyChecking=no",
        remote_src,
        str(local_dir) + "/",
    ]
    print(f"\n  scp {remote_src} → {local_dir}/")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  scp warning: {result.stderr.strip()}")
        return []

    # List what was pulled
    pulled = list(local_dir.glob(pattern.replace("*", "*")))
    return [str(f) for f in sorted(pulled)]


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remote GPU benchmark runner — SSH, run, pull results"
    )
    parser.add_argument("--host", required=True, help="GPU machine IP or hostname")
    parser.add_argument("--user", default="shubh", help="SSH username (default: shubh)")
    parser.add_argument(
        "--key",
        default=str(Path.home() / ".ssh" / "id_rsa"),
        help="SSH private key path (default: ~/.ssh/id_rsa)",
    )
    parser.add_argument(
        "--remote-dir",
        default="~/hsm-pyspark",
        help="Remote repo root directory (default: ~/hsm-pyspark)",
    )
    parser.add_argument(
        "--conda-env",
        default="rapids-spark",
        help="Conda environment on remote machine (default: rapids-spark)",
    )
    parser.add_argument(
        "--datasets",
        default="behavior1k,droid,oxe,nvidia_physicalai",
        help="Datasets to benchmark",
    )
    parser.add_argument(
        "--filesizes",
        default="1mb,10mb,100mb,1gb",
        help="File sizes to benchmark",
    )
    parser.add_argument(
        "--operations",
        default="io_scan,filter_agg,window_hsm,join",
        help="Operations to benchmark",
    )
    parser.add_argument(
        "--frameworks",
        default="pyspark-rapids,cudf,pyspark,polars,duckdb",
        help="Frameworks to benchmark on GPU machine",
    )
    parser.add_argument(
        "--rapids-jar",
        default="~/hsm-pyspark/jars/rapids-4-spark_2.12-25.02.0.jar",
        help="Path to RAPIDS JAR on remote machine",
    )
    parser.add_argument(
        "--pull-results",
        default="./benchmarks/results",
        help="Local directory to pull result JSONs into",
    )
    parser.add_argument(
        "--sync-data",
        action="store_true",
        help="Also rsync ./data/ to remote before running (slow for large datasets)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="SSH session timeout in seconds (default: 7200 = 2h)",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip rsync step (assume remote already has latest code)",
    )
    args = parser.parse_args()

    local_root = str(Path(__file__).parent.parent.resolve())

    print("\n" + "=" * 60)
    print("HSM-PySpark: Remote GPU Benchmark Runner")
    print("=" * 60)
    print(f"Host       : {args.user}@{args.host}")
    print(f"Remote dir : {args.remote_dir}")
    print(f"Conda env  : {args.conda_env}")
    print(f"Datasets   : {args.datasets}")
    print(f"File sizes : {args.filesizes}")
    print(f"Frameworks : {args.frameworks}")
    print("=" * 60)

    # Step 1: rsync code to remote
    if not args.skip_sync:
        print("\n[1/4] Syncing code to remote...")
        sync_dirs = [local_root + "/benchmarks", local_root + "/hsm_pyspark", local_root + "/setup"]
        if args.sync_data:
            sync_dirs.append(local_root + "/data")
        for local_dir in sync_dirs:
            if Path(local_dir).exists():
                cmd = [
                    "rsync", "-avz", "--progress",
                    "-e", f"ssh -i {args.key} -o StrictHostKeyChecking=no",
                    "--exclude=__pycache__",
                    "--exclude=*.pyc",
                    local_dir + "/",
                    f"{args.user}@{args.host}:{args.remote_dir}/{Path(local_dir).name}/",
                ]
                print(f"  rsync {Path(local_dir).name}/...")
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"  WARNING: rsync failed for {local_dir}, continuing...")
    else:
        print("\n[1/4] Sync skipped (--skip-sync)")

    # Step 2: Run hardware check
    print("\n[2/4] Running hardware preflight on remote...")
    preflight_cmd = (
        f"cd {args.remote_dir} && "
        f"conda run -n {args.conda_env} python setup/hardware_check.py"
    )
    exit_code = ssh_run_streaming(args.host, args.user, args.key, preflight_cmd, timeout_s=60)
    if exit_code != 0:
        print("  WARNING: Hardware check failed. Check GPU setup before proceeding.")
        print("  Continuing anyway...")

    # Step 3: Run matrix benchmark on remote
    print("\n[3/4] Running matrix benchmark on remote GPU machine...")
    benchmark_cmd = (
        f"cd {args.remote_dir} && "
        f"conda run -n {args.conda_env} python benchmarks/matrix_benchmark.py "
        f"--hardware linux-gpu "
        f"--datasets {args.datasets} "
        f"--filesizes {args.filesizes} "
        f"--operations {args.operations} "
        f"--frameworks {args.frameworks} "
        f"--rapids-jar {args.rapids_jar} "
        f"--output-dir {args.remote_dir}/benchmarks/results "
        f"--data-dir {args.remote_dir}/data"
    )
    exit_code = ssh_run_streaming(args.host, args.user, args.key, benchmark_cmd, timeout_s=args.timeout)
    if exit_code != 0:
        print(f"  WARNING: Benchmark exited with code {exit_code}")
    else:
        print("  Benchmark completed successfully.")

    # Step 4: Pull results
    print("\n[4/4] Pulling result files from remote...")
    remote_results_dir = f"{args.remote_dir}/benchmarks/results"
    pulled = scp_pull_results(
        host=args.host,
        user=args.user,
        key=args.key,
        remote_results_dir=remote_results_dir,
        local_results_dir=args.pull_results,
        pattern="matrix_linux-gpu_*.json",
    )

    print(f"\n{'=' * 60}")
    if pulled:
        print(f"  Pulled {len(pulled)} result file(s):")
        for f in pulled:
            print(f"    {f}")
    else:
        print("  No result files found (check remote benchmarks/results/ directory)")
    print(f"\n  To merge with local results:")
    print(f"    python benchmarks/collect_results.py \\")
    print(f"      --results-dir {args.pull_results} \\")
    print(f"      --output {args.pull_results}/matrix_combined.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
