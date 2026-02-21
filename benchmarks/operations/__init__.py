"""
Operations layer for the benchmark matrix.

Each operation module exposes: run(data_path, framework, hardware_cfg) -> MatrixResult
"""

from dataclasses import asdict, dataclass


@dataclass
class MatrixResult:
    """Standardised result for a single benchmark matrix cell."""

    hardware: str           # "mac-cpu" | "linux-cpu" | "linux-gpu"
    framework: str          # "pyspark" | "polars" | "pyspark-rapids" | "cudf" | etc.
    dataset: str            # "behavior1k" | "droid" | "oxe" | "nvidia_physicalai"
    filesize_target: str    # "1mb" | "10mb" | "100mb" | "1gb"
    operation: str          # "io_scan" | "filter_agg" | "window_hsm" | "join"
    total_records: int
    num_files: int
    actual_file_size_mb: float
    load_time_s: float
    compute_time_s: float
    total_time_s: float
    throughput_records_per_sec: float
    peak_memory_mb: float
    gpu_memory_mb: float    # 0.0 if not GPU
    result_rows: int        # output row count (sessions, groups, etc.)
    status: str             # "ok" | "oom" | "err" | "skip"

    def to_dict(self) -> dict:
        return asdict(self)


def make_error_result(
    hardware: str,
    framework: str,
    dataset: str,
    filesize_target: str,
    operation: str,
    status: str = "err",
    num_files: int = 0,
    actual_file_size_mb: float = 0.0,
) -> MatrixResult:
    """Return a MatrixResult with zeroed metrics and given status."""
    return MatrixResult(
        hardware=hardware,
        framework=framework,
        dataset=dataset,
        filesize_target=filesize_target,
        operation=operation,
        total_records=0,
        num_files=num_files,
        actual_file_size_mb=actual_file_size_mb,
        load_time_s=0.0,
        compute_time_s=0.0,
        total_time_s=0.0,
        throughput_records_per_sec=0.0,
        peak_memory_mb=0.0,
        gpu_memory_mb=0.0,
        result_rows=0,
        status=status,
    )
