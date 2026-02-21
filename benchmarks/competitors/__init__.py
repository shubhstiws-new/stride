"""
Shared interface for framework benchmarks.

Each benchmark module exposes: run(data_path: str) -> BenchmarkResult
"""

from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Standardised results from a single framework benchmark run."""

    framework: str
    total_records: int
    activity_sessions: int
    movement_sessions: int
    load_time_s: float
    processing_time_s: float
    total_time_s: float
    throughput_events_per_sec: float
    peak_memory_mb: float  # 0.0 means "not measured / JVM heap"
