"""
Polars benchmark: gap-based activity sessions + velocity threshold movement sessions.

Uses Polars lazy API with window expressions (.over()) for per-robot computations.
"""

import time
import tracemalloc
from pathlib import Path

import polars as pl

from . import BenchmarkResult

GAP_THRESHOLD_S: float = 300.0
VELOCITY_THRESHOLD: float = 0.3
SUSTAINED_WINDOW: int = 20  # 20 samples @ 10 Hz = 2 s

# ── microseconds per second (Polars stores timestamps as µs since epoch) ──────
_US_PER_S: int = 1_000_000


def run(data_path: str, scale_factor: int = 1) -> BenchmarkResult:
    """Run Polars benchmark on robotics session detection."""
    tracemalloc.start()
    t_total = time.perf_counter()

    # ── Load ──────────────────────────────────────────────────────────────────
    t_load = time.perf_counter()
    # Accept directory; Polars reads all parquet files found inside
    parquet_glob = str(Path(data_path) / "*.parquet")
    base = pl.read_parquet(parquet_glob)
    if scale_factor > 1:
        frames = [
            base.with_columns(
                (pl.col("robot_id") + f"_s{i}").alias("robot_id")
            )
            for i in range(scale_factor)
        ]
        df = pl.concat(frames, rechunk=False)
    else:
        df = base
    load_time = time.perf_counter() - t_load
    total_records = len(df)

    # ── Process ───────────────────────────────────────────────────────────────
    t_proc = time.perf_counter()

    df = df.sort(["robot_id", "timestamp"])

    # L0: gap-based activity sessions
    # Cast timestamp to Int64 (µs since epoch) for reliable arithmetic
    df = df.with_columns(
        (
            pl.col("timestamp").cast(pl.Int64).diff().over("robot_id") / _US_PER_S
        ).alias("_gap_s")
    ).with_columns(
        (pl.col("_gap_s").is_null() | (pl.col("_gap_s") > GAP_THRESHOLD_S))
        .cast(pl.UInt32)
        .alias("_is_new")
    ).with_columns(
        pl.col("_is_new").cum_sum().over("robot_id").alias("_sess_num")
    ).with_columns(
        (pl.col("robot_id") + "_activity_" + pl.col("_sess_num").cast(pl.String))
        .alias("activity_session_id")
    ).drop(["_gap_s", "_is_new", "_sess_num"])

    # L1: velocity threshold movement sessions (computed on velocity subset)
    vel_df = df.filter(pl.col("signal_type") == "velocity")

    vel_df = (
        vel_df.with_columns(
            (pl.col("signal_value") > VELOCITY_THRESHOLD).cast(pl.Int64).alias("_above")
        )
        .with_columns(
            pl.col("_above")
            .rolling_sum(window_size=SUSTAINED_WINDOW, min_periods=SUSTAINED_WINDOW)
            .over("robot_id")
            .alias("_rolling_sum")
        )
        .with_columns(
            (pl.col("_rolling_sum") == SUSTAINED_WINDOW).alias("_active")
        )
        .with_columns(
            pl.col("_active").shift(1).over("robot_id").fill_null(False).alias("_prev_active")
        )
        .with_columns(
            (pl.col("_active") & ~pl.col("_prev_active")).cast(pl.UInt32).alias("_is_start")
        )
        .with_columns(
            pl.col("_is_start").cum_sum().over("robot_id").alias("_mnum")
        )
        .with_columns(
            pl.when(pl.col("_active"))
            .then(pl.col("robot_id") + "_movement_" + pl.col("_mnum").cast(pl.String))
            .otherwise(None)
            .alias("movement_session_id")
        )
        .drop(["_above", "_rolling_sum", "_active", "_prev_active", "_is_start", "_mnum"])
    )

    # Join movement session IDs back onto the full dataset
    # Keep only original columns + movement_session_id
    join_cols = [c for c in df.columns if c not in ("movement_session_id",)]
    df = df.join(
        vel_df.select(["robot_id", "timestamp", "movement_session_id"]),
        on=["robot_id", "timestamp"],
        how="left",
    )

    # Activity summary
    activity_summary = (
        df.group_by("activity_session_id")
        .agg(
            pl.col("timestamp").min().alias("session_start"),
            pl.col("timestamp").max().alias("session_end"),
            pl.len().alias("event_count"),
            pl.col("signal_value").mean().alias("signal_value_avg"),
        )
    )

    # Movement summary
    movement_summary = (
        df.filter(pl.col("movement_session_id").is_not_null())
        .group_by("movement_session_id")
        .agg(
            pl.col("timestamp").min().alias("session_start"),
            pl.col("timestamp").max().alias("session_end"),
            pl.len().alias("event_count"),
            pl.col("signal_value").max().alias("signal_value_max"),
        )
    )

    proc_time = time.perf_counter() - t_proc
    total_time = time.perf_counter() - t_total

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        framework="Polars",
        total_records=total_records,
        activity_sessions=len(activity_summary),
        movement_sessions=len(movement_summary),
        load_time_s=load_time,
        processing_time_s=proc_time,
        total_time_s=total_time,
        throughput_events_per_sec=total_records / proc_time,
        peak_memory_mb=peak_mem / 1024 / 1024,
    )
