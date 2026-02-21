"""
Pandas benchmark: gap-based activity sessions + velocity threshold movement sessions.

Session logic mirrors the PySpark SessionDetector:
  L0  gap > 300 s between consecutive events per robot  → new activity session
  L1  velocity signal sustains > 0.3 for ≥ 20 samples  → movement session
"""

import time
import tracemalloc

import pandas as pd

from . import BenchmarkResult

GAP_THRESHOLD_S: float = 300.0
VELOCITY_THRESHOLD: float = 0.3
SUSTAINED_WINDOW: int = 20  # 20 samples @ 10 Hz = 2 s


def run(data_path: str, scale_factor: int = 1) -> BenchmarkResult:
    """Run Pandas benchmark on robotics session detection."""
    tracemalloc.start()
    t_total = time.perf_counter()

    # ── Load ──────────────────────────────────────────────────────────────────
    t_load = time.perf_counter()
    base = pd.read_parquet(data_path)
    if scale_factor > 1:
        frames = [
            base.assign(robot_id=base["robot_id"] + f"_s{i}")
            for i in range(scale_factor)
        ]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = base
    load_time = time.perf_counter() - t_load
    total_records = len(df)

    # ── Process ───────────────────────────────────────────────────────────────
    t_proc = time.perf_counter()

    df = df.sort_values(["robot_id", "timestamp"]).reset_index(drop=True)

    # L0: gap-based activity sessions
    df["_gap"] = df.groupby("robot_id")["timestamp"].diff().dt.total_seconds()
    df["_is_new"] = df["_gap"].isna() | (df["_gap"] > GAP_THRESHOLD_S)
    df["_sess_num"] = df.groupby("robot_id")["_is_new"].cumsum().astype(int)
    df["activity_session_id"] = df["robot_id"] + "_activity_" + df["_sess_num"].astype(str)
    df.drop(columns=["_gap", "_is_new", "_sess_num"], inplace=True)

    # L1: velocity threshold movement sessions
    vel_mask = df["signal_type"] == "velocity"
    vel_df = df.loc[vel_mask].copy()

    vel_df["_above"] = (vel_df["signal_value"] > VELOCITY_THRESHOLD).astype(int)
    vel_df["_rolling_sum"] = vel_df.groupby("robot_id")["_above"].transform(
        lambda x: x.rolling(SUSTAINED_WINDOW, min_periods=SUSTAINED_WINDOW).sum()
    )
    vel_df["_active"] = vel_df["_rolling_sum"] == float(SUSTAINED_WINDOW)
    vel_df["_prev_active"] = (
        vel_df.groupby("robot_id")["_active"].shift(1).fillna(False)
    )
    vel_df["_is_start"] = vel_df["_active"] & ~vel_df["_prev_active"]
    vel_df["_mnum"] = vel_df.groupby("robot_id")["_is_start"].cumsum().astype(int)

    vel_df["movement_session_id"] = None
    active_idx = vel_df.index[vel_df["_active"]]
    vel_df.loc[active_idx, "movement_session_id"] = (
        vel_df.loc[active_idx, "robot_id"] + "_movement_"
        + vel_df.loc[active_idx, "_mnum"].astype(str)
    )

    df["movement_session_id"] = None
    df.loc[vel_df.index, "movement_session_id"] = vel_df["movement_session_id"]

    # Summaries
    activity_summary = (
        df.groupby("activity_session_id")
        .agg(
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
            event_count=("timestamp", "count"),
            signal_value_avg=("signal_value", "mean"),
        )
        .reset_index()
    )

    mov_df = df.loc[df["movement_session_id"].notna()]
    if not mov_df.empty:
        movement_summary = (
            mov_df.groupby("movement_session_id")
            .agg(
                session_start=("timestamp", "min"),
                session_end=("timestamp", "max"),
                event_count=("timestamp", "count"),
                signal_value_max=("signal_value", "max"),
            )
            .reset_index()
        )
    else:
        movement_summary = pd.DataFrame()

    proc_time = time.perf_counter() - t_proc
    total_time = time.perf_counter() - t_total

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        framework="Pandas",
        total_records=total_records,
        activity_sessions=len(activity_summary),
        movement_sessions=len(movement_summary),
        load_time_s=load_time,
        processing_time_s=proc_time,
        total_time_s=total_time,
        throughput_events_per_sec=total_records / proc_time,
        peak_memory_mb=peak_mem / 1024 / 1024,
    )
