"""
Dask benchmark: gap-based activity sessions + velocity threshold movement sessions.

Strategy:
  1. Read parquet lazily with dask.dataframe (parallel I/O across partitions)
  2. Shuffle by robot_id → each partition holds one complete robot's data
  3. map_partitions applies per-robot session detection in parallel
  4. .compute() materialises; tracemalloc tracks peak Python heap usage
"""

import time
import tracemalloc

import pandas as pd

from . import BenchmarkResult

GAP_THRESHOLD_S: float = 300.0
VELOCITY_THRESHOLD: float = 0.3
SUSTAINED_WINDOW: int = 20  # 20 samples @ 10 Hz = 2 s

_OUTPUT_COLS = [
    "robot_id",
    "timestamp",
    "signal_type",
    "signal_value",
    "activity_session_id",
    "movement_session_id",
]


def _process_robot(robot_df: pd.DataFrame, robot_id: str) -> pd.DataFrame:
    """Apply session detection to a single robot's sorted DataFrame."""
    # reset_index avoids duplicate-index issues from Dask shuffle partitioning
    robot_df = robot_df.reset_index(drop=True).sort_values("timestamp").reset_index(drop=True)

    # L0: gap-based activity sessions
    robot_df["_gap"] = robot_df["timestamp"].diff().dt.total_seconds()
    robot_df["_is_new"] = robot_df["_gap"].isna() | (robot_df["_gap"] > GAP_THRESHOLD_S)
    robot_df["_sess_num"] = robot_df["_is_new"].cumsum().astype(int)
    robot_df["activity_session_id"] = robot_id + "_activity_" + robot_df["_sess_num"].astype(str)
    robot_df.drop(columns=["_gap", "_is_new", "_sess_num"], inplace=True)

    # L1: velocity threshold movement sessions
    robot_df["movement_session_id"] = None
    vel_mask = robot_df["signal_type"] == "velocity"
    if vel_mask.any():
        vel_df = robot_df.loc[vel_mask].copy()
        vel_df["_above"] = (vel_df["signal_value"] > VELOCITY_THRESHOLD).astype(int)
        vel_df["_roll"] = vel_df["_above"].rolling(
            SUSTAINED_WINDOW, min_periods=SUSTAINED_WINDOW
        ).sum()
        vel_df["_active"] = vel_df["_roll"] == float(SUSTAINED_WINDOW)
        vel_df["_prev"] = vel_df["_active"].shift(1).fillna(False)
        vel_df["_start"] = vel_df["_active"] & ~vel_df["_prev"]
        vel_df["_mnum"] = vel_df["_start"].cumsum().astype(int)

        active_idx = vel_df.index[vel_df["_active"]]
        if len(active_idx) > 0:
            robot_df.loc[active_idx, "movement_session_id"] = (
                robot_id + "_movement_" + vel_df.loc[active_idx, "_mnum"].astype(str)
            ).values

    return robot_df[_OUTPUT_COLS]


def _process_partition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a Dask partition that may contain one or more complete robots.

    After shuffle(on='robot_id'), each partition holds all rows for a subset
    of robots, so we groupby robot_id and apply session logic to each group.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    results = []
    for robot_id, robot_df in df.groupby("robot_id"):
        results.append(_process_robot(robot_df, str(robot_id)))

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=_OUTPUT_COLS)


def run(data_path: str, scale_factor: int = 1) -> BenchmarkResult:
    """Run Dask benchmark on robotics session detection."""
    import dask.dataframe as dd  # deferred import — startup cost

    tracemalloc.start()
    t_total = time.perf_counter()

    # ── Load ──────────────────────────────────────────────────────────────────
    t_load = time.perf_counter()
    base_ddf = dd.read_parquet(data_path)
    if scale_factor > 1:
        parts = [
            base_ddf.assign(robot_id=base_ddf["robot_id"] + f"_s{i}")
            for i in range(scale_factor)
        ]
        ddf = dd.concat(parts)  # still lazy — never materialises all N copies
    else:
        ddf = base_ddf
    total_records = len(ddf)  # triggers actual file reads
    load_time = time.perf_counter() - t_load

    # ── Process ───────────────────────────────────────────────────────────────
    t_proc = time.perf_counter()

    nrobots = ddf["robot_id"].nunique().compute()

    # Shuffle so all rows for each robot land in one partition
    ddf_by_robot = ddf.shuffle(on="robot_id", npartitions=int(nrobots))

    # Output schema for map_partitions
    meta = pd.DataFrame(
        {
            "robot_id": pd.Series([], dtype="str"),
            "timestamp": pd.Series([], dtype="datetime64[us]"),
            "signal_type": pd.Series([], dtype="str"),
            "signal_value": pd.Series([], dtype="float64"),
            "activity_session_id": pd.Series([], dtype="str"),
            "movement_session_id": pd.Series([], dtype="object"),
        }
    )

    result_ddf = ddf_by_robot.map_partitions(_process_partition, meta=meta)
    result = result_ddf.compute()

    # Activity summary
    activity_summary = (
        result.groupby("activity_session_id")
        .agg(
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
            event_count=("timestamp", "count"),
            signal_value_avg=("signal_value", "mean"),
        )
        .reset_index()
    )

    mov_df = result.loc[result["movement_session_id"].notna()]
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
        framework="Dask",
        total_records=total_records,
        activity_sessions=len(activity_summary),
        movement_sessions=len(movement_summary),
        load_time_s=load_time,
        processing_time_s=proc_time,
        total_time_s=total_time,
        throughput_events_per_sec=total_records / proc_time,
        peak_memory_mb=peak_mem / 1024 / 1024,
    )
