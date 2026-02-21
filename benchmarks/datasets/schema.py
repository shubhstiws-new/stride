"""
Unified long-form schema and per-dataset normalization functions.

All 4 robotics datasets are normalized to the same schema so that
session_detector.py and all benchmark operations work without changes.

Unified schema columns:
    episode_id    str      entity key for gap detection (maps from robot_id)
    frame_index   int64    ordinal position within episode
    timestamp     float64  seconds from episode start
    dim_index     int32    which dimension of state/action vector
    dim_name      str      e.g. "state_0", "action_2", "gripper"
    signal_value  float64  scalar value for that dim at that frame
    task_id       str      task_index or hashed language instruction
    dataset       str      source dataset tag
"""

from typing import Callable

import polars as pl

# ── Unified Polars dtype mapping ──────────────────────────────────────────────

UNIFIED_SCHEMA: dict[str, pl.DataType] = {
    "episode_id": pl.Utf8,
    "frame_index": pl.Int64,
    "timestamp": pl.Float64,
    "dim_index": pl.Int32,
    "dim_name": pl.Utf8,
    "signal_value": pl.Float64,
    "task_id": pl.Utf8,
    "dataset": pl.Utf8,
}

UNIFIED_COLUMNS = list(UNIFIED_SCHEMA.keys())


# ── Normalization helpers ─────────────────────────────────────────────────────

def _explode_vector_columns(
    df: pl.DataFrame,
    vector_cols: list[str],
    prefix_map: dict[str, str] | None = None,
) -> pl.DataFrame:
    """
    Explode multiple fixed-width vector columns into long form rows.

    For each row in df, produces one output row per (column, dim_index) pair.
    prefix_map: {col_name: dim_name_prefix}, e.g. {"observation.state": "state"}
    """
    if prefix_map is None:
        prefix_map = {c: c for c in vector_cols}

    frames = []
    for col in vector_cols:
        if col not in df.columns:
            continue
        prefix = prefix_map.get(col, col)
        # Each list element → one row
        exploded = df.with_columns(
            pl.col(col).alias("_vec")
        ).explode("_vec")

        # Add dim_index: sequential within each (episode_id, frame_index) group
        exploded = exploded.with_columns(
            pl.int_range(pl.len()).over(["episode_id", "frame_index"]).cast(pl.Int32).alias("dim_index")
        ).with_columns(
            (pl.lit(prefix + "_") + pl.col("dim_index").cast(pl.Utf8)).alias("dim_name"),
            pl.col("_vec").cast(pl.Float64).alias("signal_value"),
        ).drop("_vec")

        frames.append(exploded)

    if not frames:
        raise ValueError(f"None of the vector columns {vector_cols} found in DataFrame")

    return pl.concat(frames, how="diagonal_relaxed")


def _ensure_columns(df: pl.DataFrame, dataset_tag: str) -> pl.DataFrame:
    """Cast all columns to unified schema types and select only unified columns."""
    for col_name, dtype in UNIFIED_SCHEMA.items():
        if col_name in df.columns:
            df = df.with_columns(pl.col(col_name).cast(dtype))
    return df.select(UNIFIED_COLUMNS)


# ── Per-dataset normalization functions ───────────────────────────────────────

def normalize_behavior1k(raw_df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize BEHAVIOR-1K dataset to unified schema.

    Raw columns: episode_index, timestamp, observation.state (23-dim list),
                 task_index (or task_name)
    """
    df = raw_df.clone()

    # Rename episode key
    if "episode_index" in df.columns:
        df = df.rename({"episode_index": "episode_id"})
    df = df.with_columns(pl.col("episode_id").cast(pl.Utf8))

    # frame_index: use existing or generate from row number within episode
    if "frame_index" not in df.columns:
        df = df.with_columns(
            pl.int_range(pl.len()).over("episode_id").cast(pl.Int64).alias("frame_index")
        )

    # timestamp: should exist; cast to float seconds
    if "timestamp" not in df.columns:
        # derive from frame_index assuming 30 FPS
        df = df.with_columns(
            (pl.col("frame_index") / 30.0).alias("timestamp")
        )

    # task_id
    task_col = next((c for c in ["task_index", "task_name", "task_id"] if c in df.columns), None)
    if task_col:
        df = df.with_columns(pl.col(task_col).cast(pl.Utf8).alias("task_id"))
    else:
        df = df.with_columns(pl.lit("unknown").alias("task_id"))

    # Explode observation.state (23-dim)
    state_col = next((c for c in ["observation.state", "obs_state", "state"] if c in df.columns), None)
    if state_col:
        df = df.with_columns(pl.col(state_col).alias("episode_id"))  # placeholder
        # Re-attach episode_id properly
        df = df.drop(state_col)

    # If state column is a list, explode; otherwise use signal_value directly
    obs_cols = [c for c in raw_df.columns if "state" in c.lower() or "action" in c.lower()]
    list_cols = [c for c in obs_cols if raw_df[c].dtype == pl.List(pl.Float64)
                 or raw_df[c].dtype == pl.List(pl.Float32)
                 or str(raw_df[c].dtype).startswith("List")]

    if list_cols:
        # Rebuild from raw_df
        df = raw_df.clone()
        if "episode_index" in df.columns:
            df = df.rename({"episode_index": "episode_id"})
        df = df.with_columns(pl.col("episode_id").cast(pl.Utf8))
        if "frame_index" not in df.columns:
            df = df.with_columns(
                pl.int_range(pl.len()).over("episode_id").cast(pl.Int64).alias("frame_index")
            )
        if "timestamp" not in df.columns:
            df = df.with_columns((pl.col("frame_index") / 30.0).alias("timestamp"))
        if task_col:
            df = df.with_columns(pl.col(task_col).cast(pl.Utf8).alias("task_id"))
        else:
            df = df.with_columns(pl.lit("unknown").alias("task_id"))

        prefix_map = {}
        for c in list_cols:
            if "action" in c.lower():
                prefix_map[c] = "action"
            else:
                prefix_map[c] = "state"

        df = _explode_vector_columns(df, list_cols, prefix_map)
    else:
        # Scalar signal_value already present or use first numeric column
        numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]
        val_col = next((c for c in numeric_cols if c not in UNIFIED_COLUMNS), None)
        if val_col:
            df = df.with_columns(
                pl.col(val_col).cast(pl.Float64).alias("signal_value"),
                pl.lit(0).cast(pl.Int32).alias("dim_index"),
                pl.lit("state_0").alias("dim_name"),
            )

    df = df.with_columns(pl.lit("behavior1k").alias("dataset"))
    return _ensure_columns(df, "behavior1k")


def normalize_droid(raw_df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize DROID / RoboInter dataset to unified schema.

    Raw columns: episode_id (or episode_index), frame_index,
                 state/action vectors (list columns), language_instruction
    Timestamp reconstructed from frame_index × 0.033s (30 FPS).
    """
    df = raw_df.clone()

    if "episode_index" in df.columns and "episode_id" not in df.columns:
        df = df.rename({"episode_index": "episode_id"})
    df = df.with_columns(pl.col("episode_id").cast(pl.Utf8))

    if "frame_index" not in df.columns:
        df = df.with_columns(
            pl.int_range(pl.len()).over("episode_id").cast(pl.Int64).alias("frame_index")
        )

    # Reconstruct timestamp from frame_index (30 FPS)
    df = df.with_columns(
        (pl.col("frame_index").cast(pl.Float64) * (1.0 / 30.0)).alias("timestamp")
    )

    # task_id from language instruction
    lang_col = next(
        (c for c in ["language_instruction", "task_id", "task_name", "task"] if c in df.columns),
        None,
    )
    if lang_col:
        df = df.with_columns(pl.col(lang_col).cast(pl.Utf8).alias("task_id"))
    else:
        df = df.with_columns(pl.lit("unknown").alias("task_id"))

    list_cols = [
        c for c in df.columns
        if str(df[c].dtype).startswith("List") and c not in UNIFIED_COLUMNS
    ]
    if list_cols:
        prefix_map = {c: ("action" if "action" in c.lower() else "state") for c in list_cols}
        df = _explode_vector_columns(df, list_cols, prefix_map)
    else:
        numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
        val_col = next((c for c in numeric_cols if c not in UNIFIED_COLUMNS), None)
        if val_col:
            df = df.with_columns(
                pl.col(val_col).cast(pl.Float64).alias("signal_value"),
                pl.lit(0).cast(pl.Int32).alias("dim_index"),
                pl.lit("state_0").alias("dim_name"),
            )

    df = df.with_columns(pl.lit("droid").alias("dataset"))
    return _ensure_columns(df, "droid")


def normalize_oxe(raw_df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize Open X-Embodiment dataset to unified schema.

    Raw columns: episode_id, timestamp (exists natively), state (list), action (list)
    """
    df = raw_df.clone()

    if "episode_index" in df.columns and "episode_id" not in df.columns:
        df = df.rename({"episode_index": "episode_id"})
    df = df.with_columns(pl.col("episode_id").cast(pl.Utf8))

    if "frame_index" not in df.columns:
        df = df.with_columns(
            pl.int_range(pl.len()).over("episode_id").cast(pl.Int64).alias("frame_index")
        )

    # timestamp exists natively in OXE
    if "timestamp" not in df.columns:
        df = df.with_columns(
            (pl.col("frame_index").cast(pl.Float64) / 30.0).alias("timestamp")
        )

    task_col = next(
        (c for c in ["task_id", "task_name", "language_instruction", "task"] if c in df.columns),
        None,
    )
    if task_col:
        df = df.with_columns(pl.col(task_col).cast(pl.Utf8).alias("task_id"))
    else:
        df = df.with_columns(pl.lit("unknown").alias("task_id"))

    list_cols = [
        c for c in df.columns
        if str(df[c].dtype).startswith("List") and c not in UNIFIED_COLUMNS
    ]
    if list_cols:
        prefix_map = {c: ("action" if "action" in c.lower() else "state") for c in list_cols}
        df = _explode_vector_columns(df, list_cols, prefix_map)
    else:
        numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
        val_col = next((c for c in numeric_cols if c not in UNIFIED_COLUMNS), None)
        if val_col:
            df = df.with_columns(
                pl.col(val_col).cast(pl.Float64).alias("signal_value"),
                pl.lit(0).cast(pl.Int32).alias("dim_index"),
                pl.lit("state_0").alias("dim_name"),
            )

    df = df.with_columns(pl.lit("oxe").alias("dataset"))
    return _ensure_columns(df, "oxe")


def normalize_nvidia_physicalai(raw_df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize NVIDIA Physical AI dataset to unified schema.

    Raw columns: episode_index, timestamp, observation.state (53-dim list)
    """
    df = raw_df.clone()

    if "episode_index" in df.columns and "episode_id" not in df.columns:
        df = df.rename({"episode_index": "episode_id"})
    df = df.with_columns(pl.col("episode_id").cast(pl.Utf8))

    if "frame_index" not in df.columns:
        df = df.with_columns(
            pl.int_range(pl.len()).over("episode_id").cast(pl.Int64).alias("frame_index")
        )

    if "timestamp" not in df.columns:
        df = df.with_columns(
            (pl.col("frame_index").cast(pl.Float64) / 30.0).alias("timestamp")
        )

    task_col = next(
        (c for c in ["task_index", "task_id", "task_name"] if c in df.columns), None
    )
    if task_col:
        df = df.with_columns(pl.col(task_col).cast(pl.Utf8).alias("task_id"))
    else:
        df = df.with_columns(pl.lit("unknown").alias("task_id"))

    list_cols = [
        c for c in df.columns
        if str(df[c].dtype).startswith("List") and c not in UNIFIED_COLUMNS
    ]
    if list_cols:
        prefix_map = {c: ("action" if "action" in c.lower() else "state") for c in list_cols}
        df = _explode_vector_columns(df, list_cols, prefix_map)
    else:
        numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
        val_col = next((c for c in numeric_cols if c not in UNIFIED_COLUMNS), None)
        if val_col:
            df = df.with_columns(
                pl.col(val_col).cast(pl.Float64).alias("signal_value"),
                pl.lit(0).cast(pl.Int32).alias("dim_index"),
                pl.lit("state_0").alias("dim_name"),
            )

    df = df.with_columns(pl.lit("nvidia_physicalai").alias("dataset"))
    return _ensure_columns(df, "nvidia_physicalai")


# ── Dataset registry ─────────────────────────────────────────────────────────

DATASET_REGISTRY: dict[str, tuple[str, Callable[[pl.DataFrame], pl.DataFrame]]] = {
    "behavior1k": (
        "behavior-1k/2025-challenge-demos",  # HF: 10k episodes, 119M frames, 50 household tasks
        normalize_behavior1k,
    ),
    "droid": (
        "InternRobotics/RoboInter-Data",  # HF: 236k episodes, LeRobot v2.1 format
        normalize_droid,
    ),
    "oxe": (
        "lerobot/fractal20220817_data",  # OXE subset via LeRobot (RT-X fractal, ~100k episodes)
        normalize_oxe,
    ),
    "nvidia_physicalai": (
        "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",  # NVIDIA Isaac Sim / GR00T
        normalize_nvidia_physicalai,
    ),
}
