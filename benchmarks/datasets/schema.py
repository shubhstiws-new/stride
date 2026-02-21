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

    Raw columns: episode_index (int), index (frame), timestamp (float),
                 task_index (int), observation.state (list[f64] 23-dim),
                 observation.cam_rel_poses, action, observation.task_info

    Strategy: extract observation.state[0] as signal_value (one row per frame,
    no list explosion) to avoid O(dims) memory amplification.
    200 episodes × ~2k frames = ~400k rows in unified schema.
    """
    df = raw_df.clone()

    # episode_id
    ep_col = next((c for c in ["episode_index", "episode_id"] if c in df.columns), None)
    if ep_col and ep_col != "episode_id":
        df = df.rename({ep_col: "episode_id"})
    df = df.with_columns(pl.col("episode_id").cast(pl.Utf8))

    # frame_index — use "index" column if present, else row-number within episode
    if "frame_index" not in df.columns:
        if "index" in df.columns:
            df = df.rename({"index": "frame_index"})
        else:
            df = df.with_columns(
                pl.int_range(pl.len()).over("episode_id").cast(pl.Int64).alias("frame_index")
            )
    df = df.with_columns(pl.col("frame_index").cast(pl.Int64))

    # timestamp
    if "timestamp" not in df.columns:
        df = df.with_columns((pl.col("frame_index").cast(pl.Float64) / 30.0).alias("timestamp"))

    # task_id
    task_col = next((c for c in ["task_index", "task_name", "task_id"] if c in df.columns), None)
    df = df.with_columns(
        pl.col(task_col).cast(pl.Utf8).alias("task_id") if task_col
        else pl.lit("unknown").alias("task_id")
    )

    # signal_value: extract element [0] from observation.state list (no explosion)
    state_col = next(
        (c for c in ["observation.state", "obs_state", "state"] if c in df.columns), None
    )
    if state_col and str(df[state_col].dtype).startswith("List"):
        df = df.with_columns(
            pl.col(state_col).list.get(0).cast(pl.Float64).alias("signal_value"),
            pl.lit(0).cast(pl.Int32).alias("dim_index"),
            pl.lit("state_0").alias("dim_name"),
        )
    else:
        # Fallback: use first available numeric column
        numeric_col = next(
            (c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)
             and c not in UNIFIED_COLUMNS),
            None,
        )
        val = pl.col(numeric_col).cast(pl.Float64) if numeric_col else pl.lit(0.0)
        df = df.with_columns(
            val.alias("signal_value"),
            pl.lit(0).cast(pl.Int32).alias("dim_index"),
            pl.lit("state_0").alias("dim_name"),
        )

    df = df.with_columns(pl.lit("behavior1k").alias("dataset"))
    return _ensure_columns(df, "behavior1k")


def normalize_droid(raw_df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize DROID (lerobot/droid) dataset to unified schema.

    Raw columns: episode_index, frame_index, timestamp,
                 observation.state (list), action (list), language_instruction
    Strategy: extract list[0] as signal_value — no explosion.
    """
    df = raw_df.clone()

    if "episode_index" in df.columns and "episode_id" not in df.columns:
        df = df.rename({"episode_index": "episode_id"})
    df = df.with_columns(pl.col("episode_id").cast(pl.Utf8))

    if "frame_index" not in df.columns:
        if "index" in df.columns:
            df = df.rename({"index": "frame_index"})
        else:
            df = df.with_columns(
                pl.int_range(pl.len()).over("episode_id").cast(pl.Int64).alias("frame_index")
            )
    df = df.with_columns(pl.col("frame_index").cast(pl.Int64))

    if "timestamp" not in df.columns:
        df = df.with_columns(
            (pl.col("frame_index").cast(pl.Float64) / 30.0).alias("timestamp")
        )

    lang_col = next(
        (c for c in ["language_instruction", "task_id", "task_name", "task"] if c in df.columns),
        None,
    )
    df = df.with_columns(
        pl.col(lang_col).cast(pl.Utf8).alias("task_id") if lang_col
        else pl.lit("unknown").alias("task_id")
    )

    # Extract first element of first available list column as signal_value
    list_col = next(
        (c for c in df.columns if str(df[c].dtype).startswith("List") and c not in UNIFIED_COLUMNS),
        None,
    )
    if list_col:
        df = df.with_columns(
            pl.col(list_col).list.get(0).cast(pl.Float64).alias("signal_value"),
            pl.lit(0).cast(pl.Int32).alias("dim_index"),
            pl.lit("state_0").alias("dim_name"),
        )
    else:
        numeric_col = next(
            (c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)
             and c not in UNIFIED_COLUMNS), None
        )
        val = pl.col(numeric_col).cast(pl.Float64) if numeric_col else pl.lit(0.0)
        df = df.with_columns(
            val.alias("signal_value"),
            pl.lit(0).cast(pl.Int32).alias("dim_index"),
            pl.lit("state_0").alias("dim_name"),
        )

    df = df.with_columns(pl.lit("droid").alias("dataset"))
    return _ensure_columns(df, "droid")


def normalize_oxe(raw_df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize Open X-Embodiment (lerobot/fractal20220817_data) to unified schema.

    Raw columns: episode_index, frame_index, timestamp, observation.state (list),
                 action (list), language_instruction (optional)
    Strategy: extract list[0] as signal_value — no explosion.
    """
    df = raw_df.clone()

    if "episode_index" in df.columns and "episode_id" not in df.columns:
        df = df.rename({"episode_index": "episode_id"})
    df = df.with_columns(pl.col("episode_id").cast(pl.Utf8))

    if "frame_index" not in df.columns:
        if "index" in df.columns:
            df = df.rename({"index": "frame_index"})
        else:
            df = df.with_columns(
                pl.int_range(pl.len()).over("episode_id").cast(pl.Int64).alias("frame_index")
            )
    df = df.with_columns(pl.col("frame_index").cast(pl.Int64))

    if "timestamp" not in df.columns:
        df = df.with_columns(
            (pl.col("frame_index").cast(pl.Float64) / 30.0).alias("timestamp")
        )

    task_col = next(
        (c for c in ["task_id", "task_name", "language_instruction", "task"] if c in df.columns),
        None,
    )
    df = df.with_columns(
        pl.col(task_col).cast(pl.Utf8).alias("task_id") if task_col
        else pl.lit("unknown").alias("task_id")
    )

    list_col = next(
        (c for c in df.columns if str(df[c].dtype).startswith("List") and c not in UNIFIED_COLUMNS),
        None,
    )
    if list_col:
        df = df.with_columns(
            pl.col(list_col).list.get(0).cast(pl.Float64).alias("signal_value"),
            pl.lit(0).cast(pl.Int32).alias("dim_index"),
            pl.lit("state_0").alias("dim_name"),
        )
    else:
        numeric_col = next(
            (c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)
             and c not in UNIFIED_COLUMNS), None
        )
        val = pl.col(numeric_col).cast(pl.Float64) if numeric_col else pl.lit(0.0)
        df = df.with_columns(
            val.alias("signal_value"),
            pl.lit(0).cast(pl.Int32).alias("dim_index"),
            pl.lit("state_0").alias("dim_name"),
        )

    df = df.with_columns(pl.lit("oxe").alias("dataset"))
    return _ensure_columns(df, "oxe")


def normalize_nvidia_physicalai(raw_df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize NVIDIA Physical AI dataset to unified schema.

    Raw columns: episode_index, timestamp, observation.state (53-dim list),
                 action (list), task_index (optional)
    Strategy: extract list[0] as signal_value — no explosion.
    """
    df = raw_df.clone()

    if "episode_index" in df.columns and "episode_id" not in df.columns:
        df = df.rename({"episode_index": "episode_id"})
    df = df.with_columns(pl.col("episode_id").cast(pl.Utf8))

    if "frame_index" not in df.columns:
        if "index" in df.columns:
            df = df.rename({"index": "frame_index"})
        else:
            df = df.with_columns(
                pl.int_range(pl.len()).over("episode_id").cast(pl.Int64).alias("frame_index")
            )
    df = df.with_columns(pl.col("frame_index").cast(pl.Int64))

    if "timestamp" not in df.columns:
        df = df.with_columns(
            (pl.col("frame_index").cast(pl.Float64) / 30.0).alias("timestamp")
        )

    task_col = next(
        (c for c in ["task_index", "task_id", "task_name"] if c in df.columns), None
    )
    df = df.with_columns(
        pl.col(task_col).cast(pl.Utf8).alias("task_id") if task_col
        else pl.lit("unknown").alias("task_id")
    )

    list_col = next(
        (c for c in df.columns if str(df[c].dtype).startswith("List") and c not in UNIFIED_COLUMNS),
        None,
    )
    if list_col:
        df = df.with_columns(
            pl.col(list_col).list.get(0).cast(pl.Float64).alias("signal_value"),
            pl.lit(0).cast(pl.Int32).alias("dim_index"),
            pl.lit("state_0").alias("dim_name"),
        )
    else:
        numeric_col = next(
            (c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)
             and c not in UNIFIED_COLUMNS), None
        )
        val = pl.col(numeric_col).cast(pl.Float64) if numeric_col else pl.lit(0.0)
        df = df.with_columns(
            val.alias("signal_value"),
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
        "lerobot/droid",  # HF: Stanford DROID dataset in LeRobot Parquet format
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
