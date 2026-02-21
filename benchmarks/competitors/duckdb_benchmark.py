"""
DuckDB benchmark: gap-based activity sessions + velocity threshold movement sessions.

Uses pure SQL window functions (LAG, SUM OVER ROWS BETWEEN) that mirror the
PySpark window-function approach — no Python loops over rows.
"""

import time
import tracemalloc
from pathlib import Path

import duckdb

from . import BenchmarkResult

GAP_THRESHOLD_S: float = 300.0
VELOCITY_THRESHOLD: float = 0.3
SUSTAINED_WINDOW: int = 20  # 20 rows = 2 s @ 10 Hz


def run(data_path: str, scale_factor: int = 1) -> BenchmarkResult:
    """Run DuckDB benchmark on robotics session detection."""
    tracemalloc.start()
    t_total = time.perf_counter()

    con = duckdb.connect()
    parquet_glob = str(Path(data_path) / "*.parquet")

    # ── Load ──────────────────────────────────────────────────────────────────
    t_load = time.perf_counter()
    if scale_factor > 1:
        con.execute(f"""
            CREATE TABLE events AS
            WITH base AS (SELECT * FROM read_parquet('{parquet_glob}')),
                 sids AS (SELECT unnest(range(0, {scale_factor})) AS _sid)
            SELECT b.* EXCLUDE (robot_id),
                   b.robot_id || '_s' || CAST(s._sid AS VARCHAR) AS robot_id
            FROM base b CROSS JOIN sids s
        """)
    else:
        con.execute(
            f"CREATE TABLE events AS SELECT * FROM read_parquet('{parquet_glob}')"
        )
    total_records = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    load_time = time.perf_counter() - t_load

    # ── Process ───────────────────────────────────────────────────────────────
    t_proc = time.perf_counter()

    # L0: gap-based activity sessions on all events
    # epoch() on an interval gives total seconds as double
    con.execute(f"""
        CREATE TABLE activity AS
        WITH gap_calc AS (
            SELECT *,
                epoch(
                    timestamp
                    - LAG(timestamp) OVER (PARTITION BY robot_id ORDER BY timestamp)
                ) AS _gap_s
            FROM events
        ),
        new_session_flags AS (
            SELECT *,
                CASE
                    WHEN _gap_s IS NULL OR _gap_s > {GAP_THRESHOLD_S} THEN 1
                    ELSE 0
                END AS _is_new
            FROM gap_calc
        )
        SELECT *,
            SUM(_is_new) OVER (PARTITION BY robot_id ORDER BY timestamp
                               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                AS _sess_num,
            robot_id || '_activity_' ||
            CAST(
                SUM(_is_new) OVER (PARTITION BY robot_id ORDER BY timestamp
                                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            AS VARCHAR) AS activity_session_id
        FROM new_session_flags
    """)

    # L1: velocity threshold movement sessions (velocity rows only)
    con.execute(f"""
        CREATE TABLE movement AS
        WITH velocity_events AS (
            SELECT *,
                SUM(CASE WHEN signal_value > {VELOCITY_THRESHOLD} THEN 1 ELSE 0 END)
                    OVER (
                        PARTITION BY robot_id ORDER BY timestamp
                        ROWS BETWEEN {SUSTAINED_WINDOW - 1} PRECEDING AND CURRENT ROW
                    ) AS _rolling_sum
            FROM events
            WHERE signal_type = 'velocity'
        ),
        with_active AS (
            SELECT *,
                (_rolling_sum = {SUSTAINED_WINDOW}) AS _active,
                COALESCE(
                    LAG((_rolling_sum = {SUSTAINED_WINDOW})::BOOLEAN)
                        OVER (PARTITION BY robot_id ORDER BY timestamp),
                    FALSE
                ) AS _prev_active
            FROM velocity_events
        ),
        with_start AS (
            SELECT *,
                CASE WHEN _active AND NOT _prev_active THEN 1 ELSE 0 END AS _is_start
            FROM with_active
        )
        SELECT *,
            SUM(_is_start) OVER (PARTITION BY robot_id ORDER BY timestamp
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                AS _move_num,
            CASE
                WHEN _active THEN
                    robot_id || '_movement_' ||
                    CAST(
                        SUM(_is_start) OVER (PARTITION BY robot_id ORDER BY timestamp
                                             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                    AS VARCHAR)
                ELSE NULL
            END AS movement_session_id
        FROM with_start
    """)

    # Activity summary
    activity_sessions = con.execute(
        "SELECT COUNT(DISTINCT activity_session_id) FROM activity"
    ).fetchone()[0]

    # Movement summary
    movement_sessions = con.execute(
        "SELECT COUNT(DISTINCT movement_session_id) FROM movement "
        "WHERE movement_session_id IS NOT NULL"
    ).fetchone()[0]

    # Also materialise summaries (mirrors what other frameworks do)
    con.execute("""
        SELECT activity_session_id,
               MIN(timestamp) AS session_start,
               MAX(timestamp) AS session_end,
               COUNT(*) AS event_count,
               AVG(signal_value) AS signal_value_avg
        FROM activity
        GROUP BY activity_session_id
    """).fetchdf()

    con.execute("""
        SELECT movement_session_id,
               MIN(timestamp) AS session_start,
               MAX(timestamp) AS session_end,
               COUNT(*) AS event_count,
               MAX(signal_value) AS signal_value_max
        FROM movement
        WHERE movement_session_id IS NOT NULL
        GROUP BY movement_session_id
    """).fetchdf()

    proc_time = time.perf_counter() - t_proc
    total_time = time.perf_counter() - t_total

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    con.close()

    return BenchmarkResult(
        framework="DuckDB",
        total_records=total_records,
        activity_sessions=activity_sessions,
        movement_sessions=movement_sessions,
        load_time_s=load_time,
        processing_time_s=proc_time,
        total_time_s=total_time,
        throughput_events_per_sec=total_records / proc_time,
        peak_memory_mb=peak_mem / 1024 / 1024,
    )
