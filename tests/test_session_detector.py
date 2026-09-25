from datetime import datetime, timedelta

import pytest
from pyspark.sql import functions as F

from hsm_pyspark.spark.session_detector import SessionConfig, SessionDetector, detect_sessions

T0 = datetime(2026, 1, 1, 0, 0, 0)
CONFIG = SessionConfig(
    entity_key="robot_id",
    timestamp_col="timestamp",
    signal_col="signal_type",
    value_col="signal_value",
)


def _frame(spark, rows):
    """rows: iterable of (robot_id, seconds_from_t0, signal_type, value)."""
    data = [(r, T0 + timedelta(seconds=s), sig, float(v)) for r, s, sig, v in rows]
    return spark.createDataFrame(data, ["robot_id", "timestamp", "signal_type", "signal_value"])


def _velocity_run(robot, start_s, duration_s, value, hz=10):
    return [(robot, start_s + i / hz, "velocity", value) for i in range(int(duration_s * hz))]


# ── gap sessions ────────────────────────────────────────────────────────────

def test_gap_sessions_split_on_inactivity(spark):
    rows = [("R1", s, "status", 1) for s in (0, 10, 20)]  # session 1
    rows += [("R1", s, "status", 1) for s in (1000, 1010)]  # session 2 after >300 s gap
    df = SessionDetector(CONFIG).detect_gap_sessions(_frame(spark, rows), 300)

    ids = [r.activity_session_id for r in df.orderBy("timestamp").collect()]
    assert ids == ["R1_activity_1"] * 3 + ["R1_activity_2"] * 2


def test_gap_sessions_are_independent_per_entity(spark):
    rows = [("R1", 0, "status", 1), ("R2", 5, "status", 1), ("R1", 600, "status", 1)]
    df = SessionDetector(CONFIG).detect_gap_sessions(_frame(spark, rows), 300)

    per_robot = {
        r.robot_id: r.n
        for r in df.groupBy("robot_id")
        .agg(F.countDistinct("activity_session_id").alias("n"))
        .collect()
    }
    assert per_robot == {"R1": 2, "R2": 1}


# ── threshold sessions ──────────────────────────────────────────────────────

def test_threshold_session_requires_sustained_duration(spark):
    # 1 s above threshold is too short for a 2 s requirement.
    rows = _velocity_run("R1", 0, 1.0, 0.5)
    df = SessionDetector(CONFIG).detect_threshold_sessions(
        _frame(spark, rows), "velocity", "> 0.3", sustained_seconds=2.0, session_name="move"
    )
    assert df.filter(F.col("move_session_id").isNotNull()).count() == 0


def test_threshold_session_detected_after_sustained_duration(spark):
    rows = _velocity_run("R1", 0, 3.0, 0.5)
    df = SessionDetector(CONFIG).detect_threshold_sessions(
        _frame(spark, rows), "velocity", "> 0.3", sustained_seconds=2.0, session_name="move"
    )
    active = df.filter(F.col("move_session_id").isNotNull())
    assert active.select("move_session_id").distinct().count() == 1
    # Samples from ~1.9 s to 3.0 s qualify at 10 Hz.
    assert 10 <= active.count() <= 12


def test_threshold_window_is_measured_in_seconds_not_milliseconds(spark):
    # Regression: the window was ordered by epoch seconds but bounded in
    # milliseconds, so a "2 s" window actually spanned 2000 s and merged
    # short bursts that were minutes apart.
    rows = []
    for burst_start in (0, 60, 120, 180):  # 1.5 s bursts, one minute apart
        rows += _velocity_run("R1", burst_start, 1.5, 0.5)
    df = SessionDetector(CONFIG).detect_threshold_sessions(
        _frame(spark, rows), "velocity", "> 0.3", sustained_seconds=2.0, session_name="move"
    )
    assert df.filter(F.col("move_session_id").isNotNull()).count() == 0


def test_threshold_respects_sample_rate(spark):
    rows = _velocity_run("R1", 0, 3.0, 0.5, hz=2)
    detector = SessionDetector(CONFIG)
    df = detector.detect_threshold_sessions(
        _frame(spark, rows), "velocity", "> 0.3", 2.0, session_name="move", sample_rate_hz=2
    )
    assert df.filter(F.col("move_session_id").isNotNull()).count() > 0


# ── summaries and helpers ───────────────────────────────────────────────────

def test_detect_sessions_summary(spark):
    rows = [("R1", s, "status", 1) for s in (0, 30, 60)] + [("R1", 1000, "status", 1)]
    result = detect_sessions(_frame(spark, rows), "robot_id", "timestamp", 300)

    summary = {
        r.activity_session_id: (r.event_count, r.session_duration_seconds)
        for r in result.session_summary.collect()
    }
    assert summary == {"R1_activity_1": (3, 60), "R1_activity_2": (1, 0)}


@pytest.mark.parametrize(
    "condition, value, expected",
    [("> 0.5", 0.6, True), ("> 0.5", 0.5, False), (">= 0.5", 0.5, True),
     ("< 1", 0.0, True), ("<= 1", 1.0, True), ("== 2", 2.0, True), ("!= 2", 2.0, False)],
)
def test_parse_condition(spark, condition, value, expected):
    df = spark.createDataFrame([(value,)], ["v"])
    col = SessionDetector(CONFIG)._parse_condition("v", condition)
    assert df.select(col.alias("ok")).first().ok is expected


def test_parse_condition_rejects_unknown_operator():
    with pytest.raises(ValueError):
        SessionDetector(CONFIG)._parse_condition("v", "~ 3")
