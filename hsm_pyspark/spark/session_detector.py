"""
Pure PySpark Session Detection using Window Functions

This module implements hierarchical session detection without UDFs,
using only native PySpark operations for maximum Catalyst optimization.

Key techniques:
- Window functions (lag, lead, sum over window) for state tracking
- rangeBetween for time-based windows
- Cumulative sum for session ID assignment
- Temporal joins for parent-child linking
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


@dataclass
class SessionConfig:
    """Configuration for session detection."""

    entity_key: str  # Column to partition by (e.g., robot_id)
    timestamp_col: str  # Timestamp column
    signal_col: str  # Signal type column
    value_col: str  # Signal value column


@dataclass
class SessionResult:
    """Results from session detection."""

    events_with_sessions: DataFrame
    session_summary: DataFrame
    orphan_events: Optional[DataFrame] = None
    anomalies: Optional[DataFrame] = None


class SessionDetector:
    """
    Pure PySpark session detector using window functions.

    Detects hierarchical sessions by:
    1. Evaluating entry/exit conditions per event
    2. Marking session boundaries using transitions
    3. Assigning session IDs via cumulative sum
    4. Linking child sessions to parents temporally
    """

    def __init__(self, config: SessionConfig):
        self.config = config

    def detect_level0_sessions(
        self,
        df: DataFrame,
        entry_signal: str,
        entry_condition: str,
        exit_signal: str,
        exit_condition: str,
        session_name: str = "connection",
    ) -> DataFrame:
        """
        Detect top-level sessions (e.g., connection sessions).

        Uses status signal to detect session boundaries.

        Args:
            df: Input DataFrame with sensor readings
            entry_signal: Signal type that indicates session start
            entry_condition: Condition for entry (e.g., "> 0")
            exit_signal: Signal type for session end
            exit_condition: Condition for exit
            session_name: Name prefix for session columns

        Returns:
            DataFrame with session IDs and boundaries marked
        """
        entity_key = self.config.entity_key
        timestamp_col = self.config.timestamp_col
        signal_col = self.config.signal_col
        value_col = self.config.value_col

        # Define window for sequential analysis
        window = Window.partitionBy(entity_key).orderBy(timestamp_col)

        # Filter to relevant signals and pivot to wide format
        # For MVP, we'll work with pre-pivoted or filtered data
        session_col = f"{session_name}_session_id"
        start_col = f"{session_name}_start"
        end_col = f"{session_name}_end"

        result = (
            df
            # Identify entry events
            .withColumn(
                "_entry_match",
                (F.col(signal_col) == entry_signal)
                & self._parse_condition(value_col, entry_condition),
            )
            # Identify exit events
            .withColumn(
                "_exit_match",
                (F.col(signal_col) == exit_signal)
                & self._parse_condition(value_col, exit_condition),
            )
            # Track previous state
            .withColumn("_prev_entry", F.lag("_entry_match").over(window))
            .withColumn("_prev_exit", F.lag("_exit_match").over(window))
            # Detect session start (entry after not being in entry state)
            .withColumn(
                start_col,
                F.col("_entry_match") & ~F.coalesce(F.col("_prev_entry"), F.lit(False)),
            )
            # Detect session end
            .withColumn(
                end_col,
                F.col("_exit_match") & F.coalesce(F.col("_prev_entry"), F.lit(False)),
            )
            # Assign session IDs using cumulative sum
            .withColumn(
                session_col,
                F.when(
                    F.sum(F.when(F.col(start_col), 1).otherwise(0)).over(window) > 0,
                    F.concat(
                        F.col(entity_key),
                        F.lit(f"_{session_name}_"),
                        F.sum(F.when(F.col(start_col), 1).otherwise(0)).over(window),
                    ),
                ).otherwise(None),
            )
            # Clean up temp columns
            .drop("_entry_match", "_exit_match", "_prev_entry", "_prev_exit")
        )

        return result

    def detect_gap_sessions(
        self,
        df: DataFrame,
        gap_threshold_seconds: float,
        session_name: str = "activity",
    ) -> DataFrame:
        """
        Detect sessions based on activity gaps.

        A new session starts when the time gap from previous event
        exceeds the threshold.

        Args:
            df: Input DataFrame
            gap_threshold_seconds: Minimum gap to start new session
            session_name: Name prefix for session columns

        Returns:
            DataFrame with gap-based session IDs
        """
        entity_key = self.config.entity_key
        timestamp_col = self.config.timestamp_col

        window = Window.partitionBy(entity_key).orderBy(timestamp_col)
        session_col = f"{session_name}_session_id"

        result = (
            df
            # Calculate time gap from previous event
            .withColumn("_prev_ts", F.lag(timestamp_col).over(window))
            .withColumn(
                "_gap_seconds",
                (F.col(timestamp_col).cast("double") - F.col("_prev_ts").cast("double")),
            )
            # New session if gap exceeds threshold or first event
            .withColumn(
                "_is_new_session",
                (F.col("_gap_seconds") > gap_threshold_seconds) | F.col("_gap_seconds").isNull(),
            )
            # Assign session IDs
            .withColumn(
                session_col,
                F.concat(
                    F.col(entity_key),
                    F.lit(f"_{session_name}_"),
                    F.sum(F.when(F.col("_is_new_session"), 1).otherwise(0)).over(window),
                ),
            )
            .drop("_prev_ts", "_gap_seconds", "_is_new_session")
        )

        return result

    def detect_threshold_sessions(
        self,
        df: DataFrame,
        signal_type: str,
        threshold_condition: str,
        sustained_seconds: float,
        session_name: str = "threshold",
    ) -> DataFrame:
        """
        Detect sessions where a signal sustains a condition for a duration.

        Example: velocity > 0.3 for at least 2 seconds = movement session.

        Args:
            df: Input DataFrame
            signal_type: Signal to monitor
            threshold_condition: Condition (e.g., "> 0.3")
            sustained_seconds: Minimum duration in seconds
            session_name: Name prefix

        Returns:
            DataFrame with threshold-based sessions
        """
        entity_key = self.config.entity_key
        timestamp_col = self.config.timestamp_col
        signal_col = self.config.signal_col
        value_col = self.config.value_col

        session_col = f"{session_name}_session_id"
        sustained_ms = int(sustained_seconds * 1000)

        # Window for time-based aggregation
        window_time = (
            Window.partitionBy(entity_key)
            .orderBy(F.col(timestamp_col).cast("long"))
            .rangeBetween(-sustained_ms, 0)
        )
        window_seq = Window.partitionBy(entity_key).orderBy(timestamp_col)

        result = (
            df
            # Filter to target signal
            .withColumn(
                "_is_target", F.col(signal_col) == signal_type
            )
            # Check if condition is met
            .withColumn(
                "_condition_met",
                F.when(
                    F.col("_is_target"),
                    self._parse_condition(value_col, threshold_condition),
                ).otherwise(False),
            )
            # Calculate sustained duration (simplified: count matching samples)
            .withColumn(
                "_sustained_count",
                F.sum(F.when(F.col("_condition_met"), 1).otherwise(0)).over(window_time),
            )
            # Threshold met if enough samples in window
            # (assumes roughly consistent sampling rate)
            .withColumn(
                "_threshold_active",
                F.col("_sustained_count") >= (sustained_seconds * 10),  # 10 samples/sec
            )
            # Detect transitions
            .withColumn("_prev_active", F.lag("_threshold_active").over(window_seq))
            .withColumn(
                "_session_start",
                F.col("_threshold_active") & ~F.coalesce(F.col("_prev_active"), F.lit(False)),
            )
            # Assign session IDs
            .withColumn(
                session_col,
                F.when(
                    F.col("_threshold_active"),
                    F.concat(
                        F.col(entity_key),
                        F.lit(f"_{session_name}_"),
                        F.sum(F.when(F.col("_session_start"), 1).otherwise(0)).over(window_seq),
                    ),
                ).otherwise(None),
            )
            .drop(
                "_is_target",
                "_condition_met",
                "_sustained_count",
                "_threshold_active",
                "_prev_active",
                "_session_start",
            )
        )

        return result

    def summarize_sessions(
        self,
        df: DataFrame,
        session_col: str,
        agg_cols: Optional[list[tuple[str, str]]] = None,
    ) -> DataFrame:
        """
        Aggregate session-level statistics.

        Args:
            df: DataFrame with session IDs
            session_col: Column containing session IDs
            agg_cols: List of (column, agg_func) tuples, e.g., [("value", "sum")]

        Returns:
            Session summary DataFrame
        """
        entity_key = self.config.entity_key
        timestamp_col = self.config.timestamp_col

        aggs = [
            F.min(timestamp_col).alias("session_start"),
            F.max(timestamp_col).alias("session_end"),
            F.count("*").alias("event_count"),
        ]

        if agg_cols:
            for col, func in agg_cols:
                if func == "sum":
                    aggs.append(F.sum(col).alias(f"{col}_sum"))
                elif func == "avg":
                    aggs.append(F.avg(col).alias(f"{col}_avg"))
                elif func == "max":
                    aggs.append(F.max(col).alias(f"{col}_max"))
                elif func == "min":
                    aggs.append(F.min(col).alias(f"{col}_min"))

        result = (
            df.filter(F.col(session_col).isNotNull())
            .groupBy(entity_key, session_col)
            .agg(*aggs)
            .withColumn(
                "session_duration_seconds",
                F.col("session_end").cast("long") - F.col("session_start").cast("long"),
            )
        )

        return result

    def _parse_condition(self, col_name: str, condition: str) -> F.Column:
        """
        Parse a simple condition string into a PySpark Column expression.

        Supports: >, <, >=, <=, ==, !=
        Examples: "> 0.5", "== 1.0", "<= 100"

        For MVP, this is intentionally simple. Full DSL comes later.
        """
        condition = condition.strip()

        if condition.startswith(">="):
            value = float(condition[2:].strip())
            return F.col(col_name) >= value
        elif condition.startswith("<="):
            value = float(condition[2:].strip())
            return F.col(col_name) <= value
        elif condition.startswith("=="):
            value = float(condition[2:].strip())
            return F.col(col_name) == value
        elif condition.startswith("!="):
            value = float(condition[2:].strip())
            return F.col(col_name) != value
        elif condition.startswith(">"):
            value = float(condition[1:].strip())
            return F.col(col_name) > value
        elif condition.startswith("<"):
            value = float(condition[1:].strip())
            return F.col(col_name) < value
        else:
            raise ValueError(f"Unsupported condition format: {condition}")


def detect_sessions(
    df: DataFrame,
    entity_key: str,
    timestamp_col: str,
    gap_threshold_seconds: float = 300.0,
) -> SessionResult:
    """
    Convenience function for simple gap-based session detection.

    This is the simplest form of session detection - useful for quick demos.

    Args:
        df: Input DataFrame
        entity_key: Column to partition by
        timestamp_col: Timestamp column
        gap_threshold_seconds: Gap threshold to start new session

    Returns:
        SessionResult with detected sessions
    """
    config = SessionConfig(
        entity_key=entity_key,
        timestamp_col=timestamp_col,
        signal_col="signal_type",
        value_col="signal_value",
    )

    detector = SessionDetector(config)
    events_with_sessions = detector.detect_gap_sessions(
        df, gap_threshold_seconds, session_name="activity"
    )

    session_summary = detector.summarize_sessions(
        events_with_sessions, "activity_session_id"
    )

    return SessionResult(
        events_with_sessions=events_with_sessions,
        session_summary=session_summary,
    )
