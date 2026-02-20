"""
Synthetic Robotics Sensor Data Generator

Generates realistic IoT sensor data for a robotic arm system with hierarchical sessions:
- Level 0: Connection sessions (auth/connection lifecycle)
- Level 1: Task sessions (pick-and-place operations)
- Level 2: Micro-movement sessions (approach, grasp, lift, move, place, release)

Data schema matches typical embodied AI / robotics telemetry.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterator
import numpy as np


@dataclass
class SensorReading:
    """Single sensor reading from robotic system."""

    robot_id: str
    timestamp: datetime
    signal_type: str
    signal_value: float
    # Aggregations over the sample window
    value_min: float
    value_max: float
    value_first: float
    value_last: float
    # Raw sub-millisecond samples (simplified as list of dicts)
    raw_samples: list = field(default_factory=list)


@dataclass
class SessionMarker:
    """Tracks current session state for generation."""

    connection_active: bool = False
    connection_id: int = 0
    task_active: bool = False
    task_id: int = 0
    micro_active: bool = False
    micro_id: int = 0
    micro_type: str = ""


class RoboticsDataGenerator:
    """
    Generates synthetic robotics sensor data with embedded session patterns.

    The generator creates data for multiple robots over a time period,
    with realistic session hierarchies that can be detected by HSM.
    """

    SIGNAL_TYPES = [
        "position_x",
        "position_y",
        "position_z",
        "velocity",
        "force",
        "torque",
        "gripper_position",
        "temperature",
        "heartbeat",
        "status",
    ]

    MICRO_MOVEMENT_TYPES = ["approach", "grasp", "lift", "move", "place", "release"]

    def __init__(
        self,
        num_robots: int = 10,
        duration_hours: int = 24,
        samples_per_second: float = 10.0,
        anomaly_rate: float = 0.001,
        gap_rate: float = 0.005,
        seed: int = 42,
    ):
        self.num_robots = num_robots
        self.duration_hours = duration_hours
        self.samples_per_second = samples_per_second
        self.anomaly_rate = anomaly_rate
        self.gap_rate = gap_rate
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def generate(self) -> Iterator[dict]:
        """
        Generate sensor readings as dictionaries (for Spark DataFrame creation).

        Yields dictionaries with sensor readings including session context.
        """
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        interval_ms = int(1000 / self.samples_per_second)

        for robot_idx in range(self.num_robots):
            robot_id = f"ROBOT_{robot_idx:04d}"
            yield from self._generate_robot_timeline(robot_id, start_time, interval_ms)

    def _generate_robot_timeline(
        self, robot_id: str, start_time: datetime, interval_ms: int
    ) -> Iterator[dict]:
        """Generate timeline of events for a single robot."""
        total_samples = int(self.duration_hours * 3600 * self.samples_per_second)
        session = SessionMarker()

        current_time = start_time
        sample_idx = 0

        while sample_idx < total_samples:
            # Check for data gaps (simulate network issues)
            if self.rng.random() < self.gap_rate:
                gap_duration = self.rng.integers(5, 60)  # 5-60 samples gap
                current_time += timedelta(milliseconds=interval_ms * gap_duration)
                sample_idx += gap_duration
                continue

            # Session state transitions
            session = self._update_session_state(session, sample_idx, total_samples)

            # Generate sensor readings for this timestamp
            readings = self._generate_sensor_readings(
                robot_id, current_time, session, sample_idx
            )

            for reading in readings:
                yield reading

            current_time += timedelta(milliseconds=interval_ms)
            sample_idx += 1

    def _update_session_state(
        self, session: SessionMarker, sample_idx: int, total_samples: int
    ) -> SessionMarker:
        """Update session states based on probabilistic transitions."""
        # Connection session (Level 0) - long running, occasional reconnects
        if not session.connection_active:
            if self.rng.random() < 0.01:  # 1% chance to connect
                session.connection_active = True
                session.connection_id += 1
        else:
            # Disconnect every ~2 hours on average
            if self.rng.random() < 0.0001:
                session.connection_active = False
                session.task_active = False
                session.micro_active = False
                return session

        # Task session (Level 1) - pick-and-place cycles
        if session.connection_active and not session.task_active:
            if self.rng.random() < 0.05:  # Start new task
                session.task_active = True
                session.task_id += 1
        elif session.task_active:
            # End task after completing micro-movements or randomly
            if self.rng.random() < 0.01:
                session.task_active = False
                session.micro_active = False

        # Micro-movement session (Level 2)
        if session.task_active and not session.micro_active:
            if self.rng.random() < 0.1:  # Start micro-movement
                session.micro_active = True
                session.micro_id += 1
                session.micro_type = random.choice(self.MICRO_MOVEMENT_TYPES)
        elif session.micro_active:
            if self.rng.random() < 0.15:  # End micro-movement
                session.micro_active = False

        return session

    def _generate_sensor_readings(
        self,
        robot_id: str,
        timestamp: datetime,
        session: SessionMarker,
        sample_idx: int,
    ) -> list[dict]:
        """Generate sensor readings for current timestamp."""
        readings = []
        is_anomaly = self.rng.random() < self.anomaly_rate

        # Base values depend on session state
        if session.micro_active:
            base_velocity = 0.5 + self.rng.random() * 0.5
            base_force = 10 + self.rng.random() * 20
        elif session.task_active:
            base_velocity = 0.1 + self.rng.random() * 0.2
            base_force = 5 + self.rng.random() * 10
        else:
            base_velocity = self.rng.random() * 0.05
            base_force = self.rng.random() * 2

        # Position (smooth trajectory with noise)
        phase = sample_idx * 0.01
        position_x = 100 + 50 * np.sin(phase) + self.rng.normal(0, 0.1)
        position_y = 100 + 50 * np.cos(phase) + self.rng.normal(0, 0.1)
        position_z = 50 + 20 * np.sin(phase * 0.5) + self.rng.normal(0, 0.1)

        if is_anomaly:
            position_x = 9999.0  # Impossible value for anomaly detection
            base_force = 999.0

        signal_data = {
            "position_x": position_x,
            "position_y": position_y,
            "position_z": position_z,
            "velocity": base_velocity + self.rng.normal(0, 0.02),
            "force": base_force + self.rng.normal(0, 1),
            "torque": base_force * 0.1 + self.rng.normal(0, 0.1),
            "gripper_position": 0.8 if session.micro_type == "grasp" else 0.2,
            "temperature": 45 + self.rng.normal(0, 2),
            "heartbeat": 1.0,  # Always 1 when sending data
            "status": self._get_status_code(session),
        }

        for signal_type, value in signal_data.items():
            # Generate raw sub-millisecond samples (simplified)
            raw_samples = [
                {"ts_offset_us": i * 100, "value": value + self.rng.normal(0, 0.01)}
                for i in range(10)
            ]
            raw_values = [s["value"] for s in raw_samples]

            readings.append(
                {
                    "robot_id": robot_id,
                    "timestamp": timestamp.isoformat(),
                    "signal_type": signal_type,
                    "signal_value": float(value),
                    "value_min": float(min(raw_values)),
                    "value_max": float(max(raw_values)),
                    "value_first": float(raw_values[0]),
                    "value_last": float(raw_values[-1]),
                    # Session context (for validation, not used in detection)
                    "_connection_id": session.connection_id if session.connection_active else None,
                    "_task_id": session.task_id if session.task_active else None,
                    "_micro_id": session.micro_id if session.micro_active else None,
                    "_micro_type": session.micro_type if session.micro_active else None,
                    "_is_anomaly": is_anomaly,
                }
            )

        return readings

    def _get_status_code(self, session: SessionMarker) -> float:
        """Return numeric status code based on session state."""
        if not session.connection_active:
            return 0.0  # Disconnected
        if not session.task_active:
            return 1.0  # Idle
        if not session.micro_active:
            return 2.0  # Task active, no micro-movement

        # Micro-movement status codes
        status_map = {
            "approach": 10.0,
            "grasp": 20.0,
            "lift": 30.0,
            "move": 40.0,
            "place": 50.0,
            "release": 60.0,
        }
        return status_map.get(session.micro_type, 99.0)


def generate_to_parquet(
    output_path: str,
    num_robots: int = 10,
    duration_hours: int = 24,
    samples_per_second: float = 10.0,
    partition_by_date: bool = True,
):
    """
    Generate data and write to Parquet using PySpark.

    Args:
        output_path: Directory to write parquet files
        num_robots: Number of robots to simulate
        duration_hours: Duration of simulation
        samples_per_second: Sampling rate
        partition_by_date: Whether to partition output by date
    """
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        DoubleType,
        BooleanType,
        IntegerType,
        TimestampType,
    )

    spark = SparkSession.builder.appName("RoboticsDataGenerator").getOrCreate()

    schema = StructType(
        [
            StructField("robot_id", StringType(), False),
            StructField("timestamp", StringType(), False),  # ISO format, convert later
            StructField("signal_type", StringType(), False),
            StructField("signal_value", DoubleType(), False),
            StructField("value_min", DoubleType(), False),
            StructField("value_max", DoubleType(), False),
            StructField("value_first", DoubleType(), False),
            StructField("value_last", DoubleType(), False),
            StructField("_connection_id", IntegerType(), True),
            StructField("_task_id", IntegerType(), True),
            StructField("_micro_id", IntegerType(), True),
            StructField("_micro_type", StringType(), True),
            StructField("_is_anomaly", BooleanType(), False),
        ]
    )

    generator = RoboticsDataGenerator(
        num_robots=num_robots,
        duration_hours=duration_hours,
        samples_per_second=samples_per_second,
    )

    # Generate in batches for memory efficiency
    batch_size = 100_000
    batch = []
    batch_num = 0

    for record in generator.generate():
        batch.append(record)
        if len(batch) >= batch_size:
            df = spark.createDataFrame(batch, schema)
            mode = "overwrite" if batch_num == 0 else "append"
            df.write.mode(mode).parquet(output_path)
            batch = []
            batch_num += 1
            print(f"Written batch {batch_num} ({batch_size:,} records)")

    # Write remaining
    if batch:
        df = spark.createDataFrame(batch, schema)
        df.write.mode("append").parquet(output_path)
        print(f"Written final batch ({len(batch):,} records)")

    spark.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic robotics data")
    parser.add_argument("--output", "-o", default="./data/robotics_sensors", help="Output path")
    parser.add_argument("--robots", "-r", type=int, default=10, help="Number of robots")
    parser.add_argument("--hours", "-H", type=int, default=24, help="Duration in hours")
    parser.add_argument("--rate", type=float, default=10.0, help="Samples per second")
    args = parser.parse_args()

    print(f"Generating data for {args.robots} robots over {args.hours} hours...")
    print(f"Expected records: ~{args.robots * args.hours * 3600 * args.rate * 10:,.0f}")

    generate_to_parquet(
        output_path=args.output,
        num_robots=args.robots,
        duration_hours=args.hours,
        samples_per_second=args.rate,
    )
