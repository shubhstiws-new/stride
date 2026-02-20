#!/usr/bin/env python3
"""
HSM-PySpark Demo: Session Detection on Robotics Sensor Data

This script demonstrates the core session detection capability:
1. Generates synthetic robotics sensor data (or loads existing)
2. Detects activity sessions using gap-based detection
3. Detects movement sessions using threshold-based detection
4. Outputs session statistics

Run locally:
    python scripts/run_demo.py --local --robots 5 --hours 1

Run with Docker Spark:
    docker-compose -f docker/docker-compose.yaml up -d
    python scripts/run_demo.py --robots 10 --hours 24
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_spark_session(local: bool = True, app_name: str = "HSM-PySpark-Demo"):
    """Create Spark session for local or cluster mode."""
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name)

    if local:
        builder = (
            builder.master("local[*]")
            .config("spark.driver.memory", "8g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.ui.showConsoleProgress", "true")
        )
    else:
        builder = builder.master("spark://spark-master:7077")

    return builder.getOrCreate()


def generate_sample_data(spark, num_robots: int, duration_hours: int, output_path: str):
    """Generate synthetic robotics data."""
    from benchmarks.data_generator.robotics_generator import RoboticsDataGenerator
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        DoubleType,
        BooleanType,
        IntegerType,
    )

    print(f"\n{'='*60}")
    print(f"Generating data for {num_robots} robots over {duration_hours} hours...")
    expected_records = num_robots * duration_hours * 3600 * 10 * 10  # 10 signals, 10 samples/sec
    print(f"Expected records: ~{expected_records:,.0f}")
    print(f"{'='*60}\n")

    schema = StructType(
        [
            StructField("robot_id", StringType(), False),
            StructField("timestamp", StringType(), False),
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
        samples_per_second=10.0,
    )

    # Generate data in batches
    batch_size = 50_000
    batch = []
    all_batches = []

    start_time = time.time()
    for record in generator.generate():
        batch.append(record)
        if len(batch) >= batch_size:
            all_batches.extend(batch)
            print(f"  Generated {len(all_batches):,} records...")
            batch = []

    all_batches.extend(batch)
    gen_time = time.time() - start_time
    print(f"\nGenerated {len(all_batches):,} records in {gen_time:.1f}s")

    # Create DataFrame
    df = spark.createDataFrame(all_batches, schema)

    # Convert timestamp string to timestamp type
    from pyspark.sql import functions as F

    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))

    # Save to parquet
    print(f"Writing to {output_path}...")
    df.write.mode("overwrite").parquet(output_path)

    return df


def run_session_detection(spark, data_path: str):
    """Run session detection on robotics data."""
    from pyspark.sql import functions as F
    from hsm_pyspark.spark.session_detector import SessionDetector, SessionConfig

    print(f"\n{'='*60}")
    print("Loading data...")
    print(f"{'='*60}\n")

    df = spark.read.parquet(data_path)
    total_records = df.count()
    unique_robots = df.select("robot_id").distinct().count()

    print(f"Loaded {total_records:,} records from {unique_robots} robots")
    print("\nSchema:")
    df.printSchema()

    # Configure session detector
    config = SessionConfig(
        entity_key="robot_id",
        timestamp_col="timestamp",
        signal_col="signal_type",
        value_col="signal_value",
    )
    detector = SessionDetector(config)

    print(f"\n{'='*60}")
    print("Detecting sessions...")
    print(f"{'='*60}\n")

    start_time = time.time()

    # 1. Gap-based activity sessions (300 second gap = new session)
    print("1. Detecting activity sessions (gap-based, 5 min threshold)...")
    df_with_activity = detector.detect_gap_sessions(
        df, gap_threshold_seconds=300.0, session_name="activity"
    )

    # 2. Threshold-based movement sessions (velocity > 0.3 for 2+ seconds)
    print("2. Detecting movement sessions (velocity threshold)...")
    df_with_movement = detector.detect_threshold_sessions(
        df_with_activity,
        signal_type="velocity",
        threshold_condition="> 0.3",
        sustained_seconds=2.0,
        session_name="movement",
    )

    # 3. Summarize sessions
    print("3. Computing session summaries...")

    activity_summary = detector.summarize_sessions(
        df_with_movement,
        "activity_session_id",
        agg_cols=[("signal_value", "avg")],
    )

    movement_summary = detector.summarize_sessions(
        df_with_movement.filter(F.col("movement_session_id").isNotNull()),
        "movement_session_id",
        agg_cols=[("signal_value", "max")],
    )

    # Force computation
    activity_count = activity_summary.count()
    movement_count = movement_summary.count()

    detection_time = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print("SESSION DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"\nTotal Events Processed:     {total_records:,}")
    print(f"Unique Robots:              {unique_robots}")
    print(f"{'-'*60}")
    print(f"Activity Sessions (L0):     {activity_count:,}")
    print(f"Movement Sessions (L1):     {movement_count:,}")
    print(f"{'-'*60}")
    print(f"Processing Time:            {detection_time:.2f}s")
    print(f"Throughput:                 {total_records/detection_time:,.0f} events/sec")
    print(f"{'='*60}\n")

    # Show sample sessions
    print("Sample Activity Sessions:")
    activity_summary.orderBy("session_start").show(10, truncate=False)

    print("\nSample Movement Sessions:")
    movement_summary.orderBy("session_start").show(10, truncate=False)

    # Check for anomalies
    anomaly_count = df.filter(F.col("_is_anomaly") == True).count()
    if anomaly_count > 0:
        print(f"\nAnomalies detected: {anomaly_count:,}")
        df.filter(F.col("_is_anomaly") == True).show(5)

    return {
        "total_records": total_records,
        "activity_sessions": activity_count,
        "movement_sessions": movement_count,
        "processing_time": detection_time,
        "throughput": total_records / detection_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="HSM-PySpark Demo: Session Detection on Robotics Data"
    )
    parser.add_argument(
        "--local", action="store_true", help="Run in local mode (default: cluster)"
    )
    parser.add_argument(
        "--robots", "-r", type=int, default=5, help="Number of robots to simulate"
    )
    parser.add_argument(
        "--hours", "-H", type=int, default=1, help="Duration in hours"
    )
    parser.add_argument(
        "--data-path",
        "-d",
        default="./data/robotics_demo",
        help="Path for data storage",
    )
    parser.add_argument(
        "--skip-generate", action="store_true", help="Skip data generation, use existing"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("HSM-PySpark: Hierarchical State Machine for Event Detection")
    print("=" * 60)
    print(f"Mode: {'Local' if args.local else 'Cluster'}")
    print(f"Robots: {args.robots}")
    print(f"Duration: {args.hours} hours")
    print(f"Data path: {args.data_path}")
    print("=" * 60 + "\n")

    # Create Spark session
    spark = create_spark_session(local=args.local)
    spark.sparkContext.setLogLevel("WARN")

    try:
        # Generate or load data
        if not args.skip_generate:
            generate_sample_data(
                spark,
                num_robots=args.robots,
                duration_hours=args.hours,
                output_path=args.data_path,
            )

        # Run session detection
        results = run_session_detection(spark, args.data_path)

        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print(f"Results: {results}")
        print("\nNext steps:")
        print("  - View Spark UI at http://localhost:4040 (if still running)")
        print("  - Check generated sessions in ./data/robotics_demo")
        print("  - Modify configs/demo.yaml for different session patterns")
        print("=" * 60 + "\n")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
