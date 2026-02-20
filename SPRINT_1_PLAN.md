# Sprint 1: MVP Demo

## Goal
Get a working end-to-end demo of simple session detection running locally on MacBook with multi-GB data.

**Definition of Done**: Run a PySpark job that detects sessions from an open-source dataset and outputs results.

---

## Tasks

### 1. Project Setup (Day 1 morning)

```bash
# Project structure (minimal)
hsm-pyspark/
├── hsm_pyspark/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── loader.py           # YAML loading (minimal)
│   └── spark/
│       ├── __init__.py
│       └── session_detector.py # Core demo logic
├── configs/
│   └── demo.yaml               # Simple session config
├── data/                       # Downloaded dataset
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yaml
├── notebooks/
│   └── demo.ipynb             # Quick visualization
├── scripts/
│   ├── download_data.sh
│   └── run_demo.py
├── pyproject.toml
├── README.md
└── .gitignore
```

**Git Setup**:
```bash
git init
git remote add origin <your-private-repo>
# .gitignore includes: data/, __pycache__/, .spark-warehouse/, etc.
```

---

### 2. Dataset Selection

**Candidates (multi-GB, session-friendly)**:

| Dataset | Size | Sessions? | Signal Types | Pros/Cons |
|---------|------|-----------|--------------|-----------|
| **NYC Taxi (2019)** | ~8 GB/month | Trip start/end | pickup, dropoff, fare | Easy sessions, well-known |
| **DEBS 2012 GC** | ~2 GB | Machine operation | power, temp, vibration | Industrial IoT, realistic |
| **Azure Predictive Maintenance** | ~1 GB | Equipment cycles | sensors, errors | Clear maintenance sessions |
| **WESAD** | ~3 GB | Activity sessions | heart rate, accel, temp | Wearable IoT |
| **Power Consumption** | ~500 MB | Daily patterns | power, voltage, current | Simple but good for demo |

**Recommendation**: **NYC Taxi 2019**
- Parquet format (no conversion needed)
- Clear session structure (trip = session)
- 8 GB/month = good benchmark size
- Widely understood domain
- Can add complexity: rider sessions across trips, surge detection

**Alternative**: DEBS 2012 for more "industrial IoT" feel (closer to robotics).

---

### 3. Docker Spark Setup

```yaml
# docker/docker-compose.yaml
version: '3.8'

services:
  spark-master:
    image: bitnami/spark:3.5
    environment:
      - SPARK_MODE=master
      - SPARK_RPC_AUTHENTICATION_ENABLED=no
      - SPARK_RPC_ENCRYPTION_ENABLED=no
    ports:
      - "8080:8080"   # Spark UI
      - "7077:7077"   # Master port
      - "4040:4040"   # Job UI
    volumes:
      - ../data:/data
      - ../configs:/configs
      - ../hsm_pyspark:/app/hsm_pyspark
      - ../scripts:/app/scripts

  spark-worker:
    image: bitnami/spark:3.5
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
      - SPARK_WORKER_MEMORY=4G
      - SPARK_WORKER_CORES=2
    depends_on:
      - spark-master
    volumes:
      - ../data:/data
```

**Local (no Docker) fallback**:
```python
# For quick iteration on MacBook
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("hsm-demo") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()
```

---

### 4. Minimal Session Detection Logic

**Demo Config** (`configs/demo.yaml`):
```yaml
version: "1.0"
name: "taxi_trip_sessions"

schema:
  entity_key: "medallion"        # Taxi ID
  timestamp_col: "pickup_datetime"

states:
  trip_session:
    level: 0
    entry_condition:
      signal: "event_type"
      value: "pickup"
    exit_condition:
      signal: "event_type"
      value: "dropoff"
```

**Core Logic** (`hsm_pyspark/spark/session_detector.py`):
```python
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def detect_simple_sessions(
    df: DataFrame,
    entity_key: str,
    timestamp_col: str,
    session_start_condition,  # PySpark Column
    session_end_condition,    # PySpark Column
) -> DataFrame:
    """
    Simplest possible session detection using window functions.
    No YAML parsing, no DSL - just demonstrate the core pattern.
    """
    window = Window.partitionBy(entity_key).orderBy(timestamp_col)

    return (
        df
        # Mark session boundaries
        .withColumn("is_session_start", session_start_condition)
        .withColumn("is_session_end", session_end_condition)

        # Assign session IDs using cumsum trick
        .withColumn(
            "session_id",
            F.concat(
                F.col(entity_key),
                F.lit("_"),
                F.sum(F.when(F.col("is_session_start"), 1).otherwise(0)).over(window)
            )
        )

        # Calculate session metrics
        .withColumn("prev_timestamp", F.lag(timestamp_col).over(window))
        .withColumn("time_in_session",
            F.col(timestamp_col).cast("long") -
            F.first(timestamp_col).over(
                Window.partitionBy(entity_key, "session_id").orderBy(timestamp_col)
            ).cast("long")
        )
    )


def summarize_sessions(df: DataFrame, entity_key: str) -> DataFrame:
    """Aggregate session-level statistics."""
    return (
        df
        .groupBy(entity_key, "session_id")
        .agg(
            F.min("pickup_datetime").alias("session_start"),
            F.max("dropoff_datetime").alias("session_end"),
            F.count("*").alias("event_count"),
            F.sum("trip_distance").alias("total_distance"),
        )
        .withColumn(
            "session_duration_minutes",
            (F.col("session_end").cast("long") - F.col("session_start").cast("long")) / 60
        )
    )
```

---

### 5. Run Script

```python
# scripts/run_demo.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# For now, inline. Later: from hsm_pyspark.spark.session_detector import ...


def main():
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("hsm-demo") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

    # Load NYC taxi data (parquet)
    df = spark.read.parquet("/data/nyc_taxi_2019_01.parquet")

    print(f"Loaded {df.count():,} records")
    df.printSchema()

    # Simple session detection
    # For taxi: each trip IS a session (pickup to dropoff)
    # But let's detect "driving sessions" per taxi (consecutive trips)

    window = Window.partitionBy("medallion").orderBy("pickup_datetime")

    sessions = (
        df
        # Gap detection: if > 4 hours between trips, new session
        .withColumn("prev_dropoff", F.lag("dropoff_datetime").over(window))
        .withColumn("gap_hours",
            (F.col("pickup_datetime").cast("long") -
             F.col("prev_dropoff").cast("long")) / 3600
        )
        .withColumn("is_new_session",
            (F.col("gap_hours") > 4) | F.col("gap_hours").isNull()
        )
        .withColumn("session_id",
            F.concat(
                F.col("medallion"),
                F.lit("_"),
                F.sum(F.when(F.col("is_new_session"), 1).otherwise(0)).over(window)
            )
        )
    )

    # Session summary
    summary = (
        sessions
        .groupBy("medallion", "session_id")
        .agg(
            F.count("*").alias("trips_in_session"),
            F.min("pickup_datetime").alias("session_start"),
            F.max("dropoff_datetime").alias("session_end"),
            F.sum("trip_distance").alias("total_distance"),
            F.sum("total_amount").alias("total_revenue"),
        )
        .withColumn("session_duration_hours",
            (F.col("session_end").cast("long") -
             F.col("session_start").cast("long")) / 3600
        )
    )

    print("\n=== SESSION DETECTION RESULTS ===")
    print(f"Total taxis: {sessions.select('medallion').distinct().count():,}")
    print(f"Total sessions: {summary.count():,}")

    summary.show(20, truncate=False)

    # Save results
    summary.write.mode("overwrite").parquet("/data/output/sessions")

    spark.stop()


if __name__ == "__main__":
    main()
```

---

### 6. Data Treatment Notes (for expert review)

**Forward Fill Decisions**:
| Signal Type | Forward Fill? | Max Duration | Notes |
|------------|---------------|--------------|-------|
| Position/GPS | Yes | 5 min | Interpolate if stationary |
| Sensor readings | Yes | 1 min | Last known value |
| Status flags | Yes | Until next status | State persists |
| Heartbeat | No | - | Missing = anomaly |

**Multi-day Sessions**:
- Maximum session duration: Configurable (default 24h)
- Sessions spanning midnight: Handle via timestamp, not date partition
- Orphan detection window: N days lookback (configurable)

---

## Deliverables

1. **Git repo** initialized with structure above
2. **Docker compose** for local Spark cluster
3. **Data download script** for NYC Taxi (or chosen dataset)
4. **Working demo** that outputs detected sessions
5. **README** with setup instructions

## Success Metrics

- [ ] Load 1+ GB of data locally
- [ ] Process in under 2 minutes
- [ ] Output session statistics
- [ ] Spark UI accessible at localhost:8080

---

## Next Steps (Sprint 2 Preview)

- YAML config loader
- Pattern DSL compiler
- Multiple pattern types (threshold, sequence)
- 2-level hierarchy
- Exception handling (orphans)
