# HSM-PySpark

**Hierarchical State Machine for Event Detection in PySpark**

A lightweight, YAML-configurable framework for detecting complex temporal events in IoT/sensor time-series data at scale using pure PySpark (no UDFs).

## The Problem

Detecting sessions and events in time-series data is common:
- IoT device sessions
- User activity sessions
- Equipment operational cycles
- Robotics task sequences

Existing solutions are limited:
- **pandas/polars**: Single-machine, doesn't scale
- **Spark UDFs**: Performance bottleneck at petabyte scale
- **Custom SQL**: Hard to maintain, not reusable

## The Solution

HSM-PySpark compiles YAML session definitions into pure PySpark operations:
- **Window functions** for temporal patterns
- **Catalyst optimizer** fully utilized
- **Hierarchical sessions** (parent-child relationships)
- **Petabyte scale** without UDF overhead

```yaml
# Define a session in YAML
states:
  movement_session:
    detection_type: "threshold"
    signal: "velocity"
    entry_condition:
      threshold: "> 0.3"
      sustained_duration: "2s"
```

```python
# Detect sessions with pure PySpark
from hsm_pyspark.spark import detect_sessions

result = detect_sessions(df, entity_key="robot_id", timestamp_col="timestamp")
print(f"Found {result.session_summary.count()} sessions")
```

## Quick Start

### Prerequisites
- Python 3.10+
- Java 11+ (for Spark)
- Docker (optional, for cluster mode)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/hsm-pyspark.git
cd hsm-pyspark

# Install dependencies
pip install -e ".[dev,notebook]"
```

### Run the Demo (Local Mode)

```bash
# Generate synthetic robotics data and detect sessions
python scripts/run_demo.py --local --robots 5 --hours 1
```

Expected output:
```
================================================================
SESSION DETECTION RESULTS
================================================================

Total Events Processed:     1,800,000
Unique Robots:              5
----------------------------------------------------------------
Activity Sessions (L0):     15
Movement Sessions (L1):     342
----------------------------------------------------------------
Processing Time:            12.34s
Throughput:                 145,867 events/sec
================================================================
```

### Run with Docker Spark Cluster

```bash
# Start Spark cluster
docker-compose -f docker/docker-compose.yaml up -d

# View Spark UI at http://localhost:8080

# Run demo on cluster
python scripts/run_demo.py --robots 20 --hours 24

# Stop cluster
docker-compose -f docker/docker-compose.yaml down
```

## Project Structure

```
hsm-pyspark/
├── hsm_pyspark/
│   ├── config/          # YAML config loading
│   ├── core/            # State machine logic (future)
│   └── spark/           # PySpark session detection
├── configs/
│   └── demo.yaml        # Example configuration
├── benchmarks/
│   └── data_generator/  # Synthetic data generation
├── docker/
│   └── docker-compose.yaml
├── scripts/
│   └── run_demo.py      # Demo runner
└── tests/
```

## How It Works

### Session Detection Strategy

Traditional state machines are sequential, but PySpark needs declarative operations. HSM-PySpark transforms session patterns into window functions:

```python
# Gap-based sessions: New session after 5 min inactivity
.withColumn("_gap_seconds",
    F.col("timestamp").cast("long") - F.lag("timestamp").cast("long").over(window))
.withColumn("_is_new_session", F.col("_gap_seconds") > 300)
.withColumn("session_id",
    F.sum(F.when(F.col("_is_new_session"), 1).otherwise(0)).over(window))
```

### Hierarchical Sessions

```
Level 0: Activity Session (gap-based)
    └── Level 1: Movement Session (velocity threshold)
            └── Level 2: Micro-movement (status changes)
```

Parent-child relationships are linked via temporal joins.

## Configuration

See `configs/demo.yaml` for full example:

```yaml
states:
  activity_session:
    level: 0
    detection_type: "gap"
    gap_threshold: "5min"

  movement_session:
    level: 1
    parent: "activity_session"
    detection_type: "threshold"
    signal: "velocity"
    entry_condition:
      threshold: "> 0.3"
      sustained_duration: "2s"
```

## Roadmap

- [x] MVP: Gap-based session detection
- [x] MVP: Threshold-based sessions
- [x] Synthetic robotics data generator
- [x] Docker Spark setup
- [ ] Full YAML DSL parser
- [ ] 3-level hierarchy linking
- [ ] Orphan session handling
- [ ] Incremental processing
- [ ] Benchmarking suite
- [ ] Cloud deployment examples

## Contributing

Contributions welcome! See `IMPLEMENTATION_PLAN.md` for the full roadmap.

## License

MIT
