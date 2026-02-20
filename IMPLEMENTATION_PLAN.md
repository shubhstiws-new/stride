# HSM-PySpark: Hierarchical State Machine for Event Detection

## Overview

A lightweight, YAML-configurable hierarchical state machine framework for detecting complex temporal events in IoT/sensor time-series data at scale using PySpark.

**Gap Addressed**: No existing PySpark implementation for hierarchical state machine event detection (only pandas/polars equivalents exist).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HSM-PySpark Framework                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│   │    YAML     │    │   Schema    │    │    HSM      │    │   PySpark   │  │
│   │   Config    │ ─→ │  Validator  │ ─→ │   Engine    │ ─→ │  Executor   │  │
│   │   Loader    │    │  + Parser   │    │   (Core)    │    │  (Runtime)  │  │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                   │                  │                  │         │
│         ▼                   ▼                  ▼                  ▼         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Event Pattern DSL Engine                          │   │
│   │  • Temporal patterns (X units for Y duration)                       │   │
│   │  • Sequence patterns (A followed by B within Z)                     │   │
│   │  • Gap detection (N gaps of signal C)                               │   │
│   │  • Exception rules (early termination, orphan handling)             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Model (IoT Robotics Example)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Raw Event Schema                                                         │
├──────────────────────────────────────────────────────────────────────────┤
│  device_id      : STRING          # Unique device identifier             │
│  timestamp      : TIMESTAMP       # Aggregated timestamp (1-level)       │
│  signal_type    : STRING          # IOT signal identifier                │
│  signal_value   : DOUBLE          # Primary signal value                 │
│  aggregations   : STRUCT          # max, min, first, last                │
│  raw_data       : ARRAY<STRUCT>   # Sub-millisecond: [{ts, value}, ...]  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Hierarchical Event Model (3-Level Demo)

```
Level 0 (Root)      ┌─────────────────────────┐
                    │   Device Session        │
                    │   (Authentication)      │
                    └───────────┬─────────────┘
                                │
Level 1 (Parent)    ┌───────────┴─────────────┐
                    │                         │
              ┌─────┴─────┐           ┌───────┴───────┐
              │ Movement  │           │  Idle State   │
              │  Session  │           │   Session     │
              └─────┬─────┘           └───────────────┘
                    │
Level 2 (Child)     ├──────────────┬──────────────┐
              ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐
              │ Micro     │  │ Direction │  │ Velocity  │
              │ Movement  │  │ Change    │  │ Spike     │
              └───────────┘  └───────────┘  └───────────┘

Exception Types:
  • Orphan Child   → Child event with no parent (data gap)
  • Orphan Parent  → Parent with no root (late arrival)
  • Anomaly        → Events outside hierarchy (corruption)
```

---

## Component Breakdown

### Phase 1: Core Foundation

#### 1.1 Project Structure
```
hsm-pyspark/
├── hsm_pyspark/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── state_machine.py      # HSM engine
│   │   ├── state.py              # State definitions
│   │   ├── transition.py         # Transition rules
│   │   └── hierarchy.py          # Parent-child relationships
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py             # YAML loading
│   │   ├── validator.py          # Schema validation
│   │   └── schema.py             # Pydantic models
│   ├── dsl/
│   │   ├── __init__.py
│   │   ├── parser.py             # Pattern DSL parser
│   │   ├── patterns.py           # Temporal/sequence patterns
│   │   └── conditions.py         # Condition evaluators
│   ├── spark/
│   │   ├── __init__.py
│   │   ├── executor.py           # Main PySpark executor
│   │   ├── session_builder.py    # Session window logic
│   │   ├── query_analyzer.py     # Query plan analysis & optimization
│   │   └── checkpointing.py      # Incremental processing
│   ├── exceptions/
│   │   ├── __init__.py
│   │   ├── handlers.py           # Orphan/anomaly handlers
│   │   └── recovery.py           # Session recovery logic
│   └── utils/
│       ├── __init__.py
│       └── time_utils.py
├── configs/
│   └── examples/
│       ├── robotics_iot.yaml     # Demo config
│       └── schema_reference.yaml # Full DSL reference
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmark/
├── benchmarks/
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yaml
│   │   └── spark-defaults.conf
│   ├── data_generator/
│   │   └── iot_generator.py
│   ├── runners/
│   │   ├── local_runner.py
│   │   └── cloud_runner.py
│   └── results/
├── docs/
│   ├── getting_started.md
│   ├── dsl_reference.md
│   └── benchmarking.md
├── pyproject.toml
├── README.md
└── LICENSE
```

#### 1.2 YAML DSL Specification
```yaml
# Example: robotics_iot.yaml
version: "1.0"
name: "robotics_session_detector"

schema:
  entity_key: "device_id"
  timestamp_col: "timestamp"
  max_hierarchy_depth: 3

states:
  # Level 0: Root state
  authenticated:
    level: 0
    entry_condition:
      pattern: "signal_match"
      signal: "auth_handshake"
      value_condition: "== 'SUCCESS'"
    exit_condition:
      any_of:
        - pattern: "signal_match"
          signal: "auth_terminate"
        - pattern: "timeout"
          duration: "24h"
    children: ["movement_session", "idle_session"]

  # Level 1: Parent states
  movement_session:
    level: 1
    parent: "authenticated"
    entry_condition:
      pattern: "sequence"
      steps:
        - signal: "motion_sensor"
          condition: "value > 0.5"
          duration: "2s"
        - signal: "accelerometer"
          condition: "value > 0.1"
          within: "500ms"
    exit_condition:
      any_of:
        - pattern: "sequence"
          steps:
            - signal: "motion_sensor"
              condition: "value < 0.1"
              duration: "5s"
        - pattern: "gap_count"
          signal: "heartbeat"
          count: 3
    children: ["micro_movement", "direction_change", "velocity_spike"]

  # Level 2: Child states
  micro_movement:
    level: 2
    parent: "movement_session"
    entry_condition:
      pattern: "threshold"
      signal: "fine_motion"
      condition: "0.01 < value < 0.1"
      duration: "100ms"
    exit_condition:
      pattern: "threshold_exit"
      condition: "value >= 0.1 OR value <= 0.01"

exceptions:
  orphan_child:
    action: "create_synthetic_parent"
    log_level: "WARNING"
  orphan_parent:
    action: "mark_incomplete"
    retention: "7d"
  data_anomaly:
    detection:
      - signal_type: "motion_sensor"
        condition: "value > 1000"  # Impossible value
    action: "quarantine"

incremental:
  enabled: true
  checkpoint_interval: "5min"
  orphan_resolution_window: "1h"

output:
  session_table: "detected_sessions"
  event_table: "session_events"
  anomaly_table: "anomalies"
  include_raw_events: false
```

---

### Phase 2: Core Engine Implementation

#### 2.1 State Machine Core (`core/state_machine.py`)
- `HierarchicalStateMachine` class
  - State registration with depth validation
  - Transition evaluation pipeline
  - Parent-child relationship enforcement
  - Event emission on state changes

#### 2.2 Pattern DSL Engine (`dsl/`)
| Pattern Type | Description | Example | PySpark Compilation |
|-------------|-------------|---------|---------------------|
| `signal_match` | Single signal condition | `signal == 'X' AND value > 5` | `F.col("signal") == "X" & F.col("value") > 5` |
| `threshold` | Value in range for duration | `0.1 < value < 0.5 for 2s` | Window sum with `rangeBetween` |
| `sequence` | Ordered signal patterns | `A then B within 500ms` | `lag()`/`lead()` with time bounds |
| `gap_count` | Missing signal detection | `3 gaps of heartbeat` | Window count of expected vs actual |
| `timeout` | Duration-based exit | `no activity for 5min` | Time diff from `lag()` |
| `any_of` / `all_of` | Compound conditions | Multiple exit conditions | `|` / `&` expression trees |

#### 2.2.1 DSL to PySpark Compiler
```python
# Core concept: YAML → AST → PySpark Column Expression
class PatternCompiler:
    """Compiles declarative patterns to PySpark expressions."""

    def compile(self, pattern: PatternConfig) -> Column:
        """Returns a PySpark Column expression (no UDFs)."""
        match pattern.type:
            case "signal_match":
                return self._compile_signal_match(pattern)
            case "threshold":
                return self._compile_threshold(pattern)
            case "sequence":
                return self._compile_sequence(pattern)
            case "gap_count":
                return self._compile_gap_count(pattern)
            case "any_of":
                return reduce(lambda a, b: a | b,
                             [self.compile(p) for p in pattern.conditions])
            case "all_of":
                return reduce(lambda a, b: a & b,
                             [self.compile(p) for p in pattern.conditions])

    def _compile_threshold(self, pattern: PatternConfig) -> Column:
        """Threshold pattern: value in range sustained for duration."""
        # Duration in milliseconds
        duration_ms = parse_duration(pattern.duration)
        condition = parse_condition(pattern.condition)  # Returns Column

        # Compile to window-based sustained check
        window = (Window
            .partitionBy("device_id")
            .orderBy(F.col("timestamp").cast("long"))
            .rangeBetween(-duration_ms, 0))

        sustained_time = F.sum(
            F.when(condition, F.col("event_duration_ms")).otherwise(0)
        ).over(window)

        return sustained_time >= duration_ms
```

#### 2.3 PySpark Executor (`spark/executor.py`)
```python
# Conceptual flow - PURE PYSPARK (no pandas UDFs)
class HSMExecutor:
    def __init__(self, config_path: str, spark: SparkSession):
        self.config = ConfigLoader.load(config_path)
        self.hsm = HierarchicalStateMachine.from_config(self.config)
        self.spark = spark

    def detect_sessions(self, df: DataFrame) -> SessionResults:
        # 1. Read from Iceberg catalog (partitioned by date)
        # 2. Window functions for temporal pattern matching
        # 3. State transitions via conditional expressions
        # 4. Session boundary detection with gap analysis
        # 5. Hierarchical session linking
        # 6. Handle orphans and anomalies
        pass
```

**Key Implementation Decisions:**
- **Pure PySpark ONLY** - No pandas UDFs (petabyte-scale optimization)
- Window functions for temporal patterns (`lag`, `lead`, `sum over window`)
- Catalyst optimizer fully utilized - no UDF blackbox
- State machine compiled to PySpark expression trees
- Iceberg catalog as source (daily batch, UTC schedule)
- Query plan analysis built-in for performance tuning

#### 2.4 Pure PySpark State Machine Strategy
```
Challenge: State machines are inherently sequential
Solution: Transform to declarative pattern matching

┌─────────────────────────────────────────────────────────────────┐
│  YAML Pattern Definition                                        │
│  ─────────────────────                                          │
│  entry: signal='motion' AND value > 0.5 FOR 2s                  │
│                                                                  │
│                        ▼ Compile to                              │
│                                                                  │
│  PySpark Expression Tree                                        │
│  ───────────────────────                                        │
│  .withColumn("motion_sustained",                                │
│      F.sum(F.when(                                              │
│          (F.col("signal") == "motion") & (F.col("value") > 0.5),│
│          F.col("duration_ms")                                   │
│      ).otherwise(0))                                            │
│      .over(Window.partitionBy("device_id")                      │
│                  .orderBy("timestamp")                          │
│                  .rangeBetween(-2000, 0))  # 2s lookback        │
│  )                                                              │
│  .withColumn("entry_triggered",                                 │
│      F.col("motion_sustained") >= 2000                          │
│  )                                                              │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.5 Session Detection via Window Functions
```python
# Pattern: Detect session boundaries without iteration
def detect_session_boundaries(df: DataFrame, config: StateConfig) -> DataFrame:
    window = Window.partitionBy("device_id").orderBy("timestamp")

    return (
        df
        # 1. Evaluate entry/exit conditions per row
        .withColumn("entry_condition_met", compile_condition(config.entry))
        .withColumn("exit_condition_met", compile_condition(config.exit))

        # 2. Mark state transitions
        .withColumn("prev_entry", F.lag("entry_condition_met").over(window))
        .withColumn("is_session_start",
            F.col("entry_condition_met") & ~F.coalesce(F.col("prev_entry"), F.lit(False)))

        # 3. Assign session IDs (cumsum trick)
        .withColumn("session_id",
            F.sum(F.when(F.col("is_session_start"), 1).otherwise(0)).over(window))

        # 4. Find session ends within each session
        .withColumn("session_end_detected", ...)

        # 5. Link child sessions to parents via temporal joins
    )
```

---

### Phase 3: Exception Handling

#### 3.1 Orphan Detection & Resolution
```
Incremental Processing Timeline:
─────────────────────────────────────────────────────────────→ time
     Batch 1          Batch 2          Batch 3
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ Parent   │     │ Child    │     │ Child    │
  │ Start    │     │ (orphan) │     │ End      │
  └──────────┘     └──────────┘     └──────────┘
                        │                │
                        └──── Resolution window ───┘
                              (look back for parent)
```

#### 3.2 Exception Actions
| Exception | Action | Description |
|-----------|--------|-------------|
| Orphan Child | `create_synthetic_parent` | Infer parent from child timing |
| Orphan Child | `buffer_for_resolution` | Wait for parent in next batch |
| Orphan Parent | `mark_incomplete` | Flag session as incomplete |
| Data Anomaly | `quarantine` | Move to anomaly table |
| Data Anomaly | `drop` | Discard silently |

---

### Phase 4: Benchmarking Suite

#### 4.1 Data Generator (`benchmarks/data_generator/`)
```python
# Generate realistic IoT robotics data
class IoTDataGenerator:
    def generate(
        self,
        num_devices: int,
        duration_hours: int,
        events_per_second: float,
        anomaly_rate: float = 0.01
    ) -> DataFrame:
        # Generates:
        # - Normal operational patterns
        # - Session start/end events
        # - Hierarchical nested events
        # - Intentional anomalies
        # - Data gaps (for orphan testing)
        pass
```

#### 4.2 Docker Setup (`benchmarks/docker/`)
```yaml
# docker-compose.yaml
services:
  spark-master:
    image: bitnami/spark:3.5
    environment:
      - SPARK_MODE=master
    ports:
      - "8080:8080"
      - "7077:7077"
    volumes:
      - ./data:/data
      - ./configs:/configs

  spark-worker:
    image: bitnami/spark:3.5
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
    deploy:
      replicas: 2
    depends_on:
      - spark-master

  hsm-runner:
    build: .
    depends_on:
      - spark-master
    volumes:
      - ./results:/results
```

#### 4.3 Benchmark Scenarios
| Scenario | Data Size | Devices | Duration | Purpose |
|----------|-----------|---------|----------|---------|
| Small | 100 MB | 10 | 1 day | Quick validation |
| Medium | 1 GB | 100 | 7 days | Local stress test |
| Large | 10 GB | 1000 | 30 days | Docker cluster test |
| Production | 100+ GB | 10000+ | 90 days | Cloud verification |

#### 4.4 Metrics Collection
```python
@dataclass
class BenchmarkMetrics:
    total_events: int
    processing_time_seconds: float
    events_per_second: float
    sessions_detected: int
    orphan_rate: float
    anomaly_rate: float
    memory_peak_mb: float
    cpu_utilization: float

    def extrapolate_to_production(
        self,
        target_events: int,
        cluster_nodes: int
    ) -> ProductionEstimate:
        # Linear/sub-linear extrapolation
        # Based on Spark scaling characteristics
        pass
```

#### 4.5 Cost Estimation
```python
# Cloud cost projection
class CostEstimator:
    PRICING = {
        "aws_emr": {"per_node_hour": 0.27},
        "databricks": {"dbu_rate": 0.15},
        "gcp_dataproc": {"per_node_hour": 0.24}
    }

    def estimate(
        self,
        benchmark_result: BenchmarkMetrics,
        target_scale: int,
        cloud_provider: str
    ) -> CostBreakdown:
        pass
```

---

### Phase 5: Demo Analysis

#### 5.1 Jupyter Notebook Demo
- Load sample IoT data
- Define session config via YAML
- Run HSM detection
- Visualize:
  - Session timeline
  - Hierarchy tree
  - Orphan/anomaly distribution
  - Performance metrics

#### 5.2 Sample Outputs
```
Session Detection Summary
═════════════════════════════════════════════════════════════
Total Events Processed:     1,234,567
Unique Devices:             100
─────────────────────────────────────────────────────────────
Level 0 Sessions (Auth):    342
Level 1 Sessions (Motion):  1,205
Level 2 Sessions (Micro):   4,892
─────────────────────────────────────────────────────────────
Orphan Events:              23 (0.002%)
Anomalies Quarantined:      7 (0.0006%)
Processing Time:            45.2s
Throughput:                 27,312 events/sec
═════════════════════════════════════════════════════════════
```

---

## Implementation Order

### Milestone 1: MVP Core
1. [ ] Project scaffolding (pyproject.toml, structure)
2. [ ] YAML config loader + Pydantic schema validation
3. [ ] Basic state machine (2-level, single pattern type)
4. [ ] DSL-to-PySpark compiler (signal_match pattern)
5. [ ] Pure PySpark executor with window functions
6. [ ] Unit tests for core components

### Milestone 2: Full DSL
6. [ ] Pattern DSL parser (all pattern types)
7. [ ] 3-level hierarchy support
8. [ ] Compound conditions (any_of, all_of)
9. [ ] Exception detection (orphans, anomalies)
10. [ ] Integration tests

### Milestone 3: Production Features
11. [ ] Incremental processing with checkpointing
12. [ ] Orphan resolution across batches
13. [ ] Performance optimization (broadcast, caching)
14. [ ] Comprehensive error handling

### Milestone 4: Benchmarking
15. [ ] IoT data generator
16. [ ] Docker compose setup
17. [ ] Benchmark runner scripts
18. [ ] Metrics collection + visualization
19. [ ] Cost estimation module

### Milestone 5: Documentation & Polish
20. [ ] README with quick start
21. [ ] DSL reference documentation
22. [ ] Jupyter demo notebook
23. [ ] GitHub Actions CI/CD
24. [ ] PyPI packaging

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Core Language | Python 3.10+ | Ecosystem compatibility |
| Big Data | PySpark 3.5+ | Target platform - **PURE PYSPARK ONLY** |
| Table Format | Apache Iceberg | Production catalog, daily batch ingestion |
| Config Parsing | PyYAML + Pydantic | Validation + type safety |
| Pattern Parsing | Custom recursive descent | Compile to PySpark expressions (no eval) |
| Testing | pytest + chispa | Spark DataFrame testing |
| Benchmarking | Docker + docker-compose | Reproducibility |
| Query Analysis | Spark UI + explain() | Built-in plan optimization |
| Visualization | matplotlib + seaborn | Lightweight |
| Docs | MkDocs | Simple, markdown-based |

## Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Daily Batch Processing                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Iceberg Catalog                HSM-PySpark                   Output        │
│   ──────────────                 ───────────                   ──────        │
│                                                                              │
│   ┌──────────────┐     ┌─────────────────────┐     ┌──────────────────┐     │
│   │ Raw Events   │     │  Session Detection  │     │ Detected         │     │
│   │ (Daily Part) │ ──→ │  Pure PySpark       │ ──→ │ Sessions         │     │
│   │              │     │  Window Functions   │     │ (Iceberg)        │     │
│   └──────────────┘     └─────────────────────┘     └──────────────────┘     │
│         │                        │                          │               │
│   UTC Schedule            Query Plan Analysis         Incremental           │
│   T+1 availability        for optimization            Merge/Upsert          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Daily Run Schedule:
  • Data arrives: UTC 00:00 (previous day complete)
  • Processing window: UTC 01:00-03:00 (configurable)
  • Orphan resolution: Look back N days for unresolved sessions
```

---

## Design Decisions (Locked In)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Streaming | **Batch-only v1** | Daily UTC schedule on Iceberg - streaming is v2 |
| Execution | **Pure PySpark** | Petabyte scale requires Catalyst optimization |
| UDFs | **None** | No pandas UDFs - compile DSL to native expressions |
| DSL | **Declarative only** | No Python eval - safe, optimizable |
| Table Format | **Iceberg** | Production-ready catalog integration |

## Remaining Open Questions

1. **Multi-tenancy**: Single config per run, or support multiple configs for different device types in same job?

2. **Query Plan Export**: Should we auto-export explain plans for all runs for debugging?

3. **Testing at Scale**: Mock Iceberg or real local catalog for integration tests?

---

## Success Criteria

- [ ] Process 1M events in <60s on local Docker (2 workers)
- [ ] Correctly detect 3-level hierarchical sessions
- [ ] Handle orphan resolution across batch boundaries
- [ ] <1% memory overhead vs baseline PySpark operations
- [ ] Clear documentation enabling self-service usage
- [ ] Reproducible benchmarks with scaling extrapolation
