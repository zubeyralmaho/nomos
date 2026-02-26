# Nomos Chaos & Stress Testing Suite

Validates the **Nomos Law** (~714ns latency) and **Haltless** (zero-downtime) architecture under extreme load.

## Test Coverage Summary

```
Unit Tests: 46/46 PASSED
├── Core Module Tests (22)
│   ├── control::tests::* (4 tests)
│   ├── ebpf::tests::* (2 tests)
│   ├── proxy::tests::* (3 tests)
│   ├── schema::tests::* (4 tests)
│   └── wasm_host::tests::* (3 tests)
│
├── NLP Module Tests (10)
│   ├── nlp::levenshtein::tests::* (2 tests)
│   ├── nlp::jaro::tests::* (2 tests)
│   ├── nlp::tfidf::tests::* (2 tests)
│   ├── nlp::synonym::tests::* (3 tests)
│   └── nlp::combined::tests::* (1 test)
│
└── Engine Module Tests (14)
    ├── engine::simd::tests::* (2 tests)
    ├── engine::embedding::tests::* (2 tests)
    ├── engine::confidence::tests::* (3 tests)
    ├── engine::lsh::tests::* (4 tests)
    ├── engine::healing::tests::* (2 tests)
    └── engine::healer::tests::* (3 tests)
```

Run all tests:
```bash
cargo test -p nomos-core
```

## Installation

```bash
cd tests
pip install -r requirements.txt
```

## Components

### 1. High-Velocity Load Generator (`stress_test.py`)

Floods the proxy with thousands of drifted JSON requests per second.

```bash
# Basic test at 10k RPS for 60 seconds
python stress_test.py --rps 10000 --duration 60

# High-intensity test
python stress_test.py --rps 50000 --workers 16 --duration 120

# Save results to JSON
python stress_test.py --rps 10000 --duration 30 --output results.json
```

**Output:**
- Real-time RPS, p50/p99/p99.9 latencies
- Nomos Law compliance indicator (p99 < 1ms)
- Healing rate percentage

### 2. Real-Time Observability TUI (`nomos_tui.py`)

Terminal User Interface for monitoring Nomos metrics in real-time.

```bash
# Start the TUI
python nomos_tui.py

# Simple text mode (no rich dependency)
python nomos_tui.py --simple

# Custom control URL
python nomos_tui.py --control-url http://localhost:8081
```

**Displays:**
- eBPF Fast/Slow path packets
- WASM healing events
- Per-core CPU utilization
- Memory (RSS) monitoring
- Active alerts

### 3. Haltless Hot-Swap Validation (`hot_swap_validator.py`)

Validates zero-downtime WASM module hot-swap under load.

```bash
# Test with 3 hot-swaps over 60 seconds
python hot_swap_validator.py --duration 60 --swap-interval 20

# Higher load
python hot_swap_validator.py --rps 10000 --duration 90 --swap-interval 30
```

**Validates:**
- Zero packets dropped during swap
- p99 latency remains < 1ms during swap
- Version increment after swap

### 4. Boundary Testing (`boundary_test.py`)

Tests system stability under extreme schema drift scenarios.

```bash
# Run all drift scenarios
python boundary_test.py

# Run specific scenario
python boundary_test.py --scenario total_rename

# More requests per scenario
python boundary_test.py --requests 200
```

**Scenarios:**
| Scenario | Description |
|----------|-------------|
| `total_rename` | Every field renamed to gibberish |
| `type_chaos` | All types changed (String→Int, Bool→Array) |
| `structure_explosion` | Flat → deeply nested structure |
| `field_avalanche` | 500+ random fields |
| `encoding_attack` | Unicode, Zalgo text, null bytes in field names |
| `null_storm` | All values are null |
| `empty_response` | Empty or minimal responses |
| `mixed_chaos` | Combination of all patterns |

### 6. NLP Algorithm Tests (Unit Tests)

Academic NLP algorithms with proper citations for course evaluation.

```bash
# Run all NLP tests
cargo test -p nomos-core nlp --release

# Run specific algorithm tests
cargo test -p nomos-core levenshtein --release
cargo test -p nomos-core jaro --release
cargo test -p nomos-core tfidf --release
cargo test -p nomos-core synonym --release
cargo test -p nomos-core combined --release
```

**Algorithms Tested:**
| Algorithm | Reference | Test Cases |
|-----------|-----------|------------|
| Levenshtein Distance | Levenshtein (1966) | kitten→sitting, edit ops |
| Jaro Similarity | Jaro (1989) | MARTHA→MARHTA, DWAYNE→DUANE |
| Jaro-Winkler | Winkler (1990) | Prefix bonus verification |
| N-gram TF-IDF | Salton & McGill (1983) | Trigram extraction, similarity |
| Synonym Matching | Custom | user↔person, email↔mail |
| Combined Ensemble | Custom | Weighted voting, threshold |

### 7. Memory Leak Monitor (`memory_monitor.py`)

Monitors RSS memory to detect leaks in WASM pool, eBPF maps, etc.

```bash
# Monitor for 5 minutes
python memory_monitor.py --duration 300

# Generate memory plot
python memory_monitor.py --duration 300 --plot memory.png

# Custom leak threshold
python memory_monitor.py --leak-threshold 5.0  # MB/hour
```

**Detects:**
- Memory growth rate (MB/hour)
- Leak confidence score
- Per-request memory consumption

### 8. Full Suite Runner (`run_chaos_suite.py`)

Orchestrates all tests in sequence.

```bash
# Quick validation (~2 minutes)
python run_chaos_suite.py --quick

# Full comprehensive test (~10 minutes)
python run_chaos_suite.py --full
```

## Expected Results

### Nomos Law Compliance

| Metric | Target | Acceptable |
|--------|--------|------------|
| p50 latency | < 100µs | < 500µs |
| p99 latency | < 714ns | < 1ms |
| p99.9 latency | < 5ms | < 10ms |

### Haltless Validation

| Criteria | Requirement |
|----------|-------------|
| Packets dropped during swap | 0 |
| p99 spike during swap | None (< 1ms) |
| Version increment | Pre → Post |

### Memory Requirements

| Metric | Threshold |
|--------|-----------|
| Growth rate | < 10 MB/hour |
| Absolute max | < 500 MB |
| Per-request overhead | < 1 KB |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CHAOS TEST ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   stress_test.py ──────────────────────► :8080 (Proxy)          │
│                                               │                 │
│   nomos_tui.py ◄────────────────────────┬───┤                   │
│                                         │   │                   │
│   hot_swap_validator.py ────────────────┤   │                   │
│                   │                     │   ▼                   │
│                   └────────────► :8081 (Control Plane)          │
│                                         │                       │
│   memory_monitor.py ────────────────────┴──► psutil             │
│                                                                 │
│   boundary_test.py ─────────────────────► :8080 (Proxy)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## CI/CD Integration

```bash
# Add to CI pipeline
python tests/run_chaos_suite.py --quick

# Exit code 0 = all tests passed, 1 = failures
```

## Troubleshooting

### "Nomos not responding"
```bash
# Ensure Nomos is running
cargo run --release --package nomos-core &
```

### "WASM healer not found" 
```bash
# Build the WASM healer
./build-wasm.sh
```

### "Connection refused on :8081"
```bash
# Check control plane is enabled
CONTROL_ADDR=0.0.0.0:8081 cargo run --release --package nomos-core
```

---

*Engineering Standard: Zero memory leaks. Zero downtime. It never stops.*
