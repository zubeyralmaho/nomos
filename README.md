# Nomos

> **It never stops.** Zero-latency, autonomous schema-healing proxy.

[![Tests](https://img.shields.io/badge/tests-46%20passed-brightgreen)]()
[![Rust](https://img.shields.io/badge/rust-1.82%2B-orange)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

---

## What is Nomos?

Nomos is a **transparent proxy** that sits between your client and external APIs. When APIs change their JSON response structure (schema drift), Nomos automatically **heals** the response back to the format your client expects — without code changes or redeployments.

```
┌──────────┐     ┌────────────────┐     ┌─────────────┐
│  Client  │────▶│  Nomos Proxy   │────▶│ External API│
│  (v1.0)  │◀────│  (heals drift) │◀────│   (v2.0)    │
└──────────┘     └────────────────┘     └─────────────┘
```

### The Problem

External APIs change without warning:
- `user_id` becomes `userId`
- `email` becomes `contact.email.primary`
- New fields appear, old fields disappear
- Types change (string → number)

Your client app crashes. Users complain. You scramble to deploy a fix.

### The Solution

Nomos **heals** these changes in real-time:
- Detects schema drift automatically
- Renames fields back to expected names
- Flattens nested structures
- Coerces types when possible
- **< 1ms latency** (invisible to users)

---

## Quick Start

### Prerequisites

- Rust 1.82+
- Linux (for eBPF features, optional)
- Python 3.8+ (for tests)

### Build & Run

```bash
# Clone and build
git clone https://github.com/zubeyralmaho/nomos.git
cd nomos
cargo build --release

# Start test upstream (simulates drifted API)
python3 tests/upstream_server.py --port 9090 --drift-mode v2 &

# Start Nomos proxy
RUST_LOG=info ./target/release/nomos-core
```

### Test Healing

```bash
# Direct API call (drifted)
curl http://localhost:9090/api/user
# Returns: {"userId": "123", "userName": "Alice", ...}

# Through Nomos (healed)
curl http://localhost:8080/api/user
# Returns: {"user_id": "123", "user_name": "Alice", ...}
```

---

## The Nomos Law

> **Proxy overhead must not exceed 1ms at p99 under sustained load.**

Every architectural decision serves this law. Current performance:

| Metric | Target | Achieved |
|--------|--------|----------|
| p99 Latency | < 1ms | **0.22ms** |
| Peak Throughput | > 1000 RPS | **5,146 RPS** |
| Success Rate | > 99% | **100%** |

---

## Architecture

### Core Components

```
nomos-core/src/
├── nlp/                     # NLP Algorithms
│   ├── levenshtein.rs       # Edit distance
│   ├── jaro.rs              # Jaro-Winkler similarity
│   ├── tfidf.rs             # N-gram TF-IDF
│   ├── synonym.rs           # Semantic synonym dictionary
│   └── combined.rs          # Weighted ensemble scoring
│
├── engine/                  # Semantic Healing Engine
│   ├── simd.rs              # AVX2/SSE2/NEON acceleration
│   ├── embedding.rs         # Trigram field embeddings
│   ├── lsh.rs               # O(1) candidate lookup
│   └── healer.rs            # Main healing logic
│
├── middleware.rs            # Response transformation
├── proxy.rs                 # HTTP proxy server
├── ebpf.rs                  # Kernel-level acceleration
└── schema.rs                # Schema store & fingerprinting
```

### NLP Ensemble

Nomos uses **5 NLP algorithms** in a weighted ensemble:

| Algorithm | Weight | Purpose |
|-----------|--------|---------|
| Synonym Dictionary | 35% | Semantic equivalence (user↔person) |
| Jaro-Winkler | 25% | Prefix-aware matching |
| Levenshtein | 20% | Typo detection |
| TF-IDF N-grams | 20% | Structural similarity |

### Zero-Copy Design

- No allocation in the hot path
- `bytes::Bytes` for reference-counted buffers
- `simd-json` for SIMD-accelerated parsing
- Arena allocation (`bumpalo`) for transformations

---

## Drift Modes Supported

| Mode | Description | Example |
|------|-------------|---------|
| `healthy` | No drift (baseline) | - |
| `v2` | API v2 style renames | `user_id` → `userId` |
| `camel` | CamelCase conversion | `user_name` → `userName` |
| `nested` | Nested structures | `name` → `user.profile.name` |
| `deep` | 6+ level nesting | `data.response.user.identity.name` |
| `typo` | Common typos | `emial`, `usesr` |
| `abbrev` | Abbreviations | `desc`, `addr`, `tel` |
| `legacy` | Legacy naming | `usr_nm`, `crt_dt` |
| `mixed` | All patterns combined | - |

---

## eBPF Acceleration (Optional)

Nomos includes an **XDP (eXpress Data Path)** eBPF program for kernel-level packet classification. This enables:

- **Fast-path bypass**: Healthy routes skip userspace entirely
- **Circuit breaker**: Drop blocked traffic at kernel level (XDP_DROP)
- **Zero-copy stats**: Per-CPU counters without atomic contention

### Build eBPF

```bash
# Prerequisites: nightly Rust + rust-src + bpf-linker
rustup install nightly
rustup component add rust-src --toolchain nightly
cargo install bpf-linker

# Build XDP program
./build-ebpf.sh

# Output: target/ebpf/nomos-xdp.o
```

### Load XDP Program

```bash
# Attach to interface (requires root)
sudo bpftool prog load target/ebpf/nomos-xdp.o /sys/fs/bpf/nomos_xdp
sudo bpftool net attach xdp pinned /sys/fs/bpf/nomos_xdp dev eth0

# Or let nomos-core auto-load
sudo ./target/release/nomos-core --ebpf
```

### Route Classification

| Class | XDP Action | Description |
|-------|------------|-------------|
| Healthy | XDP_PASS (fast) | Schema matches, minimal processing |
| NeedsHealing | XDP_PASS (slow) | Needs transformation in userspace |
| Blocked | XDP_DROP | Circuit breaker open |

---

## Configuration

```bash
# Environment variables
NOMOS_UPSTREAM_HOST=api.example.com
NOMOS_UPSTREAM_PORT=443
NOMOS_LISTEN_PORT=8080
NOMOS_CONFIDENCE_THRESHOLD=0.70
RUST_LOG=info
```

---

## Testing

```bash
# Run all unit tests (46 tests)
cargo test -p nomos-core

# Run chaos suite
python3 tests/chaos_suite.py

# Run stress test
python3 tests/stress_test.py --rps 5000 --duration 30

# Run specific drift mode
python3 tests/upstream_server.py --drift-mode deep &
curl http://localhost:8080/api/user
```

---

## Documentation

- [Architecture Specification](architecture.md) - Full engineering spec
- [Benchmark Report](BENCHMARK_REPORT.md) - Performance data & NLP analysis
- [Test Suite](tests/README.md) - Chaos & stress testing tools

---

## API Headers

Nomos adds diagnostic headers to responses:

| Header | Description |
|--------|-------------|
| `X-Nomos-Healed` | `true` if healing was applied |
| `X-Nomos-Latency-us` | Healing latency in microseconds |
| `X-Nomos-Healing-Ops` | Number of operations applied |
| `X-Nomos-Version` | Nomos version |

---

## Performance Tuning

### Enable SIMD (default)

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Enable eBPF (requires root)

```bash
sudo ./target/release/nomos-core --enable-ebpf
```

### Increase Worker Threads

```bash
TOKIO_WORKER_THREADS=8 ./target/release/nomos-core
```

---

## Roadmap

- [x] Core proxy with response healing
- [x] SIMD-accelerated JSON parsing
- [x] NLP ensemble (5 algorithms)
- [x] Modular architecture (13 files)
- [x] Chaos & stress testing suite
- [ ] WASM healer hot-swap
- [ ] GraphQL schema drift detection
- [ ] Kubernetes operator
- [ ] Web dashboard

---

## Contributing

Contributions welcome! Please read the architecture spec before submitting PRs.

```bash
# Run tests before committing
cargo test -p nomos-core
cargo clippy -p nomos-core
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

