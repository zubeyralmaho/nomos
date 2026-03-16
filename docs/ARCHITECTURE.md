# Nomos Architecture Guide

This document provides a comprehensive overview of Nomos's internal architecture, design principles, and implementation details.

## Table of Contents

- [Overview](#overview)
- [Design Principles](#design-principles)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [NLP Pipeline](#nlp-pipeline)
- [Healing Engine](#healing-engine)
- [eBPF Acceleration](#ebpf-acceleration)
- [WASM Healer Hot-Swap](#wasm-healer-hot-swap)
- [Memory Management](#memory-management)
- [Concurrency Model](#concurrency-model)

---

## Overview

Nomos is a **transparent schema-healing proxy** that sits between clients and external APIs. When APIs change their JSON response structure (schema drift), Nomos automatically transforms responses back to the format clients expect.

```
┌──────────┐     ┌────────────────────────────────────────┐     ┌─────────────┐
│          │     │              NOMOS PROXY               │     │             │
│  Client  │────▶│  ┌──────┐  ┌──────┐  ┌──────────────┐ │────▶│ External API│
│  (v1.0)  │◀────│  │Proxy │──│NLP   │──│Schema Healer │ │◀────│   (v2.0)    │
│          │     │  │Server│  │Engine│  │              │ │     │             │
└──────────┘     │  └──────┘  └──────┘  └──────────────┘ │     └─────────────┘
                 │                                        │
                 │  ┌──────────────────────────────────┐ │
                 │  │         Control Plane            │ │
                 │  │  Metrics │ Config │ Schema Store │ │
                 │  └──────────────────────────────────┘ │
                 └────────────────────────────────────────┘
```

---

## Design Principles

### The Nomos Law

> **Proxy overhead must not exceed 1ms at p99 under sustained load.**

Every architectural decision serves this law. The target latency budget:

| Component | Budget | Achieved |
|-----------|--------|----------|
| JSON Parsing | 200µs | ~100µs |
| Field Matching | 300µs | ~50µs |
| Transformation | 400µs | ~70µs |
| Serialization | 100µs | ~50µs |
| **Total** | **1000µs** | **~220µs** |

### Zero-Copy Philosophy

- No heap allocation in the hot path
- `bytes::Bytes` for reference-counted buffers
- `simd-json` for SIMD-accelerated parsing
- Arena allocation (`bumpalo`) for transformations

### Lock-Free Reads

- `ArcSwap` for schema store (readers never block)
- Per-CPU counters for metrics (no atomic contention)
- Sharded data structures where contention is unavoidable

---

## Core Components

### Project Structure

```
nomos/
├── nomos-core/                 # Main proxy library
│   └── src/
│       ├── lib.rs              # Library entry point
│       ├── main.rs             # Binary entry point
│       ├── proxy.rs            # HTTP proxy server
│       ├── middleware.rs       # Response transformation
│       ├── schema.rs           # Schema store & fingerprinting
│       ├── control.rs          # Control plane API
│       ├── ebpf.rs             # eBPF/XDP integration
│       ├── wasm_host.rs        # WASM healer host
│       ├── runtime.rs          # Tokio runtime builder
│       ├── error.rs            # Error types
│       │
│       ├── nlp/                # NLP Algorithms
│       │   ├── mod.rs          # Module exports
│       │   ├── levenshtein.rs  # Edit distance
│       │   ├── jaro.rs         # Jaro-Winkler similarity
│       │   ├── tfidf.rs        # N-gram TF-IDF
│       │   ├── synonym.rs      # Semantic synonyms
│       │   └── combined.rs     # Weighted ensemble
│       │
│       └── engine/             # Semantic Healing Engine
│           ├── mod.rs          # Module exports
│           ├── simd.rs         # SIMD acceleration
│           ├── embedding.rs    # Field embeddings
│           ├── lsh.rs          # Locality-Sensitive Hashing
│           ├── confidence.rs   # Match confidence scoring
│           ├── healing.rs      # Healing operations
│           └── healer.rs       # Main healer logic
│
├── nomos-ebpf/                 # eBPF/XDP kernel program
├── nomos-ebpf-common/          # Shared eBPF types
├── nomos-healer-guest/         # WASM healer guest module
├── dashboard/                  # Web monitoring UI
└── tests/                      # Chaos & stress tests
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `proxy.rs` | Accept HTTP requests, forward to upstream, receive responses |
| `middleware.rs` | Intercept responses, apply healing transformations |
| `schema.rs` | Store expected schema, detect drift |
| `nlp/` | String similarity algorithms for field name matching |
| `engine/` | High-performance matching and transformation |
| `control.rs` | Metrics, configuration, health checks |
| `ebpf.rs` | Kernel-level packet classification |
| `wasm_host.rs` | Execute WASM healers for custom logic |

---

## Data Flow

### Request/Response Flow

```
1. Client Request
      │
      ▼
2. ┌─────────────────────────────────────────────────┐
   │ Proxy Server (proxy.rs)                         │
   │  • Parse incoming request                       │
   │  • Add request tracing headers                  │
   └───────────────────────┬─────────────────────────┘
                           │
      ▼                    ▼
3. ┌─────────────────────────────────────────────────┐
   │ eBPF Fast Path (optional)                       │
   │  • Classify: Healthy / NeedsHealing / Blocked   │
   │  • Fast-path bypass for healthy routes          │
   └───────────────────────┬─────────────────────────┘
                           │
      ▼                    ▼
4. Forward to Upstream API
      │
      ▼
5. Receive Response
      │
      ▼
6. ┌─────────────────────────────────────────────────┐
   │ Response Middleware (middleware.rs)             │
   │  • Parse JSON response (simd-json)              │
   │  • Check schema fingerprint                     │
   │  • If drift detected: call Healer              │
   └───────────────────────┬─────────────────────────┘
                           │
      ▼                    ▼
7. ┌─────────────────────────────────────────────────┐
   │ Semantic Healer (engine/healer.rs)              │
   │  • Extract field names from response            │
   │  • Match against expected schema                │
   │  • Apply healing operations                     │
   └───────────────────────┬─────────────────────────┘
                           │
      ▼                    ▼
8. ┌─────────────────────────────────────────────────┐
   │ Serialization                                   │
   │  • Convert healed JSON back to bytes            │
   │  • Add diagnostic headers (X-Nomos-*)           │
   └───────────────────────┬─────────────────────────┘
                           │
      ▼                    ▼
9. Send Healed Response to Client
```

### Hot Path vs Cold Path

| Path | Description | Target Latency |
|------|-------------|----------------|
| Hot Path | No drift, pass-through | < 100µs |
| Warm Path | Known drift, cached healing | < 500µs |
| Cold Path | New drift, full NLP analysis | < 1ms |

---

## NLP Pipeline

Nomos uses a **weighted ensemble** of 5 NLP algorithms to match field names with high accuracy.

### Algorithm Overview

| Algorithm | Weight | Purpose | Time Complexity |
|-----------|--------|---------|-----------------|
| Synonym Dictionary | 35% | Semantic equivalence | O(1) lookup |
| Jaro-Winkler | 25% | Prefix-aware matching | O(m×n) |
| Levenshtein | 20% | Typo detection | O(m×n) |
| TF-IDF N-grams | 20% | Structural similarity | O(n) |

### Algorithm Details

#### 1. Synonym Dictionary (`nlp/synonym.rs`)

Pre-computed semantic mappings:

```rust
// Common synonyms
"user" ↔ "person", "account", "member"
"email" ↔ "mail", "address"
"id" ↔ "identifier", "key", "uid"
"created" ↔ "timestamp", "date", "time"
```

**Why 35% weight:** Synonyms catch semantic equivalence that string-based algorithms miss.

#### 2. Jaro-Winkler (`nlp/jaro.rs`)

Optimized for names with matching prefixes:

```
jaro_winkler("userName", "user_name") = 0.933
jaro_winkler("MARTHA", "MARHTA") = 0.944
```

**Why 25% weight:** Excellent for API versioning where prefixes stay consistent.

#### 3. Levenshtein Distance (`nlp/levenshtein.rs`)

Minimum edit operations (insert/delete/substitute):

```
levenshtein("kitten", "sitting") = 3
levenshtein("email", "emial") = 2  // typo detection
```

**Why 20% weight:** Essential for catching typos in field names.

#### 4. TF-IDF N-grams (`nlp/tfidf.rs`)

Character trigram comparison with TF-IDF weighting:

```
trigrams("user_name") = ["use", "ser", "er_", "r_n", "_na", "nam", "ame"]
trigrams("userName")  = ["use", "ser", "erN", "rNa", "Nam", "ame"]

similarity = cosine(tfidf(a), tfidf(b))
```

**Why 20% weight:** Handles underscore/camelCase conversions well.

### Ensemble Scoring

```rust
fn ensemble_score(source: &str, target: &str) -> f64 {
    let synonym = synonym_score(source, target) * 0.35;
    let jaro = jaro_winkler(source, target) * 0.25;
    let levenshtein = 1.0 - (edit_distance(source, target) as f64 
                            / max(source.len(), target.len()) as f64) * 0.20;
    let tfidf = ngram_similarity(source, target) * 0.20;
    
    synonym + jaro + levenshtein + tfidf
}
```

### Match Confidence Threshold

- **Default threshold:** 0.70
- **Conservative:** 0.85 (fewer false positives)
- **Aggressive:** 0.60 (catches more drift, higher risk)

---

## Healing Engine

The healing engine (`engine/`) provides high-performance field matching and transformation.

### Field Embedding (`engine/embedding.rs`)

Each field name is converted to a fixed-size embedding vector:

```rust
struct FieldEmbedding {
    trigrams: [i8; 64],  // Packed trigram frequencies
    length: u8,          // Original string length
    hash: u64,           // xxHash for fast comparison
}
```

### LSH Index (`engine/lsh.rs`)

Locality-Sensitive Hashing enables O(1) candidate lookup:

```rust
// Build index from expected schema
let index = LshIndex::new(&expected_fields);

// Find candidates for unknown field
let candidates = index.query(&unknown_field);  // O(1) average
```

### SIMD Acceleration (`engine/simd.rs`)

AVX2/SSE2/NEON intrinsics for vector operations:

```rust
#[cfg(target_arch = "x86_64")]
fn simd_dot_product_i8(a: &[i8; 64], b: &[i8; 64]) -> i32 {
    // AVX2 implementation
    let a_vec = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
    let b_vec = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
    // ... vectorized dot product
}
```

### Healing Operations (`engine/healing.rs`)

| Operation | Description | Example |
|-----------|-------------|---------|
| `Rename` | Change field name | `userId` → `user_id` |
| `Flatten` | Extract nested field | `user.profile.name` → `name` |
| `Nest` | Create nested structure | `name` → `user.name` |
| `Coerce` | Type conversion | `"123"` → `123` |
| `Default` | Add missing field | - → `"status": "unknown"` |

---

## eBPF Acceleration

### XDP Program (`nomos-ebpf/`)

The XDP (eXpress Data Path) program runs in the kernel for minimal latency:

```c
// Simplified XDP logic
SEC("xdp")
int nomos_xdp(struct xdp_md *ctx) {
    struct route_class class = classify_packet(ctx);
    
    switch (class) {
        case HEALTHY:
            return XDP_PASS;  // Fast path to userspace
        case NEEDS_HEALING:
            mark_for_healing(ctx);
            return XDP_PASS;
        case BLOCKED:
            return XDP_DROP;  // Circuit breaker
    }
}
```

### Classification Map

```rust
// Userspace updates classification map
ebpf_map.insert(route_hash, RouteClass::Healthy);

// Kernel reads classification
let class = ebpf_map.get(&route_hash);
```

### Benefits

| Metric | Without eBPF | With eBPF |
|--------|--------------|-----------|
| Healthy route latency | ~100µs | ~50µs |
| CPU usage | Higher | Lower |
| Circuit breaker | Userspace | Kernel-level |

---

## WASM Healer Hot-Swap

### Architecture

Custom healing logic runs in sandboxed WASM modules:

```
┌──────────────────────────────────────────────┐
│ Nomos Host Process                           │
│  ┌────────────────────────────────────────┐  │
│  │ WASM Pool (nomos-core/wasm_host.rs)    │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │  │
│  │  │Instance │ │Instance │ │Instance │   │  │
│  │  │   #1    │ │   #2    │ │   #3    │   │  │
│  │  └─────────┘ └─────────┘ └─────────┘   │  │
│  │                                        │  │
│  │  ┌─────────────────────────────────┐   │  │
│  │  │ Module Registry                 │   │  │
│  │  │ v1.0.0 (active) │ v1.1.0 (hot)  │   │  │
│  │  └─────────────────────────────────┘   │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

### Hot-Swap Process

1. **Compile new WASM module** (background)
2. **Pre-warm instances** (parallel compilation)
3. **Atomic swap** (ArcSwap pointer exchange)
4. **Drain old instances** (graceful cleanup)

```rust
// Hot-swap is atomic - no request drops
let new_registry = Arc::new(ModuleRegistry::from_bytes(&new_wasm)?);
registry.store(new_registry);  // Atomic pointer swap
```

### WASM Interface (WIT)

```wit
// nomos-core/wit/healer.wit
interface healer {
    record heal-request {
        original-json: string,
        expected-schema: list<string>,
    }
    
    record heal-response {
        healed-json: string,
        operations: list<heal-op>,
    }
    
    heal: func(request: heal-request) -> result<heal-response, string>;
}
```

---

## Memory Management

### Arena Allocation

Per-request arena allocators eliminate heap fragmentation:

```rust
fn process_request(req: Request) -> Response {
    // Create arena for this request
    let arena = Bump::new();
    
    // All allocations use the arena
    let parsed: JsonValue = parse_json(&arena, req.body());
    let healed = heal_json(&arena, parsed);
    
    // Arena is dropped at end of request (single deallocation)
}
```

### Buffer Pool

Pre-allocated buffers for common sizes:

```rust
struct BufferPool {
    small: Vec<Buffer<1024>>,   // 1KB buffers
    medium: Vec<Buffer<16384>>, // 16KB buffers  
    large: Vec<Buffer<65536>>,  // 64KB buffers
}
```

### Memory Limits

| Component | Limit | Purpose |
|-----------|-------|---------|
| Arena per request | 1 MB | Prevent DoS |
| WASM instance | 64 MB | Sandbox limit |
| Total heap | 512 MB | Process limit |

---

## Concurrency Model

### Tokio Runtime

Multi-threaded async runtime with work-stealing:

```rust
fn build_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .build()
        .unwrap()
}
```

### Sharded Metrics

Per-CPU counters avoid atomic contention:

```rust
struct ShardedMetrics {
    shards: Vec<Shard>,  // One per CPU core
}

impl ShardedMetrics {
    fn increment(&self) {
        let shard = &self.shards[current_cpu()];
        shard.count.fetch_add(1, Relaxed);  // No cross-core contention
    }
    
    fn aggregate(&self) -> u64 {
        self.shards.iter().map(|s| s.count.load(Relaxed)).sum()
    }
}
```

### Lock-Free Schema Store

Readers never block; writers are rare:

```rust
struct SchemaStore {
    inner: ArcSwap<SchemaInner>,
}

impl SchemaStore {
    fn read(&self) -> Guard<Arc<SchemaInner>> {
        self.inner.load()  // Lock-free
    }
    
    fn update(&self, new: SchemaInner) {
        self.inner.store(Arc::new(new));  // Atomic swap
    }
}
```

---

## Performance Characteristics

### Latency Breakdown (p99)

```
Total: 220µs
├── Network I/O: 50µs (22%)
├── JSON Parse: 60µs (27%)
├── Field Match: 40µs (18%)
├── Transform: 50µs (23%)
└── Serialize: 20µs (9%)
```

### Throughput Scaling

| Workers | RPS | Notes |
|---------|-----|-------|
| 1 | 2,000 | Single-threaded |
| 2 | 3,800 | Linear scaling |
| 4 | 5,100 | Peak efficiency |
| 8 | 5,200 | Diminishing returns |

### Memory Profile

| State | RSS |
|-------|-----|
| Idle | 15 MB |
| Under load (1k RPS) | 45 MB |
| Peak (5k RPS) | 80 MB |
| Post-peak | 25 MB (GC) |

---

## Further Reading

- **[Getting Started](GETTING_STARTED.md)** - Quick start guide
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment
- **[API Reference](API.md)** - Control plane API
- **[Benchmark Report](../BENCHMARK_REPORT.md)** - Performance data

---

*For implementation details, see the source code with inline documentation.*
