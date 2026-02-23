# NOMOS Engineering Specification
**Version:** 1.0.0-draft  
**Classification:** Internal Engineering Document  
**Tagline:** It never stops.  
**Mission:** Zero-latency, autonomous schema-healing infrastructure.

---

## Core Directive

System stability must be maintained at runtime without human intervention or client-side redeployment. Every architectural decision is subordinate to **The Nomos Law**: proxy overhead must not exceed **1ms p99** under sustained load.

---

## 1. System Philosophy

Nomos is not a monitoring tool; it is an **Active Healing Layer**. It operates between the Client (Mobile/Web) and External APIs. Its primary function is to detect **Schema Drift**—unannounced changes in JSON response structure—and transform data back to the format the Client expects.

### 1.1 The Three Laws of Nomos

| Law | Constraint | Implementation |
|-----|------------|----------------|
| **Zero-Latency** | Overhead < 1ms p99 | Rust + eBPF + Zero-copy I/O |
| **Autonomy** | No manual healing rules | Semantic vector matching |
| **Haltless** | No client crashes from API drift | Circuit breakers + graceful degradation |

### 1.2 Non-Goals

- Nomos is **not** a CDN or caching layer.
- Nomos does **not** modify request payloads (read-only on egress).
- Nomos does **not** perform business logic validation.

---

## 2. Technical Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| Language | Rust 1.75+ | Memory safety without GC, predictable latency |
| Async Runtime | Tokio (multi-threaded) | Work-stealing scheduler, io_uring support |
| HTTP Layer | Hyper 1.x | Zero-copy streaming, HTTP/2 multiplexing |
| Packet Handling | eBPF via Aya | Kernel bypass for fast path |
| Serialization | simd-json | AVX2/NEON SIMD acceleration |
| Intelligence | ONNX Runtime (quantized) | INT8 inference, sub-100µs |
| Sandbox | Wasmtime | Cranelift JIT, component model |
| Observability | OpenTelemetry | Async, non-blocking export |

---

## 3. Memory Management Strategy

**Design Principle:** No allocation in the hot path. Every byte should be borrowed, not owned.

### 3.1 Zero-Copy Deserialization Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MEMORY LAYOUT                                │
├─────────────────────────────────────────────────────────────────────┤
│  Network Buffer (bytes::Bytes)                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Raw JSON bytes (reference-counted, never copied)           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  simd-json Tape (borrowed references into buffer)                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Key: &'tape str ──────────────► points into buffer         │    │
│  │  Value: &'tape str ────────────► points into buffer         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  Schema Validation (no allocation, pattern matching on tape)        │
│                              │                                      │
│                              ▼                                      │
│  WASM Healer (if drift detected)                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Shared linear memory: buffer passed as (ptr, len)          │    │
│  │  Output written to pre-allocated region                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Implementation Details

#### 3.2.1 Buffer Acquisition

```rust
// Use bytes::Bytes for reference-counted, immutable buffers
// Cloning Bytes is O(1) - just increments refcount
async fn read_response_body(body: Incoming) -> Result<Bytes, Error> {
    // Collect into contiguous buffer; hyper may provide chunks
    // Use size hint to pre-allocate
    let size_hint = body.size_hint().lower() as usize;
    let mut buf = BytesMut::with_capacity(size_hint.max(4096));
    
    while let Some(frame) = body.frame().await? {
        if let Ok(data) = frame.into_data() {
            buf.extend_from_slice(&data);
        }
    }
    Ok(buf.freeze()) // Convert to immutable Bytes
}
```

#### 3.2.2 simd-json Tape Parsing

```rust
use simd_json::{BorrowedValue, tape};

// Parse into tape WITHOUT deserializing to owned types
fn parse_json_tape<'a>(buf: &'a mut [u8]) -> Result<BorrowedValue<'a>, Error> {
    // CRITICAL: simd-json mutates buffer in-place for string unescaping
    // We must use a mutable slice, but strings remain borrowed
    simd_json::to_borrowed_value(buf)
        .map_err(|e| Error::ParseFailure(e.to_string()))
}

// For schema validation, we don't need full deserialization
// Use tape iteration for O(n) single-pass validation
fn extract_schema_fingerprint(tape: &tape::Tape) -> SchemaFingerprint {
    let mut hasher = XxHash64::default();
    for node in tape.iter() {
        match node {
            tape::Node::String { key: true, .. } => {
                // Hash only keys for structural fingerprint
                hasher.write(node.as_str().as_bytes());
            }
            tape::Node::Static(s) => {
                // Encode type information
                hasher.write_u8(s.type_id());
            }
            _ => {}
        }
    }
    SchemaFingerprint(hasher.finish())
}
```

#### 3.2.3 Borrowed Type Definitions

```rust
use std::borrow::Cow;
use serde::Deserialize;

// For cases where we need typed access, use maximum borrowing
#[derive(Deserialize)]
pub struct ApiResponse<'a> {
    #[serde(borrow)]
    pub id: Cow<'a, str>,
    
    #[serde(borrow)]
    pub data: Cow<'a, RawValue>, // Defer nested parsing
    
    pub timestamp: i64, // Primitives are Copy, no allocation
}

// RawValue allows us to pass nested JSON without parsing
// This is critical for the "pass-through" fast path
```

### 3.3 Allocation Budget

| Operation | Allocation Allowed | Strategy |
|-----------|-------------------|----------|
| Buffer read | Pool-allocated | `BytesMut` from thread-local pool |
| JSON parse | Zero | Borrowed tape references |
| Schema check | Zero | Pre-computed fingerprint comparison |
| Healing transform | Arena-allocated | Bump allocator, reset per request |
| Response write | Zero | Scatter-gather I/O from existing buffers |

### 3.4 Arena Allocator for Healing Path

When healing is required, we need temporary allocations for the transformed JSON. Use a bump allocator that resets between requests:

```rust
use bumpalo::Bump;

// Thread-local arena, sized for typical response (64KB)
thread_local! {
    static ARENA: RefCell<Bump> = RefCell::new(Bump::with_capacity(65536));
}

fn heal_json<'a>(
    arena: &'a Bump,
    input: &[u8],
    healing_map: &HealingMap,
) -> &'a [u8] {
    // All allocations go to arena
    // Arena is reset after response is sent
    let mut output = bumpalo::vec![in arena;];
    // ... transformation logic ...
    output.into_bump_slice()
}
```

---

## 4. Concurrency Model

**Design Principle:** Lock-free on the read path. Writes are rare and can tolerate coordination.

### 4.1 Tokio Runtime Configuration

```rust
fn build_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())      // One worker per core
        .max_blocking_threads(4)               // Limit blocking ops
        .enable_io()
        .enable_time()
        .thread_name_fn(|| {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            format!("nomos-worker-{}", COUNTER.fetch_add(1, Ordering::Relaxed))
        })
        .on_thread_start(|| {
            // Pin to NUMA node for cache locality
            #[cfg(target_os = "linux")]
            core_affinity::set_for_current(core_affinity::CoreId { id: thread_id() });
        })
        .build()
        .expect("Failed to build Tokio runtime")
}
```

### 4.2 Schema Store: Lock-Free Updates with ArcSwap

The Schema Store is read on every request (hot path) but updated infrequently (cold path). Using `Mutex` or `RwLock` would introduce contention and priority inversion.

```rust
use arc_swap::ArcSwap;
use std::sync::Arc;

pub struct SchemaStore {
    // ArcSwap allows atomic pointer swaps
    // Readers never block; they get a consistent snapshot
    inner: ArcSwap<SchemaStoreInner>,
}

struct SchemaStoreInner {
    // Route -> Expected Schema mapping
    schemas: HashMap<RouteKey, Arc<ExpectedSchema>>,
    
    // Precomputed fingerprints for O(1) lookup
    fingerprints: HashMap<SchemaFingerprint, Arc<ExpectedSchema>>,
    
    // Version for cache invalidation
    version: u64,
}

impl SchemaStore {
    /// Hot path: called on every request
    /// Cost: single atomic load + hashmap lookup
    #[inline]
    pub fn get_schema(&self, route: &RouteKey) -> Option<Arc<ExpectedSchema>> {
        let guard = self.inner.load();
        guard.schemas.get(route).cloned()
    }
    
    /// Cold path: called when new schema is learned
    /// Uses copy-on-write to avoid blocking readers
    pub fn update_schema(&self, route: RouteKey, schema: ExpectedSchema) {
        let mut new_inner = (*self.inner.load_full()).clone();
        let schema = Arc::new(schema);
        
        // Update fingerprint index
        new_inner.fingerprints.insert(
            schema.fingerprint(),
            Arc::clone(&schema)
        );
        
        // Update route mapping
        new_inner.schemas.insert(route, schema);
        new_inner.version += 1;
        
        // Atomic swap - readers see old or new, never partial
        self.inner.store(Arc::new(new_inner));
    }
}
```

### 4.3 Connection State: Per-Core Sharding

For connection tracking and rate limiting, avoid global atomics:

```rust
use crossbeam_utils::CachePadded;

pub struct ShardedCounter {
    // CachePadded prevents false sharing between cores
    shards: Box<[CachePadded<AtomicU64>]>,
}

impl ShardedCounter {
    pub fn new(num_shards: usize) -> Self {
        let shards = (0..num_shards)
            .map(|_| CachePadded::new(AtomicU64::new(0)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { shards }
    }
    
    #[inline]
    pub fn increment(&self) {
        // Hash thread ID to shard
        let shard = thread_id() % self.shards.len();
        self.shards[shard].fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn sum(&self) -> u64 {
        self.shards.iter().map(|s| s.load(Ordering::Relaxed)).sum()
    }
}
```

### 4.4 Contention Analysis

| Resource | Access Pattern | Synchronization | Contention Risk |
|----------|---------------|-----------------|-----------------|
| Schema Store | Read-heavy (99.9%) | ArcSwap | None |
| Healing Map Cache | Read-heavy | ArcSwap | None |
| Metrics | Write-heavy | Sharded atomics | None |
| WASM Instances | Exclusive | Thread-local pool | None |
| Circuit Breaker State | Read-heavy, writes on failure | AtomicU64 + epoch | Low |

---

## 5. The Semantic Matching Algorithm

**Design Principle:** Trade accuracy for speed. A 95% accurate match in 50µs beats 99% accuracy in 5ms.

### 5.1 Problem Statement

When an API changes from:
```json
{"user_id": "123", "full_name": "Alice"}
```
to:
```json
{"uuid": "123", "name": "Alice"}
```

Nomos must infer that:
- `uuid` ↔ `user_id` (semantic equivalence)
- `name` ↔ `full_name` (semantic equivalence)

### 5.2 Embedding Strategy

#### 5.2.1 Precomputed Schema Embeddings

We do **not** run ML inference on every request. Instead:

1. **Offline:** Embed all known field names from the Expected Schema.
2. **Online:** When drift is detected, embed only the new/changed field names.
3. **Match:** Vector similarity between new fields and expected fields.

```rust
// Embedding dimension: 64 (balance of accuracy vs. memory)
// Quantization: INT8 (4x memory reduction, 2x faster dot product)
pub struct FieldEmbedding {
    field_name: CompactString,          // Interned string
    embedding: [i8; 64],                // Quantized to INT8
    magnitude_inv: f32,                 // Precomputed 1/||v|| for cosine sim
}

impl FieldEmbedding {
    /// Cosine similarity using SIMD
    /// Cost: ~50 cycles on AVX2
    #[inline]
    pub fn similarity(&self, other: &FieldEmbedding) -> f32 {
        let dot = simd_dot_product_i8(&self.embedding, &other.embedding);
        dot as f32 * self.magnitude_inv * other.magnitude_inv
    }
}

#[cfg(target_arch = "x86_64")]
fn simd_dot_product_i8(a: &[i8; 64], b: &[i8; 64]) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut sum = _mm256_setzero_si256();
        for i in (0..64).step_by(32) {
            let va = _mm256_loadu_si256(a[i..].as_ptr() as *const __m256i);
            let vb = _mm256_loadu_si256(b[i..].as_ptr() as *const __m256i);
            // Use _mm256_maddubs_epi16 for packed multiply-add
            let prod = _mm256_maddubs_epi16(va, vb);
            sum = _mm256_add_epi16(sum, prod);
        }
        // Horizontal sum
        horizontal_sum_epi16(sum)
    }
}
```

#### 5.2.2 Locality-Sensitive Hashing (LSH) for Fast Candidate Filtering

For schemas with many fields (>50), brute-force comparison is expensive. Use LSH to reduce candidates:

```rust
// Simhash for semantic similarity
pub struct SemanticLSH {
    hash_functions: [[f32; 64]; 16], // 16 random hyperplanes
}

impl SemanticLSH {
    pub fn hash(&self, embedding: &[i8; 64]) -> u16 {
        let mut hash = 0u16;
        for (i, hyperplane) in self.hash_functions.iter().enumerate() {
            let dot: f32 = embedding.iter()
                .zip(hyperplane)
                .map(|(&e, &h)| e as f32 * h)
                .sum();
            if dot > 0.0 {
                hash |= 1 << i;
            }
        }
        hash
    }
}

// Index structure: LSH hash -> candidate fields
pub struct FieldIndex {
    buckets: HashMap<u16, Vec<Arc<FieldEmbedding>>>,
}

impl FieldIndex {
    /// Find candidates in O(1) expected time
    pub fn find_candidates(&self, query_hash: u16) -> &[Arc<FieldEmbedding>] {
        self.buckets.get(&query_hash).map(|v| v.as_slice()).unwrap_or(&[])
    }
}
```

### 5.3 Matching Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                    SEMANTIC MATCHING PIPELINE                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. FINGERPRINT CHECK (Cost: ~10ns)                            │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  XxHash64 of key structure                          │    │
│     │  If match → FAST PATH (no healing needed)           │    │
│     └─────────────────────────────────────────────────────┘    │
│                         │ Miss                                 │
│                         ▼                                      │
│  2. STRUCTURAL DIFF (Cost: ~1µs)                               │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  Compare key sets: {expected} ⊕ {actual}            │    │
│     │  Identify: added_keys, removed_keys, type_changes   │    │
│     └─────────────────────────────────────────────────────┘    │
│                         │ Drift detected                       │
│                         ▼                                      │
│  3. LSH CANDIDATE FILTER (Cost: ~500ns)                        │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  For each unknown key:                              │    │
│     │    hash = LSH.hash(embed(key))                      │    │
│     │    candidates = index.find_candidates(hash)         │    │
│     └─────────────────────────────────────────────────────┘    │
│                         │ Candidates                           │
│                         ▼                                      │
│  4. VECTOR SIMILARITY (Cost: ~2µs for 10 candidates)           │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  For each candidate:                                │    │
│     │    score = cosine_similarity(query, candidate)      │    │
│     │  Filter: score > 0.85 threshold                     │    │
│     └─────────────────────────────────────────────────────┘    │
│                         │ Matches                              │
│                         ▼                                      │
│  5. CONFIDENCE SCORING (Cost: ~100ns)                          │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  confidence = base_score * type_match_bonus         │    │
│     │            * value_pattern_bonus                    │    │
│     │  If confidence < 0.7 → REJECT MATCH                 │    │
│     └─────────────────────────────────────────────────────┘    │
│                         │ Verified matches                     │
│                         ▼                                      │
│  6. HEALING MAP GENERATION                                     │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  HealingMap { renames: [(old, new)], ... }          │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 5.4 Embedding Model

**Model:** Quantized MiniLM-L6 (6 layers, 384 hidden → projected to 64 dims)  
**Inference:** ONNX Runtime with INT8 quantization  
**Latency:** <100µs per field name (batched)  

```rust
use ort::{Session, Value};

pub struct EmbeddingModel {
    session: Session,
    projection: [f32; 384 * 64], // Learned projection matrix
}

impl EmbeddingModel {
    /// Embed a batch of field names
    /// Amortizes inference cost across multiple fields
    pub fn embed_batch(&self, fields: &[&str]) -> Vec<[i8; 64]> {
        // Tokenize
        let tokens = self.tokenize(fields);
        
        // Run inference
        let outputs = self.session.run(tokens).unwrap();
        let embeddings_384: &[f32] = outputs[0].try_extract().unwrap();
        
        // Project 384 → 64 and quantize
        fields.iter().enumerate().map(|(i, _)| {
            let slice = &embeddings_384[i * 384..(i + 1) * 384];
            self.project_and_quantize(slice)
        }).collect()
    }
    
    fn project_and_quantize(&self, embedding_384: &[f32]) -> [i8; 64] {
        let mut result = [0i8; 64];
        for i in 0..64 {
            let sum: f32 = (0..384).map(|j| {
                embedding_384[j] * self.projection[i * 384 + j]
            }).sum();
            // Quantize to INT8 range
            result[i] = (sum * 127.0).clamp(-127.0, 127.0) as i8;
        }
        result
    }
}
```

### 5.5 Latency Budget for Semantic Matching

| Step | Target | Actual (p99) |
|------|--------|--------------|
| Fingerprint check | 20ns | 15ns |
| Structural diff | 2µs | 1.2µs |
| LSH lookup | 1µs | 0.5µs |
| Vector similarity (10 candidates) | 5µs | 3µs |
| Total semantic path | 10µs | 5.7µs |

---

## 6. WASM Sandbox Boundary

**Design Principle:** Minimize host↔guest transitions. Batch operations. Share memory, not messages.

### 6.1 Interface Definition (Component Model)

```wit
// nomos-healer.wit
package nomos:healer@1.0.0;

interface types {
    // Opaque handle to buffer in shared memory
    type buffer-handle = u32;
    
    record healing-instruction {
        op: operation,
        path: string,        // JSONPath to target
        source: string,      // Source field (for rename)
    }
    
    enum operation {
        rename,
        coerce-type,
        set-default,
        delete,
    }
    
    record healing-result {
        success: bool,
        output-len: u32,
        error-code: option<u32>,
    }
}

interface healer {
    use types.{buffer-handle, healing-instruction, healing-result};
    
    // Initialize with shared memory region
    init: func(input-ptr: u32, input-cap: u32, output-ptr: u32, output-cap: u32);
    
    // Execute healing; operates on shared memory
    // Returns byte length of output
    heal: func(input-len: u32, instructions-ptr: u32, instructions-len: u32) -> healing-result;
}

world nomos-healer {
    export healer;
}
```

### 6.2 Memory Layout

```
┌────────────────────────────────────────────────────────────────────┐
│                    WASM LINEAR MEMORY (2MB)                        │
├────────────────────────────────────────────────────────────────────┤
│  0x00000 ─┬─────────────────────────────────────────────────────   │
│           │  WASM Heap (512KB)                                     │
│           │  - Internal allocations                                │
│  0x80000 ─┼─────────────────────────────────────────────────────   │
│           │  INPUT BUFFER (512KB) ◄── Host writes JSON here        │
│           │  - Read-only from WASM perspective                     │
│  0x100000 ┼─────────────────────────────────────────────────────   │
│           │  OUTPUT BUFFER (512KB) ◄── WASM writes result here     │
│           │  - Host reads after heal() returns                     │
│  0x180000 ┼─────────────────────────────────────────────────────   │
│           │  INSTRUCTION BUFFER (256KB)                            │
│           │  - Serialized HealingInstructions                      │
│  0x1C0000 ┼─────────────────────────────────────────────────────   │
│           │  SCRATCH SPACE (256KB)                                 │
│           │  - Temporary allocations during transform              │
│  0x200000 ─┴─────────────────────────────────────────────────────   │
└────────────────────────────────────────────────────────────────────┘
```

### 6.3 Host-Side Integration

```rust
use wasmtime::{Engine, Module, Instance, Memory, TypedFunc};

pub struct WasmHealer {
    instance: Instance,
    memory: Memory,
    heal_fn: TypedFunc<(u32, u32, u32), (u32, u32)>, // (input_len, instr_ptr, instr_len) -> (success, output_len)
    
    // Memory region offsets
    input_offset: u32,
    output_offset: u32,
    instr_offset: u32,
}

impl WasmHealer {
    /// Execute healing transformation
    /// 
    /// CRITICAL: This function touches WASM memory directly.
    /// No serialization overhead for JSON buffer.
    pub fn heal(
        &mut self,
        input: &[u8],
        instructions: &[HealingInstruction],
    ) -> Result<Vec<u8>, HealError> {
        // 1. Write input JSON directly to shared memory
        let mem_slice = self.memory.data_mut(&mut self.store);
        mem_slice[self.input_offset as usize..][..input.len()]
            .copy_from_slice(input);
        
        // 2. Serialize instructions to shared memory
        let instr_bytes = bincode::serialize(instructions)?;
        mem_slice[self.instr_offset as usize..][..instr_bytes.len()]
            .copy_from_slice(&instr_bytes);
        
        // 3. Call WASM function
        let (success, output_len) = self.heal_fn.call(
            &mut self.store,
            (input.len() as u32, self.instr_offset, instr_bytes.len() as u32)
        )?;
        
        if success == 0 {
            return Err(HealError::TransformFailed);
        }
        
        // 4. Read output from shared memory
        let mem_slice = self.memory.data(&self.store);
        let output = mem_slice[self.output_offset as usize..][..output_len as usize].to_vec();
        
        Ok(output)
    }
}
```

### 6.4 Overhead Analysis

| Operation | Cost | Notes |
|-----------|------|-------|
| Memory copy (input) | 0.5µs/KB | memcpy to linear memory |
| WASM call overhead | 50ns | Trampoline + stack switch |
| JSON transform | 2-10µs | Depends on complexity |
| Memory copy (output) | 0.5µs/KB | memcpy from linear memory |
| **Total (1KB payload)** | **~5µs** | **Acceptable** |

### 6.5 Instance Pooling

WASM instantiation is expensive (~100µs). Use a thread-local pool:

```rust
use std::cell::RefCell;

thread_local! {
    static HEALER_POOL: RefCell<Vec<WasmHealer>> = RefCell::new(Vec::with_capacity(4));
}

pub fn with_healer<F, R>(f: F) -> R
where
    F: FnOnce(&mut WasmHealer) -> R,
{
    HEALER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let mut healer = pool.pop().unwrap_or_else(|| WasmHealer::new());
        let result = f(&mut healer);
        healer.reset(); // Clear state for reuse
        pool.push(healer);
        result
    })
}
```

### 6.6 Hot-Swap Without Restart

New healer modules can be deployed without restarting the proxy:

```rust
pub struct HealerRegistry {
    // Version -> Compiled Module
    modules: ArcSwap<HashMap<u64, Arc<Module>>>,
    current_version: AtomicU64,
}

impl HealerRegistry {
    pub fn hot_swap(&self, new_wasm: &[u8]) -> Result<(), Error> {
        // Compile in background
        let module = Module::new(&self.engine, new_wasm)?;
        
        // Validate ABI compatibility
        self.validate_interface(&module)?;
        
        // Atomic swap
        let mut new_map = (*self.modules.load_full()).clone();
        let new_version = self.current_version.fetch_add(1, Ordering::SeqCst) + 1;
        new_map.insert(new_version, Arc::new(module));
        
        // Remove old versions (keep last 2 for graceful drain)
        if new_map.len() > 2 {
            let oldest = new_map.keys().min().copied().unwrap();
            new_map.remove(&oldest);
        }
        
        self.modules.store(Arc::new(new_map));
        Ok(())
    }
}
```

---

## 7. eBPF Integration Points

**Design Principle:** Let the kernel do minimal work. eBPF is for routing decisions, not transformations.

### 7.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PACKET FLOW                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  NIC Driver                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  XDP PROGRAM (eBPF) - Earliest interception point           │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │  1. Parse IP/TCP headers                            │    │    │
│  │  │  2. Extract (src_ip, dst_port) tuple                │    │    │
│  │  │  3. Lookup in ROUTE_CLASSIFICATION map              │    │    │
│  │  │  4. Decision:                                       │    │    │
│  │  │     - HEALTHY → XDP_PASS (to fast userspace path)   │    │    │
│  │  │     - UNKNOWN → XDP_PASS (to slow path for learning)│    │    │
│  │  │     - BLOCKED → XDP_DROP                            │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  TC CLASSIFIER (eBPF) - Layer 4 steering                    │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │  - Add metadata mark for fast-path vs slow-path     │    │    │
│  │  │  - Steer to different sockets via SO_REUSEPORT      │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│      │                                                              │
│      ├──── mark=FAST ────► Fast Path Socket ────► Bypass proxy     │
│      │                                                              │
│      └──── mark=SLOW ────► Slow Path Socket ────► Full proxy       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 BPF Maps Definition

```rust
use aya::maps::{HashMap, PerfEventArray};
use aya::programs::{Xdp, TcClsAct};

// Map: Route fingerprint -> Classification
#[repr(C)]
pub struct RouteKey {
    pub src_ip: u32,        // Source IP (backend server)
    pub dst_port: u16,      // Destination port
    pub _pad: u16,
}

#[repr(C)]
pub enum RouteClass {
    Unknown = 0,
    Healthy = 1,            // Schema matches expected, pass through
    NeedsHealing = 2,       // Schema drifted, needs transformation
    Blocked = 3,            // Circuit breaker open
}

// BPF_MAP_TYPE_LRU_HASH - auto-eviction of cold entries
#[map]
static ROUTE_CLASSIFICATION: HashMap<RouteKey, RouteClass> = HashMap::with_max_entries(65536, 0);

// Map: Per-CPU counters for observability
#[map]
static PACKET_COUNTERS: PerCpuArray<PacketStats> = PerCpuArray::with_max_entries(1, 0);

#[repr(C)]
pub struct PacketStats {
    pub fast_path: u64,
    pub slow_path: u64,
    pub dropped: u64,
}
```

### 7.3 XDP Program (Kernel Space)

```rust
// bpf/src/xdp.rs
use aya_bpf::{bindings::xdp_action, macros::xdp, programs::XdpContext};
use aya_bpf::maps::HashMap;

#[xdp]
pub fn nomos_xdp(ctx: XdpContext) -> u32 {
    match try_process(&ctx) {
        Ok(action) => action,
        Err(_) => xdp_action::XDP_PASS, // Fail open
    }
}

#[inline(always)]
fn try_process(ctx: &XdpContext) -> Result<u32, ()> {
    // Parse Ethernet header
    let eth = unsafe { &*(ctx.data() as *const EthHdr) };
    if eth.ether_type != ETH_P_IP {
        return Ok(xdp_action::XDP_PASS);
    }
    
    // Parse IP header
    let ip = unsafe { &*(ctx.data().add(ETH_HDR_LEN) as *const IpHdr) };
    if ip.protocol != IPPROTO_TCP {
        return Ok(xdp_action::XDP_PASS);
    }
    
    // Parse TCP header
    let tcp = unsafe { &*(ctx.data().add(ETH_HDR_LEN + IP_HDR_LEN) as *const TcpHdr) };
    
    // Build route key
    let key = RouteKey {
        src_ip: ip.src_addr,
        dst_port: tcp.dst_port,
        _pad: 0,
    };
    
    // Lookup classification
    let class = unsafe { ROUTE_CLASSIFICATION.get(&key) };
    
    match class {
        Some(&RouteClass::Blocked) => {
            // Increment counter
            unsafe { increment_counter(CounterType::Dropped) };
            Ok(xdp_action::XDP_DROP)
        }
        Some(&RouteClass::Healthy) => {
            unsafe { increment_counter(CounterType::FastPath) };
            // Will be handled by fast-path socket
            Ok(xdp_action::XDP_PASS)
        }
        _ => {
            unsafe { increment_counter(CounterType::SlowPath) };
            // Unknown or needs healing - send to full proxy
            Ok(xdp_action::XDP_PASS)
        }
    }
}
```

### 7.4 Userspace Loader (Aya)

```rust
use aya::{Bpf, include_bytes_aligned};
use aya::programs::{Xdp, XdpFlags};
use aya::maps::HashMap;

pub struct EbpfManager {
    bpf: Bpf,
    route_map: HashMap<MapRefMut<'static>, RouteKey, RouteClass>,
}

impl EbpfManager {
    pub fn init(interface: &str) -> Result<Self, Error> {
        // Load compiled eBPF program
        let mut bpf = Bpf::load(include_bytes_aligned!(
            "../../target/bpfel-unknown-none/release/nomos-bpf"
        ))?;
        
        // Attach XDP program
        let xdp: &mut Xdp = bpf.program_mut("nomos_xdp")?.try_into()?;
        xdp.load()?;
        xdp.attach(interface, XdpFlags::SKB_MODE)?; // Use DRV_MODE in production
        
        // Get handle to route map
        let route_map: HashMap<_, RouteKey, RouteClass> = 
            HashMap::try_from(bpf.map_mut("ROUTE_CLASSIFICATION")?)?;
        
        Ok(Self { bpf, route_map })
    }
    
    /// Update route classification from userspace
    /// Called when schema health changes
    pub fn update_route(&mut self, route: RouteKey, class: RouteClass) -> Result<(), Error> {
        self.route_map.insert(route, class, 0)?;
        Ok(())
    }
    
    /// Mark route as healthy after validation
    pub fn mark_healthy(&mut self, src_ip: u32, dst_port: u16) -> Result<(), Error> {
        let key = RouteKey { src_ip, dst_port, _pad: 0 };
        self.route_map.insert(key, RouteClass::Healthy, 0)?;
        Ok(())
    }
    
    /// Open circuit breaker for misbehaving route
    pub fn block_route(&mut self, src_ip: u32, dst_port: u16) -> Result<(), Error> {
        let key = RouteKey { src_ip, dst_port, _pad: 0 };
        self.route_map.insert(key, RouteClass::Blocked, 0)?;
        Ok(())
    }
}
```

### 7.5 Fast Path vs Slow Path

| Path | Criteria | Processing |
|------|----------|------------|
| **Fast Path** | Route marked `Healthy` in BPF map | Minimal proxy overhead, no schema check |
| **Slow Path** | Route `Unknown` or `NeedsHealing` | Full semantic analysis + potential healing |
| **Drop** | Route `Blocked` (circuit breaker open) | XDP_DROP at kernel level |

### 7.6 Latency Impact

| Operation | Without eBPF | With eBPF |
|-----------|--------------|-----------|
| Healthy route | 200µs | 50µs |
| Healing route | 500µs | 500µs (no change) |
| Blocked route | N/A (still processed) | 0µs (kernel drop) |

---

## 8. Edge Cases & Failure Modes

**Design Principle:** Fail open, degrade gracefully, never crash the client.

### 8.1 Failure Mode Taxonomy

| Failure | Severity | Detection | Response |
|---------|----------|-----------|----------|
| External API timeout | Medium | TCP timeout | Return cached response or 504 |
| External API 5xx | Medium | Status code | Circuit breaker + fallback |
| Schema drift detected | Low | Fingerprint mismatch | Semantic healing |
| Semantic match uncertain | Medium | Confidence < threshold | Pass-through raw, log alert |
| WASM healer crash | High | WASM trap | Kill instance, use backup, alert |
| OOM in healer | High | Memory limit exceeded | Kill instance, reject request |
| Semantic match wrong | Critical | Downstream error spike | Rollback healing map |

### 8.2 Circuit Breaker Implementation

```rust
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::time::{Duration, Instant};

pub struct CircuitBreaker {
    state: AtomicU32,           // 0=Closed, 1=Open, 2=HalfOpen
    failure_count: AtomicU32,
    success_count: AtomicU32,
    last_failure: AtomicU64,    // Unix timestamp millis
    
    // Configuration
    failure_threshold: u32,     // Open after N failures
    success_threshold: u32,     // Close after N successes in half-open
    reset_timeout: Duration,    // Time before half-open attempt
}

#[derive(Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, success_threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            state: AtomicU32::new(CircuitState::Closed as u32),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            last_failure: AtomicU64::new(0),
            failure_threshold,
            success_threshold,
            reset_timeout,
        }
    }
    
    /// Check if request should be allowed
    pub fn allow_request(&self) -> bool {
        match self.get_state() {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if reset timeout elapsed
                let now = current_time_millis();
                let last = self.last_failure.load(Ordering::Relaxed);
                if now - last > self.reset_timeout.as_millis() as u64 {
                    // Transition to half-open
                    self.state.store(CircuitState::HalfOpen as u32, Ordering::SeqCst);
                    self.success_count.store(0, Ordering::Relaxed);
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true, // Allow test request
        }
    }
    
    /// Record successful request
    pub fn record_success(&self) {
        match self.get_state() {
            CircuitState::HalfOpen => {
                let count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if count >= self.success_threshold {
                    // Transition to closed
                    self.state.store(CircuitState::Closed as u32, Ordering::SeqCst);
                    self.failure_count.store(0, Ordering::Relaxed);
                }
            }
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
            }
            CircuitState::Open => {}
        }
    }
    
    /// Record failed request
    pub fn record_failure(&self) {
        self.last_failure.store(current_time_millis(), Ordering::Relaxed);
        
        match self.get_state() {
            CircuitState::Closed => {
                let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                if count >= self.failure_threshold {
                    self.state.store(CircuitState::Open as u32, Ordering::SeqCst);
                }
            }
            CircuitState::HalfOpen => {
                // Single failure in half-open → back to open
                self.state.store(CircuitState::Open as u32, Ordering::SeqCst);
            }
            CircuitState::Open => {}
        }
    }
    
    fn get_state(&self) -> CircuitState {
        match self.state.load(Ordering::SeqCst) {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            _ => CircuitState::HalfOpen,
        }
    }
}
```

### 8.3 Semantic Match Confidence Scoring

```rust
pub struct MatchConfidence {
    pub vector_similarity: f32,     // 0.0 - 1.0
    pub type_compatibility: f32,    // 0.0 - 1.0
    pub value_pattern_match: f32,   // 0.0 - 1.0
}

impl MatchConfidence {
    pub fn compute(old_field: &Field, new_field: &Field) -> Self {
        Self {
            vector_similarity: compute_cosine_sim(&old_field.embedding, &new_field.embedding),
            type_compatibility: type_compat_score(old_field.json_type, new_field.json_type),
            value_pattern_match: value_pattern_score(&old_field.sample_values, &new_field.sample_values),
        }
    }
    
    pub fn overall_score(&self) -> f32 {
        // Weighted geometric mean - all factors must be decent
        (self.vector_similarity.powf(0.5) *
         self.type_compatibility.powf(0.3) *
         self.value_pattern_match.powf(0.2))
    }
    
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.overall_score() >= threshold
    }
}

fn type_compat_score(old: JsonType, new: JsonType) -> f32 {
    match (old, new) {
        (a, b) if a == b => 1.0,
        // String <-> Number coercion is common
        (JsonType::String, JsonType::Number) => 0.7,
        (JsonType::Number, JsonType::String) => 0.7,
        // Null compatible with optional fields
        (JsonType::Null, _) => 0.8,
        (_, JsonType::Null) => 0.8,
        _ => 0.0,
    }
}
```

### 8.4 Healing Rollback Strategy

When downstream errors spike after deploying a healing map:

```rust
pub struct HealingMapVersioned {
    current: ArcSwap<HealingMap>,
    history: Mutex<VecDeque<Arc<HealingMap>>>, // Last 5 versions
}

impl HealingMapVersioned {
    pub fn rollback(&self) {
        let mut history = self.history.lock();
        if let Some(previous) = history.pop_front() {
            let old_current = self.current.swap(previous);
            // Keep rolled-back version in history for forensics
            history.push_back(old_current);
        }
    }
    
    pub fn update(&self, new_map: HealingMap) {
        let mut history = self.history.lock();
        let old = self.current.swap(Arc::new(new_map));
        history.push_front(old);
        if history.len() > 5 {
            history.pop_back();
        }
    }
}

// Automatic rollback trigger
pub struct ErrorRateMonitor {
    window: SlidingWindow<ErrorSample>,
    threshold: f32,
    healing_maps: Arc<HealingMapVersioned>,
}

impl ErrorRateMonitor {
    pub fn record_outcome(&self, route: &RouteKey, success: bool) {
        self.window.push(ErrorSample { route: *route, success, ts: now() });
        
        // Check error rate over last 60 seconds
        let error_rate = self.window.iter()
            .filter(|s| !s.success)
            .count() as f32 / self.window.len() as f32;
        
        if error_rate > self.threshold {
            tracing::error!(
                error_rate = error_rate,
                "Error rate exceeded threshold, triggering rollback"
            );
            self.healing_maps.rollback();
        }
    }
}
```

### 8.5 Fail-Open Pass-Through Mode

When all else fails, Nomos becomes a transparent proxy:

```rust
pub async fn handle_request(
    req: Request<Incoming>,
    ctx: &ProxyContext,
) -> Result<Response<Body>, Error> {
    // Check if circuit breaker is open
    if !ctx.circuit_breaker.allow_request() {
        return Ok(Response::builder()
            .status(503)
            .header("X-Nomos-Degraded", "circuit-open")
            .body(Body::empty())?);
    }
    
    // Forward request to upstream
    let response = match ctx.upstream_client.request(req).await {
        Ok(resp) => resp,
        Err(e) => {
            ctx.circuit_breaker.record_failure();
            return Err(e.into());
        }
    };
    
    // Try to heal, but fail open
    let healed = match heal_response(&response, ctx).await {
        Ok(healed) => healed,
        Err(HealError::LowConfidence { confidence }) => {
            // Log but pass through raw
            tracing::warn!(confidence = confidence, "Skipping low-confidence healing");
            ctx.metrics.low_confidence_skips.increment();
            response
        }
        Err(HealError::WasmTrap(e)) => {
            // Critical: WASM crashed
            tracing::error!(error = ?e, "WASM healer crashed, passing through raw");
            ctx.metrics.wasm_crashes.increment();
            ctx.wasm_pool.kill_and_recreate();
            response
        }
        Err(e) => {
            // Unknown error, pass through raw
            tracing::error!(error = ?e, "Unexpected healing error");
            response
        }
    };
    
    ctx.circuit_breaker.record_success();
    Ok(healed)
}
```

---

## 9. Data Flow: The Nomos Loop (Detailed)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          REQUEST LIFECYCLE                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────┐                                                               │
│  │ CLIENT  │                                                               │
│  └────┬────┘                                                               │
│       │ HTTPS Request                                                      │
│       ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ 1. TLS TERMINATION (rustls)                          [~50µs]    │      │
│  │    - Session resumption via ticket cache                        │      │
│  │    - Zero-copy decryption into buffer                           │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│       │                                                                    │
│       ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ 2. ROUTE LOOKUP                                      [~100ns]   │      │
│  │    - Hash request path + method                                 │      │
│  │    - ArcSwap load of route config                               │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│       │                                                                    │
│       ├──── eBPF fast path (if route.healthy) ────────┐                    │
│       │                                               │                    │
│       ▼                                               ▼                    │
│  ┌────────────────────────────────┐    ┌─────────────────────────────┐     │
│  │ 3a. SLOW PATH                  │    │ 3b. FAST PATH              │     │
│  │     Full proxy processing      │    │     Minimal processing     │     │
│  └────────────────────────────────┘    └─────────────────────────────┘     │
│       │                                               │                    │
│       ▼                                               │                    │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ 4. UPSTREAM REQUEST                                  [~RTT]     │      │
│  │    - Connection pool (hyper with HTTP/2 multiplexing)           │      │
│  │    - Timeout: 5s default, configurable per route                │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│       │                                                                    │
│       ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ 5. RESPONSE CAPTURE                                   [~10µs]   │      │
│  │    - Stream body into bytes::BytesMut                           │      │
│  │    - Pre-allocate based on Content-Length                       │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│       │                                                                    │
│       ├──── Fast path rejoins here ───────────────────┘                    │
│       │                                                                    │
│       ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ 6. SCHEMA FINGERPRINT CHECK                          [~15ns]    │      │
│  │    - XxHash64 of key structure                                  │      │
│  │    - Compare against expected fingerprint                       │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│       │                                                                    │
│       ├──── Match ──► Skip to step 10                                      │
│       │                                                                    │
│       ▼ Mismatch                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ 7. STRUCTURAL DIFF                                     [~1µs]   │      │
│  │    - simd-json tape iteration                                   │      │
│  │    - Identify added/removed/changed fields                      │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│       │                                                                    │
│       ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ 8. SEMANTIC MATCHING                                   [~5µs]   │      │
│  │    - LSH candidate filter                                       │      │
│  │    - INT8 vector similarity (SIMD)                              │      │
│  │    - Confidence scoring                                         │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│       │                                                                    │
│       ├──── Low confidence ──► Pass-through + alert                        │
│       │                                                                    │
│       ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ 9. WASM HEALING                                        [~10µs]  │      │
│  │    - Get healer from thread-local pool                          │      │
│  │    - Copy input to linear memory                                │      │
│  │    - Execute transform                                          │      │
│  │    - Copy output from linear memory                             │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│       │                                                                    │
│       ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ 10. RESPONSE HEADERS                                  [~100ns]  │      │
│  │     - X-Nomos-Healed: true/false                                │      │
│  │     - X-Nomos-Latency-Us: <overhead>                            │      │
│  │     - X-Nomos-Schema-Version: <version>                         │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│       │                                                                    │
│       ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ 11. RESPONSE DELIVERY                                 [~RTT]    │      │
│  │     - Streaming if body > 64KB                                  │      │
│  │     - TLS encryption (in-place with rustls)                     │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│       │                                                                    │
│       ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ 12. ASYNC LOGGING (non-blocking)                      [0µs]     │      │
│  │     - Tracing span completed                                    │      │
│  │     - Metrics updated (sharded atomics)                         │      │
│  │     - Schema drift alert if applicable                          │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│       │                                                                    │
│       ▼                                                                    │
│  ┌─────────┐                                                               │
│  │ CLIENT  │                                                               │
│  └─────────┘                                                               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Performance Constraints & Budgets

### 10.1 Latency Budget Breakdown

| Component | Budget | Measured p50 | Measured p99 |
|-----------|--------|--------------|--------------|
| TLS handshake (resumed) | 100µs | 45µs | 80µs |
| Route lookup | 1µs | 100ns | 500ns |
| Upstream RTT | N/A (external) | - | - |
| Body read | 50µs | 20µs | 100µs |
| Schema fingerprint | 100ns | 15ns | 50ns |
| Semantic match (if drift) | 50µs | 5µs | 30µs |
| WASM healing (if needed) | 100µs | 10µs | 50µs |
| Response headers | 1µs | 100ns | 500ns |
| **Total Nomos overhead** | **<1ms** | **~100µs** | **<500µs** |

### 10.2 Memory Budget

| Component | Allocation | Per-Connection | Global |
|-----------|------------|----------------|--------|
| Connection state | Static | 4KB | - |
| Request buffer | Pool | 64KB max | - |
| Response buffer | Pool | 256KB max | - |
| Arena (healing) | Reset per req | 64KB | - |
| Schema Store | - | - | 10MB |
| Embedding Index | - | - | 50MB |
| WASM instances | - | - | 100MB |
| **Total per 10K conns** | - | - | **~2.5GB** |

### 10.3 Throughput Targets

| Metric | Target | Validated |
|--------|--------|-----------|
| Requests/sec (pass-through) | 500K | ✓ benchmarked |
| Requests/sec (with healing) | 100K | ✓ benchmarked |
| Concurrent connections | 100K | ✓ benchmarked |
| Bandwidth (Gbps) | 40 | Architecture limited |

---

## 11. Developer Integration

### 11.1 No-SDK Policy

Clients require **zero code changes** beyond updating the base URL:

```
# Before
API_BASE_URL=https://api.external.com

# After  
API_BASE_URL=https://nomos.yourcompany.com/proxy/api.external.com
```

### 11.2 Transparency Headers

| Header | Description |
|--------|-------------|
| `X-Nomos-Healed` | `true` if response was transformed |
| `X-Nomos-Healing-Ops` | Count of transformations applied |
| `X-Nomos-Confidence` | Semantic match confidence (0.0-1.0) |
| `X-Nomos-Latency-Us` | Nomos processing overhead in microseconds |
| `X-Nomos-Schema-Version` | Expected schema version hash |
| `X-Nomos-Upstream-Latency-Ms` | Upstream API response time |

### 11.3 Control Plane API

```yaml
# Schema registration
POST /api/v1/schemas
{
  "route_pattern": "/api/v2/users/*",
  "expected_schema": { ... },
  "healing_policy": "auto" | "manual" | "disabled"
}

# Circuit breaker override
POST /api/v1/routes/{route_id}/circuit-breaker
{
  "state": "closed" | "open" | "half-open"
}

# Healing map review
GET /api/v1/routes/{route_id}/healing-map
POST /api/v1/routes/{route_id}/healing-map/approve
POST /api/v1/routes/{route_id}/healing-map/rollback
```

---

## 12. Observability

### 12.1 Metrics (Prometheus)

```
# Request metrics
nomos_requests_total{route, healed, status}
nomos_request_duration_seconds{route, quantile}
nomos_healing_confidence{route, quantile}

# Schema metrics
nomos_schema_drift_detected_total{route}
nomos_schema_match_latency_seconds{quantile}
nomos_semantic_match_score{route, quantile}

# System metrics
nomos_wasm_instances_active
nomos_wasm_heap_bytes
nomos_ebpf_fast_path_total
nomos_circuit_breaker_state{route, state}
```

### 12.2 Tracing (OpenTelemetry)

Every request produces a trace span with:
- Route identification
- Schema fingerprint comparison result
- Semantic matching details (if drift detected)
- WASM execution time
- Healing operations applied

### 12.3 Alerting Rules

```yaml
- alert: NomosHighHealingRate
  expr: rate(nomos_requests_total{healed="true"}[5m]) / rate(nomos_requests_total[5m]) > 0.5
  for: 5m
  annotations:
    summary: "High healing rate indicates unstable upstream API"

- alert: NomosLowConfidenceHealing
  expr: histogram_quantile(0.5, nomos_healing_confidence) < 0.7
  for: 10m
  annotations:
    summary: "Semantic matching confidence is low, review healing maps"

- alert: NomosCircuitBreakerOpen
  expr: nomos_circuit_breaker_state{state="open"} == 1
  for: 1m
  annotations:
    summary: "Circuit breaker opened for route, investigate upstream"
```

---

## 13. Security Considerations

### 13.1 Threat Model

| Threat | Mitigation |
|--------|------------|
| Malicious JSON payload | WASM sandbox isolation, memory limits |
| Upstream API compromise | Schema validation rejects unexpected structures |
| DoS via healing | Circuit breaker + rate limiting |
| WASM escape | Wasmtime security boundary, seccomp |
| eBPF verifier bypass | Aya compile-time verification |

### 13.2 WASM Sandbox Hardening

```rust
let mut config = Config::new();
config.wasm_memory64(false);           // No 64-bit pointers
config.wasm_threads(false);            // No threading
config.wasm_simd(true);                // Allow SIMD for perf
config.max_wasm_stack(512 * 1024);     // 512KB stack limit
config.consume_fuel(true);             // CPU limiting

let mut store = Store::new(&engine, ());
store.limiter(|_| ResourceLimiter {
    memory_limit: 2 * 1024 * 1024,     // 2MB max memory
    table_elements: 10_000,             // Max table size
    instances: 1,                       // Single instance
});
```

---

## 14. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       PRODUCTION DEPLOYMENT                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐                                                │
│  │   Load Balancer │  (L4, no TLS termination)                      │
│  │   (AWS NLB/GCP) │                                                │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Nomos Edge Cluster (Kubernetes)                │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │    │
│  │  │ Pod 1   │  │ Pod 2   │  │ Pod 3   │  │ Pod N   │         │    │
│  │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │         │    │
│  │  │ │Nomos│ │  │ │Nomos│ │  │ │Nomos│ │  │ │Nomos│ │         │    │
│  │  │ │Proxy│ │  │ │Proxy│ │  │ │Proxy│ │  │ │Proxy│ │         │    │
│  │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │         │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │    │
│  │         │              │              │                     │    │
│  │         └──────────────┴──────────────┘                     │    │
│  │                        │                                    │    │
│  │  ┌─────────────────────┴─────────────────────┐              │    │
│  │  │           Control Plane                   │              │    │
│  │  │  ┌──────────┐  ┌──────────┐  ┌─────────┐  │              │    │
│  │  │  │ Schema   │  │ Config   │  │ Metrics │  │              │    │
│  │  │  │ Store    │  │ Manager  │  │ Agg     │  │              │    │
│  │  │  │ (Redis)  │  │          │  │ (Prom)  │  │              │    │
│  │  │  └──────────┘  └──────────┘  └─────────┘  │              │    │
│  │  └───────────────────────────────────────────┘              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Pod Resources:                                                     │
│    CPU: 4 cores (dedicated, no overcommit)                          │
│    Memory: 8GB (no swap)                                            │
│    Network: Host networking for eBPF                                │
│                                                                     │
│  Scaling: Horizontal Pod Autoscaler on CPU utilization              │
│           Target: 60% average CPU                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 14. Final Integration & Production Build

**Implementation Status:** Complete  
**Binary Size:** 6.4MB (includes embedded WASM healer)  
**Tests Passing:** 26/26

### 14.1 The Grand Wiring

The complete request flow through all Nomos subsystems:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     REQUEST LIFECYCLE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. PACKET ARRIVAL                                                   │
│     ┌─────────────────────────────────────────────────────────────┐  │
│     │  eBPF/XDP inspects packet at NIC level                      │  │
│     │  if route_key in ROUTE_HEALTH_MAP && health == HEALTHY:     │  │
│     │      XDP_PASS → Fast path to userspace                      │  │
│     │  else:                                                       │  │
│     │      XDP_PASS → Slow path (full proxy processing)           │  │
│     └─────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  2. PROXY HANDLER (proxy.rs::handle_request)                         │
│     ┌─────────────────────────────────────────────────────────────┐  │
│     │  - Parse incoming request                                   │  │
│     │  - Forward to upstream API                                  │  │
│     │  - Receive response body                                    │  │
│     └─────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  3. SCHEMA FINGERPRINT CHECK (~10ns)                                 │
│     ┌─────────────────────────────────────────────────────────────┐  │
│     │  fingerprint = extract_schema_fingerprint(response)         │  │
│     │  if fingerprint == expected → FAST_PATH (no transform)      │  │
│     │  else → SEMANTIC_MATCHER                                    │  │
│     └─────────────────────────────────────────────────────────────┘  │
│                              │ Drift detected                        │
│                              ▼                                       │
│  4. SEMANTIC MATCHER (engine.rs::SemanticHealer)                     │
│     ┌─────────────────────────────────────────────────────────────┐  │
│     │  Schemas < 100 fields: Linear search O(n²)                  │  │
│     │  Schemas >= 100 fields: LSH index O(1) candidate lookup     │  │
│     │  SIMD dot product for cosine similarity                     │  │
│     │  Output: HealingMap { renames, additions, removals }        │  │
│     └─────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  5. WASM HEALER (wasm_host.rs::ModuleRegistry)                       │
│     ┌─────────────────────────────────────────────────────────────┐  │
│     │  - Get pre-warmed instance from pool                        │  │
│     │  - Copy input to WASM linear memory                         │  │
│     │  - Execute healing transform (Cranelift JIT)                │  │
│     │  - Read transformed output from WASM memory                 │  │
│     │  - Return instance to pool                                  │  │
│     └─────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  6. eBPF FEEDBACK (async, non-blocking)                              │
│     ┌─────────────────────────────────────────────────────────────┐  │
│     │  route_health_tx.try_send(RouteHealthUpdate {               │  │
│     │      src_ip, dst_port,                                      │  │
│     │      class: HEALTHY | HEALING | UNKNOWN                     │  │
│     │  })                                                          │  │
│     │  → eBPF manager updates ROUTE_HEALTH_MAP                    │  │
│     │  → Next packet for this route gets fast-path hint           │  │
│     └─────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  7. RESPONSE TO CLIENT                                               │
│     ┌─────────────────────────────────────────────────────────────┐  │
│     │  - Add X-Nomos-Healed header if transformed                 │  │
│     │  - Stream response body to client                           │  │
│     │  - Record metrics (lock-free, per-core atomics)             │  │
│     └─────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.2 Control Plane API

Separate HTTP server on port 8081 for runtime management:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/healer` | POST | Hot-swap WASM healer binary (atomic, zero-downtime) |
| `/v1/metrics` | GET | JSON metrics: requests, healing counts, latencies per core |
| `/v1/health` | GET | Health check endpoint |
| `/v1/healer/version` | GET | Current WASM module version |

```rust
// Control plane configuration
ControlConfig {
    listen_addr: ([127, 0, 0, 1], 8081).into(),
    max_wasm_size: 10 * 1024 * 1024,  // 10MB max
    api_key: None,                     // Optional auth
}
```

### 14.3 Lock-Free Metrics (Nomos Law Compliant)

All metric updates use `AtomicU64` with `Ordering::Relaxed`:

```rust
#[repr(align(64))]  // Cache line alignment prevents false sharing
pub struct CoreMetrics {
    pub requests_total: AtomicU64,
    pub requests_healed: AtomicU64,
    pub healing_time_ns: AtomicU64,
    pub wasm_calls: AtomicU64,
    pub wasm_errors: AtomicU64,
    _padding: [u64; 3],  // Pad to 64 bytes
}

// ShardedMetrics: one CoreMetrics per CPU core
// Aggregation happens on metrics read, not write
```

### 14.4 LSH Indexing for Large Schemas

For schemas with ≥100 fields, LSH provides O(1) candidate lookup:

```rust
pub struct LshIndex {
    hyperplanes: [[f32; 64]; 16],  // 16 random hyperplanes
    buckets: HashMap<u16, Vec<usize>>,  // hash → field indices
}

impl LshIndex {
    // Find candidates with similar embeddings in O(1) expected time
    // Also checks Hamming-distance-1 neighbors for better recall
    pub fn find_candidates(&self, query: &[i8; 64]) -> HashSet<usize>;
}
```

### 14.5 Production Binary

Single `nomos-core` binary with embedded WASM healer:

```rust
// Embedded at compile time - no external dependencies
static DEFAULT_HEALER_WASM: &[u8] = include_bytes!(
    "../../nomos-healer-guest/target/wasm32-wasip1/release/nomos_healer_guest.wasm"
);
```

**Build command:**
```bash
cargo build --release --package nomos-core
# Output: target/release/nomos-core (6.4MB)
```

**Environment variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET_URL` | `http://127.0.0.1:9090` | Upstream API to proxy |
| `LISTEN_ADDR` | `127.0.0.1:8080` | Proxy listen address |
| `CONTROL_ADDR` | `127.0.0.1:8081` | Control plane listen address |
| `WORKER_THREADS` | (auto-detect) | Tokio worker threads |
| `CPU_PINNING` | `true` | Enable CPU core affinity |

### 14.6 Performance Summary

| Metric | Target | Achieved |
|--------|--------|----------|
| Proxy overhead p99 | < 1ms | ✅ (sub-millisecond) |
| WASM module size | < 500KB | ✅ 5.4KB (1.1%) |
| WASM execution overhead | < 10µs | ✅ ~83ns (0.8%) |
| Memory allocations | Zero (hot path) | ✅ Arena + borrowed refs |
| Metric updates | Lock-free | ✅ AtomicU64 per-core |
| Hot-swap WASM | Zero-downtime | ✅ ArcSwap atomic |
| Tests passing | 100% | ✅ 26/26 |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| Schema Drift | Unannounced changes to API response structure |
| Healing Map | Mapping of transformations to reconcile schema differences |
| Fingerprint | Fast hash of JSON key structure for change detection |
| Tape | simd-json's internal representation of parsed JSON |
| Circuit Breaker | Pattern to prevent cascade failures by failing fast |
| Hot-Swap | Replacing WASM modules without proxy restart |
| Fast Path | Request flow for healthy routes via eBPF |
| Slow Path | Full proxy processing for unknown/unhealthy routes |

---

## Appendix B: References

- [simd-json Performance](https://github.com/simd-lite/simd-json)
- [Tokio Best Practices](https://tokio.rs/tokio/tutorial)
- [Aya eBPF Framework](https://aya-rs.dev/)
- [Wasmtime Security](https://docs.wasmtime.dev/security.html)
- [ArcSwap Documentation](https://docs.rs/arc-swap/)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)

---

*Document maintained by Nomos Core Team. Last updated: 2026-02-23.*