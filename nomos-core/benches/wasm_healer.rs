//! WASM Healer Benchmark
//!
//! Compares performance of:
//! 1. Native Semantic Engine (from Phase 2)
//! 2. WASM-Wrapped Healer (Phase 3)
//!
//! Nomos Law Target: WASM overhead must be < 10µs
//!
//! # Running
//! ```bash
//! # First, build the WASM guest
//! cd nomos-healer-guest
//! cargo build --target wasm32-wasip1 --release
//!
//! # Then run benchmark with WASM module path
//! WASM_MODULE=../target/wasm32-wasip1/release/nomos_healer_guest.wasm \
//!     cargo bench --package nomos-core --bench wasm_healer
//! ```

use std::sync::Arc;
use std::time::Duration;

use bumpalo::Bump;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use nomos_core::engine::{HealingMap, HealingOp, SemanticHealer};
use nomos_core::schema::SchemaFingerprint;

// ============================================================================
// Test Data Generation
// ============================================================================

/// Generate test JSON with given number of fields
fn generate_test_json(field_count: usize) -> String {
    let mut fields = Vec::with_capacity(field_count);
    for i in 0..field_count {
        fields.push(format!(r#""field_{}": "value_{}""#, i, i));
    }
    format!("{{{}}}", fields.join(", "))
}

/// Generate healing operations (field renames)
fn generate_healing_ops(rename_count: usize) -> Vec<HealingOp> {
    (0..rename_count)
        .map(|i| HealingOp::Rename {
            from: format!("field_{}", i).into(),
            to: format!("new_field_{}", i).into(),
            confidence: 0.95,
        })
        .collect()
}

/// Generate drifted JSON (some fields renamed)
fn generate_drifted_json(field_count: usize, drift_percent: usize) -> String {
    let drift_count = (field_count * drift_percent) / 100;
    let mut fields = Vec::with_capacity(field_count);
    
    for i in 0..field_count {
        if i < drift_count {
            // Drifted field (different name)
            fields.push(format!(r#""drifted_field_{}": "value_{}""#, i, i));
        } else {
            // Normal field
            fields.push(format!(r#""field_{}": "value_{}""#, i, i));
        }
    }
    format!("{{{}}}", fields.join(", "))
}

// ============================================================================
// Native Healing Benchmark
// ============================================================================

fn bench_native_healing(c: &mut Criterion) {
    let mut group = c.benchmark_group("native_healing");
    group.measurement_time(Duration::from_secs(5));
    
    let healer = SemanticHealer::new();
    
    // Test configurations: (fields, rename_count)
    let configs = [
        (10, 2),
        (50, 10),
        (100, 20),
    ];
    
    for (fields, renames) in configs {
        let json = generate_test_json(fields);
        let ops = generate_healing_ops(renames);
        let healing_map = HealingMap {
            operations: ops,
            confidence: 0.95,
            source_fingerprint: SchemaFingerprint::default(),
            target_fingerprint: SchemaFingerprint::default(),
        };
        
        group.throughput(Throughput::Bytes(json.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("transform", format!("{}f_{}r", fields, renames)),
            &(json.as_bytes(), &healing_map),
            |b, (json, map)| {
                b.iter_custom(|iters| {
                    let start = std::time::Instant::now();
                    for _ in 0..iters {
                        let arena = Bump::with_capacity(65536);
                        let result = healer.apply_healing(black_box(*json), black_box(map), &arena);
                        black_box(result);
                    }
                    start.elapsed()
                });
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// WASM Healing Benchmark (requires built WASM module)
// ============================================================================

/// Helper to load WASM module bytes
fn load_wasm_module() -> Option<Vec<u8>> {
    let path = std::env::var("WASM_MODULE").ok()?;
    std::fs::read(&path).ok()
}

fn bench_wasm_healing(c: &mut Criterion) {
    let wasm_bytes = match load_wasm_module() {
        Some(bytes) => bytes,
        None => {
            eprintln!("WASM_MODULE env var not set, skipping WASM benchmarks");
            eprintln!("Build with: cargo build -p nomos-healer-guest --target wasm32-wasip1 --release");
            return;
        }
    };
    
    let mut group = c.benchmark_group("wasm_healing");
    group.measurement_time(Duration::from_secs(5));
    
    // Report WASM module size
    println!("\nWASM module size: {} bytes ({:.1} KB)", 
             wasm_bytes.len(), 
             wasm_bytes.len() as f64 / 1024.0);
    
    // Check target: < 500KB
    if wasm_bytes.len() > 500 * 1024 {
        eprintln!("WARNING: WASM module exceeds 500KB target!");
    }
    
    // Create healer
    let healer = match nomos_core::wasm_host::WasmHealer::new(&wasm_bytes) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Failed to create WASM healer: {:?}", e);
            return;
        }
    };
    
    let configs = [
        (10, 2),
        (50, 10),
        (100, 20),
    ];
    
    for (fields, renames) in configs {
        let json = generate_test_json(fields);
        let ops = generate_healing_ops(renames);
        
        group.throughput(Throughput::Bytes(json.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("transform", format!("{}f_{}r", fields, renames)),
            &(json.as_bytes().to_vec(), &ops),
            |b, (json, ops)| {
                b.iter(|| {
                    let result = healer.heal(black_box(json), black_box(ops));
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// Overhead Comparison Benchmark
// ============================================================================

fn bench_wasm_overhead(c: &mut Criterion) {
    let wasm_bytes = match load_wasm_module() {
        Some(bytes) => bytes,
        None => {
            return;
        }
    };
    
    let mut group = c.benchmark_group("wasm_overhead");
    group.measurement_time(Duration::from_secs(5));
    
    // Small payload to isolate overhead from transform time
    let json = r#"{"a": 1, "b": 2}"#;
    let ops = vec![HealingOp::Rename {
        from: "a".into(),
        to: "x".into(),
        confidence: 0.95,
    }];
    
    let healing_map = HealingMap {
        operations: ops.clone(),
        confidence: 0.95,
        source_fingerprint: SchemaFingerprint::default(),
        target_fingerprint: SchemaFingerprint::default(),
    };
    
    let native_healer = SemanticHealer::new();
    
    // Native baseline
    group.bench_function("native_minimal", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let arena = Bump::with_capacity(4096);
                let result = native_healer.apply_healing(
                    black_box(json.as_bytes()), 
                    black_box(&healing_map),
                    &arena,
                );
                black_box(result);
            }
            start.elapsed()
        });
    });
    
    // WASM version
    let healer = nomos_core::wasm_host::WasmHealer::new(&wasm_bytes).unwrap();
    
    group.bench_function("wasm_minimal", |b| {
        b.iter(|| {
            let result = healer.heal(black_box(json.as_bytes()), black_box(&ops));
            black_box(result);
        });
    });
    
    // Instance creation overhead
    let registry = Arc::new(nomos_core::wasm_host::ModuleRegistry::new(&wasm_bytes).unwrap());
    
    group.bench_function("instance_creation", |b| {
        b.iter(|| {
            let instance = registry.create_instance();
            black_box(instance);
        });
    });
    
    // Pool acquisition overhead (should be ~0)
    let pool = nomos_core::wasm_host::HealerPool::new(Arc::clone(&registry), 4);
    
    group.bench_function("pool_acquire", |b| {
        b.iter(|| {
            let mut pooled = pool.get().unwrap();
            let healer = pooled.healer();
            black_box(healer);
            // Dropped here, returns to pool
        });
    });
    
    group.finish();
}

// ============================================================================
// Fingerprint Validation Benchmark
// ============================================================================

fn bench_fingerprint_validation(c: &mut Criterion) {
    let wasm_bytes = match load_wasm_module() {
        Some(bytes) => bytes,
        None => return,
    };
    
    let mut group = c.benchmark_group("fingerprint");
    group.measurement_time(Duration::from_secs(3));
    
    let healer = nomos_core::wasm_host::WasmHealer::new(&wasm_bytes).unwrap();
    
    // Various JSON sizes
    for &size in &[100, 1000, 5000] {
        let json = generate_test_json(size / 20); // ~20 bytes per field
        let fingerprint = 0x12345678u64; // Dummy fingerprint
        
        group.throughput(Throughput::Bytes(json.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("validate", format!("{}b", json.len())),
            &json,
            |b, json| {
                b.iter(|| {
                    let result = healer.validate_fingerprint(
                        black_box(json.as_bytes()),
                        black_box(fingerprint),
                    );
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// Hot-Swap Benchmark
// ============================================================================

fn bench_hot_swap(c: &mut Criterion) {
    let wasm_bytes = match load_wasm_module() {
        Some(bytes) => bytes,
        None => return,
    };
    
    let mut group = c.benchmark_group("hot_swap");
    group.measurement_time(Duration::from_secs(5));
    
    let registry = nomos_core::wasm_host::ModuleRegistry::new(&wasm_bytes).unwrap();
    
    // Hot-swap includes compilation time
    group.bench_function("swap_module", |b| {
        b.iter(|| {
            let version = registry.hot_swap(black_box(&wasm_bytes));
            black_box(version);
        });
    });
    
    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    benches,
    bench_native_healing,
    bench_wasm_healing,
    bench_wasm_overhead,
    bench_fingerprint_validation,
    bench_hot_swap,
);

criterion_main!(benches);
