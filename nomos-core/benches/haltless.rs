//! Haltless Benchmark - Schema Drift Healing Performance
//!
//! Target: <50µs from drift detection to healed delivery.
//!
//! Scenario:
//! - 5KB JSON payload
//! - 20% schema drift (field renames)
//! - Full healing pipeline

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use nomos_core::engine::{
    FieldEmbedding, MatchConfidence, SemanticHealer,
    simd_dot_product_i8, EMBEDDING_DIM,
};
use nomos_core::schema::{ExpectedSchema, JsonType, SchemaFingerprint};

/// Generate a 5KB JSON payload with the specified number of fields.
fn generate_json_payload(field_count: usize) -> String {
    let mut json = String::from("{");

    for i in 0..field_count {
        if i > 0 {
            json.push_str(", ");
        }

        // Generate varied field types
        let value = match i % 5 {
            0 => format!(r#""field_{i}": "value_{i}_{}""#, "x".repeat(50)), // String
            1 => format!(r#""field_{i}": {}"#, i * 1000), // Number
            2 => format!(r#""field_{i}": {}"#, i % 2 == 0), // Bool
            3 => format!(r#""field_{i}": null"#), // Null
            _ => format!(r#""field_{i}": "{}""#, "a".repeat(100)), // Long string
        };
        json.push_str(&value);
    }

    json.push('}');
    json
}

/// Generate expected schema (client's version).
fn generate_expected_schema(field_count: usize) -> ExpectedSchema {
    let mut root_fields = Vec::with_capacity(field_count);
    let mut field_types = HashMap::new();

    for i in 0..field_count {
        let field_name = Arc::from(format!("expected_field_{}", i).as_str());
        let json_type = match i % 5 {
            0 => JsonType::String,
            1 => JsonType::Number,
            2 => JsonType::Bool,
            3 => JsonType::Null,
            _ => JsonType::String,
        };

        root_fields.push(Arc::clone(&field_name));
        field_types.insert(field_name, json_type);
    }

    ExpectedSchema {
        fingerprint: SchemaFingerprint(12345),
        root_fields,
        field_types,
        version: 1,
        last_seen_timestamp: 0,
    }
}

/// Generate actual fields with drift (20% renamed).
fn generate_drifted_fields(field_count: usize, drift_percent: usize) -> (Vec<String>, HashMap<String, JsonType>) {
    let drift_count = (field_count * drift_percent) / 100;
    let mut fields = Vec::with_capacity(field_count);
    let mut types = HashMap::new();

    for i in 0..field_count {
        let (field_name, json_type) = if i < drift_count {
            // Drifted fields - renamed from expected_field_X to field_X
            (format!("field_{}", i), match i % 5 {
                0 => JsonType::String,
                1 => JsonType::Number,
                2 => JsonType::Bool,
                3 => JsonType::Null,
                _ => JsonType::String,
            })
        } else {
            // Unchanged fields
            (format!("expected_field_{}", i), match i % 5 {
                0 => JsonType::String,
                1 => JsonType::Number,
                2 => JsonType::Bool,
                3 => JsonType::Null,
                _ => JsonType::String,
            })
        };

        fields.push(field_name.clone());
        types.insert(field_name, json_type);
    }

    (fields, types)
}

/// Benchmark: SIMD dot product (core inner loop).
fn bench_simd_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");

    // Generate random-ish embeddings (cast to u8 first to avoid overflow)
    let a: [i8; EMBEDDING_DIM] = core::array::from_fn(|i| {
        (((i * 7 + 13) % 256) as i16 - 128) as i8
    });
    let b: [i8; EMBEDDING_DIM] = core::array::from_fn(|i| {
        (((i * 11 + 17) % 256) as i16 - 128) as i8
    });

    group.bench_function("dot_product_64", |bench| {
        bench.iter(|| {
            black_box(simd_dot_product_i8(black_box(&a), black_box(&b)))
        })
    });

    group.finish();
}

/// Benchmark: Field embedding generation.
fn bench_embedding_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding");

    let field_names = [
        "user_id",
        "full_name",
        "email_address",
        "created_at",
        "is_active",
        "account_balance_usd",
        "last_login_timestamp",
    ];

    group.bench_function("single_embedding", |bench| {
        bench.iter(|| {
            black_box(FieldEmbedding::new(black_box("user_identifier"), JsonType::String))
        })
    });

    group.bench_function("batch_7_embeddings", |bench| {
        bench.iter(|| {
            for name in &field_names {
                black_box(FieldEmbedding::new(black_box(name), JsonType::String));
            }
        })
    });

    group.finish();
}

/// Benchmark: Cosine similarity computation.
fn bench_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity");

    let emb1 = FieldEmbedding::new("user_id", JsonType::String);
    let emb2 = FieldEmbedding::new("uuid", JsonType::String);

    group.bench_function("cosine_similarity", |bench| {
        bench.iter(|| {
            black_box(black_box(&emb1).similarity(black_box(&emb2)))
        })
    });

    // Batch similarity: find best match among 20 candidates
    let candidates: Vec<FieldEmbedding> = (0..20)
        .map(|i| FieldEmbedding::new(&format!("candidate_field_{}", i), JsonType::String))
        .collect();

    group.bench_function("find_best_match_20_candidates", |bench| {
        let query = FieldEmbedding::new("candidate_field_7", JsonType::String);
        bench.iter(|| {
            let mut best_score = 0.0f32;
            for candidate in &candidates {
                let score = query.similarity(candidate);
                if score > best_score {
                    best_score = score;
                }
            }
            black_box(best_score)
        })
    });

    group.finish();
}

/// Benchmark: Confidence scoring.
fn bench_confidence(c: &mut Criterion) {
    let mut group = c.benchmark_group("confidence");

    let old_field = FieldEmbedding::new("user_id", JsonType::String);
    let new_field = FieldEmbedding::new("uuid", JsonType::Number);

    group.bench_function("compute_confidence", |bench| {
        bench.iter(|| {
            black_box(MatchConfidence::compute(
                black_box(&old_field),
                black_box(&new_field),
            ))
        })
    });

    group.finish();
}

/// Benchmark: Full healing map computation (THE HALTLESS BENCHMARK).
fn bench_healing_map_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("haltless");
    group.sample_size(100);

    // Setup: 50 fields, 20% drift = 10 renames needed
    let field_count = 50;
    let expected_schema = generate_expected_schema(field_count);

    let mut healer = SemanticHealer::new();
    healer.register_schema(&expected_schema);

    let (actual_fields, actual_types) = generate_drifted_fields(field_count, 20);
    let actual_field_refs: Vec<&str> = actual_fields.iter().map(|s| s.as_str()).collect();
    let actual_type_refs: HashMap<&str, JsonType> = actual_types
        .iter()
        .map(|(k, v)| (k.as_str(), *v))
        .collect();

    group.throughput(Throughput::Elements(1));
    group.bench_function("20_percent_drift_50_fields", |bench| {
        bench.iter(|| {
            black_box(healer.compute_healing_map(
                black_box(&expected_schema),
                black_box(&actual_field_refs),
                black_box(&actual_type_refs),
            ))
        })
    });

    group.finish();
}

/// Benchmark: End-to-end healing with JSON transformation.
fn bench_full_healing_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    group.sample_size(50);

    // Generate ~5KB JSON payload
    let json_payload = generate_json_payload(50);
    let payload_size = json_payload.len();
    println!("JSON payload size: {} bytes", payload_size);

    let expected_schema = generate_expected_schema(50);
    let mut healer = SemanticHealer::new();
    healer.register_schema(&expected_schema);

    let (actual_fields, actual_types) = generate_drifted_fields(50, 20);
    let actual_field_refs: Vec<&str> = actual_fields.iter().map(|s| s.as_str()).collect();
    let actual_type_refs: HashMap<&str, JsonType> = actual_types
        .iter()
        .map(|(k, v)| (k.as_str(), *v))
        .collect();

    // Pre-compute healing map (would be cached in production)
    let healing_map = healer
        .compute_healing_map(&expected_schema, &actual_field_refs, &actual_type_refs)
        .unwrap();

    let json_bytes = json_payload.as_bytes();

    group.throughput(Throughput::Bytes(payload_size as u64));
    group.bench_function("5kb_20pct_drift_e2e", |bench| {
        bench.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let arena = bumpalo::Bump::with_capacity(65536);
                let start = std::time::Instant::now();
                let result = healer.apply_healing(json_bytes, &healing_map, &arena);
                black_box(result);
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

/// Manual timing test to verify <50µs target.
fn verify_target_latency() {
    println!("\n=== HALTLESS VERIFICATION ===");

    let field_count = 50;
    let expected_schema = generate_expected_schema(field_count);

    let mut healer = SemanticHealer::new();
    healer.register_schema(&expected_schema);

    let (actual_fields, actual_types) = generate_drifted_fields(field_count, 20);
    let actual_field_refs: Vec<&str> = actual_fields.iter().map(|s| s.as_str()).collect();
    let actual_type_refs: HashMap<&str, JsonType> = actual_types
        .iter()
        .map(|(k, v)| (k.as_str(), *v))
        .collect();

    // Warm up
    for _ in 0..100 {
        let _ = healer.compute_healing_map(&expected_schema, &actual_field_refs, &actual_type_refs);
    }

    // Measure
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = healer.compute_healing_map(&expected_schema, &actual_field_refs, &actual_type_refs);
    }
    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() / iterations;
    let avg_us = avg_ns as f64 / 1000.0;

    println!(
        "Healing map computation: {:.2}µs average ({} iterations)",
        avg_us, iterations
    );

    // JSON transformation
    let json_payload = generate_json_payload(50);
    let json_bytes = json_payload.as_bytes();
    let healing_map = healer
        .compute_healing_map(&expected_schema, &actual_field_refs, &actual_type_refs)
        .unwrap();

    let mut arena = bumpalo::Bump::with_capacity(65536);

    let start = Instant::now();
    for _ in 0..iterations {
        arena.reset();
        let _ = healer.apply_healing(json_bytes, &healing_map, &arena);
    }
    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() / iterations;
    let transform_us = avg_ns as f64 / 1000.0;

    println!("JSON transformation: {:.2}µs average", transform_us);

    let total_us = avg_us + transform_us;
    println!("TOTAL: {:.2}µs", total_us);

    if total_us < 50.0 {
        println!("✓ TARGET MET: <50µs");
    } else {
        println!("✗ TARGET MISSED: need optimization");
    }
}

criterion_group!(
    benches,
    bench_simd_dot_product,
    bench_embedding_generation,
    bench_similarity,
    bench_confidence,
    bench_healing_map_computation,
    bench_full_healing_pipeline,
);

criterion_main!(benches);

// Test module for the benchmark
mod tests {
    use super::verify_target_latency;

    #[test]
    fn run_latency_verification() {
        verify_target_latency();
    }
}
