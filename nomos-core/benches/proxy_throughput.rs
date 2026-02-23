//! Criterion benchmarks for Nomos proxy components.
//!
//! These benchmarks measure the hot path operations to ensure
//! we stay within the sub-1ms overhead budget.
//!
//! Run with: cargo bench -p nomos-core

use std::sync::Arc;

use bumpalo::Bump;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use nomos_core::middleware::{MiddlewareContext, PassThroughMiddleware, ResponseMiddleware, FingerprintCheckMiddleware};
use nomos_core::schema::{ExpectedSchema, HttpMethod, JsonType, RouteKey, SchemaFingerprint, SchemaStore};

/// Sample JSON payloads of various sizes
const SMALL_JSON: &[u8] = br#"{"id":"123","name":"test"}"#;

const MEDIUM_JSON: &[u8] = br#"{
    "user_id": "usr_12345abcdef",
    "email": "user@example.com",
    "name": "Test User",
    "created_at": "2024-01-01T00:00:00Z",
    "profile": {
        "avatar_url": "https://example.com/avatar.png",
        "bio": "This is a test user bio",
        "location": "San Francisco, CA"
    },
    "settings": {
        "notifications": true,
        "theme": "dark",
        "language": "en-US"
    }
}"#;

const LARGE_JSON: &[u8] = include_bytes!("../benches/testdata/large.json");

fn make_test_route() -> RouteKey {
    RouteKey {
        method: HttpMethod::Get,
        path: Arc::from("/api/v1/users"),
    }
}

fn make_test_schema() -> ExpectedSchema {
    ExpectedSchema {
        fingerprint: SchemaFingerprint::from_fields(&["id", "name", "email", "created_at"]),
        root_fields: vec![
            Arc::from("id"),
            Arc::from("name"),
            Arc::from("email"),
            Arc::from("created_at"),
        ],
        field_types: [
            (Arc::from("id"), JsonType::String),
            (Arc::from("name"), JsonType::String),
        ]
        .into_iter()
        .collect(),
        version: 1,
        last_seen_timestamp: 0,
    }
}

/// Benchmark schema store lookup (hot path).
fn bench_schema_store_lookup(c: &mut Criterion) {
    let store = SchemaStore::new();
    let route = make_test_route();

    // Register 100 schemas to simulate real-world usage
    for i in 0..100 {
        let route = RouteKey {
            method: HttpMethod::Get,
            path: Arc::from(format!("/api/v1/resource/{}", i)),
        };
        store.register(route, make_test_schema());
    }
    store.register(make_test_route(), make_test_schema());

    c.bench_function("schema_store_lookup", |b| {
        b.iter(|| {
            black_box(store.get(&route))
        })
    });
}

/// Benchmark schema fingerprint computation.
fn bench_schema_fingerprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("schema_fingerprint");

    group.throughput(Throughput::Bytes(SMALL_JSON.len() as u64));
    group.bench_function("small_payload", |b| {
        b.iter(|| {
            black_box(SchemaFingerprint::from_json_keys(SMALL_JSON))
        })
    });

    group.throughput(Throughput::Bytes(MEDIUM_JSON.len() as u64));
    group.bench_function("medium_payload", |b| {
        b.iter(|| {
            black_box(SchemaFingerprint::from_json_keys(MEDIUM_JSON))
        })
    });

    group.throughput(Throughput::Bytes(LARGE_JSON.len() as u64));
    group.bench_function("large_payload", |b| {
        b.iter(|| {
            black_box(SchemaFingerprint::from_json_keys(LARGE_JSON))
        })
    });

    group.finish();
}

/// Benchmark arena allocation and reset.
fn bench_arena_allocator(c: &mut Criterion) {
    let mut group = c.benchmark_group("arena_allocator");

    // Benchmark arena allocation
    group.bench_function("allocate_64kb", |b| {
        let mut arena = Bump::with_capacity(65536);
        b.iter(|| {
            arena.reset();
            let data: &mut [u8] = arena.alloc_slice_fill_default(65536);
            black_box(data.len())
        })
    });

    // Benchmark arena reset only
    group.bench_function("reset_only", |b| {
        let mut arena = Bump::with_capacity(65536);
        // Fill it first
        let _: &mut [u8] = arena.alloc_slice_fill_default(65536);

        b.iter(|| {
            arena.reset();
            black_box(arena.allocated_bytes())
        })
    });

    group.finish();
}

/// Benchmark pass-through middleware (baseline).
fn bench_passthrough_middleware(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let middleware = PassThroughMiddleware;
    let store = SchemaStore::new();
    let route = make_test_route();

    let ctx = MiddlewareContext {
        route: &route,
        schema_store: &store,
        request_start_ns: 0,
    };

    let mut group = c.benchmark_group("passthrough_middleware");

    group.throughput(Throughput::Bytes(SMALL_JSON.len() as u64));
    group.bench_function("small_payload", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(middleware.process(&ctx, SMALL_JSON).await)
            })
        })
    });

    group.throughput(Throughput::Bytes(MEDIUM_JSON.len() as u64));
    group.bench_function("medium_payload", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(middleware.process(&ctx, MEDIUM_JSON).await)
            })
        })
    });

    group.finish();
}

/// Benchmark fingerprint check middleware.
fn bench_fingerprint_middleware(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let middleware = FingerprintCheckMiddleware;
    let store = SchemaStore::new();
    let route = make_test_route();

    // Register schema for the route
    store.register(route.clone(), make_test_schema());

    let ctx = MiddlewareContext {
        route: &route,
        schema_store: &store,
        request_start_ns: 0,
    };

    let mut group = c.benchmark_group("fingerprint_middleware");

    group.throughput(Throughput::Bytes(MEDIUM_JSON.len() as u64));
    group.bench_function("with_schema", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(middleware.process(&ctx, MEDIUM_JSON).await)
            })
        })
    });

    // Without schema (early exit)
    let route_no_schema = RouteKey {
        method: HttpMethod::Get,
        path: Arc::from("/api/v1/unknown"),
    };
    let ctx_no_schema = MiddlewareContext {
        route: &route_no_schema,
        schema_store: &store,
        request_start_ns: 0,
    };

    group.bench_function("without_schema", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(middleware.process(&ctx_no_schema, MEDIUM_JSON).await)
            })
        })
    });

    group.finish();
}

/// Benchmark ArcSwap vs Mutex for schema store access patterns.
fn bench_arcswap_contention(c: &mut Criterion) {
    use std::sync::RwLock;

    let store_arcswap = Arc::new(SchemaStore::new());
    let store_rwlock = Arc::new(RwLock::new(std::collections::HashMap::<RouteKey, Arc<ExpectedSchema>>::new()));

    let route = make_test_route();
    let schema = Arc::new(make_test_schema());

    // Populate both stores
    store_arcswap.register(route.clone(), (*schema).clone());
    store_rwlock.write().unwrap().insert(route.clone(), schema);

    let mut group = c.benchmark_group("store_contention");

    group.bench_function("arcswap_read", |b| {
        b.iter(|| {
            black_box(store_arcswap.get(&route))
        })
    });

    group.bench_function("rwlock_read", |b| {
        b.iter(|| {
            let guard = store_rwlock.read().unwrap();
            black_box(guard.get(&route).cloned())
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_schema_store_lookup,
    bench_schema_fingerprint,
    bench_arena_allocator,
    bench_passthrough_middleware,
    bench_fingerprint_middleware,
    bench_arcswap_contention,
);

criterion_main!(benches);
