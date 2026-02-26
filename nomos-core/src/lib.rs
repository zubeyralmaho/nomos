//! Nomos Core - Zero-latency autonomous schema-healing proxy
//!
//! This crate implements the core proxy functionality as specified in
//! the Nomos architecture document. Key design principles:
//!
//! - **Zero-copy**: All JSON handling uses borrowed references where possible
//! - **Lock-free reads**: Schema Store uses ArcSwap for contention-free access
//! - **Arena allocation**: Per-request bumpalo allocators for transformation buffers
//! - **No .clone() on hot path**: Strict memory discipline
//! - **eBPF fast path**: Kernel-level packet classification via XDP

#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::unnecessary_to_owned)]
#![deny(clippy::clone_double_ref)]

pub mod control;
pub mod ebpf;
pub mod engine;
pub mod error;
pub mod middleware;
pub mod nlp;
pub mod proxy;
pub mod runtime;
pub mod schema;
pub mod wasm_host;

pub use control::{
    AggregatedMetrics, ControlConfig, ControlServer, ControlState, CoreMetrics,
    CoreSnapshot, ShardedMetrics,
};
pub use ebpf::{
    create_route_health_channel, spawn_feedback_loop, AggregatedStats, EbpfConfig, EbpfError,
    EbpfManager, RouteHealthUpdate, XdpMode, ROUTE_HEALTH_CHANNEL_CAPACITY,
};
pub use engine::{
    FieldEmbedding, HealingMap, HealingOp, LshIndex, MatchConfidence, SemanticHealer,
    simd_dot_product_i8, EMBEDDING_DIM, LSH_THRESHOLD,
};
pub use error::{NomosError, Result};
pub use middleware::{MiddlewareContext, ResponseMiddleware, WasmHealingMiddleware};
pub use proxy::{ProxyConfig, ProxyServer, ProxyState};
pub use runtime::build_runtime;
pub use schema::SchemaStore;
pub use wasm_host::{
    HealerPool, ModuleRegistry, PooledInstance, WasmHealer, WasmHealerError, 
    WasmHealerInstance, memory_layout,
};
