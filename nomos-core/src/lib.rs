//! Nomos Core - Zero-latency autonomous schema-healing proxy
//!
//! This crate implements the core proxy functionality as specified in
//! the Nomos architecture document. Key design principles:
//!
//! - **Zero-copy**: All JSON handling uses borrowed references where possible
//! - **Lock-free reads**: Schema Store uses ArcSwap for contention-free access
//! - **Arena allocation**: Per-request bumpalo allocators for transformation buffers
//! - **No .clone() on hot path**: Strict memory discipline

#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::unnecessary_to_owned)]
#![deny(clippy::clone_double_ref)]

pub mod error;
pub mod middleware;
pub mod proxy;
pub mod runtime;
pub mod schema;

pub use error::{NomosError, Result};
pub use middleware::{MiddlewareContext, ResponseMiddleware};
pub use proxy::{ProxyConfig, ProxyServer, ProxyState};
pub use runtime::build_runtime;
pub use schema::SchemaStore;
