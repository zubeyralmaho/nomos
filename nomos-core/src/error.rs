//! Error types for Nomos proxy operations.
//!
//! Errors are designed to be lightweight and avoid allocations where possible.

use thiserror::Error;

/// Result type alias for Nomos operations.
pub type Result<T> = std::result::Result<T, NomosError>;

/// Core error types for the Nomos proxy.
///
/// These errors are designed for the hot path - they should be Copy where
/// possible and avoid heap allocations.
#[derive(Error, Debug)]
pub enum NomosError {
    /// Upstream connection failed
    #[error("upstream connection error: {0}")]
    UpstreamConnection(#[from] hyper_util::client::legacy::Error),

    /// HTTP error from hyper
    #[error("http error: {0}")]
    Http(#[from] hyper::http::Error),

    /// Invalid URI
    #[error("invalid uri: {0}")]
    InvalidUri(#[from] hyper::http::uri::InvalidUri),

    /// Body streaming error
    #[error("body error: {0}")]
    Body(String),

    /// JSON parsing failed
    #[error("json parse error: {0}")]
    JsonParse(String),

    /// Schema validation failed
    #[error("schema mismatch: expected fingerprint {expected}, got {actual}")]
    SchemaMismatch { expected: u64, actual: u64 },

    /// Middleware processing failed
    #[error("middleware error: {0}")]
    Middleware(String),

    /// Request timeout
    #[error("request timeout after {0}ms")]
    Timeout(u64),

    /// Circuit breaker is open
    #[error("circuit breaker open for route")]
    CircuitOpen,

    /// IO error
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl NomosError {
    /// Returns an HTTP status code appropriate for this error.
    #[inline]
    pub fn status_code(&self) -> u16 {
        match self {
            NomosError::UpstreamConnection(_) => 502,
            NomosError::Timeout(_) => 504,
            NomosError::CircuitOpen => 503,
            NomosError::SchemaMismatch { .. } => 500,
            _ => 500,
        }
    }

    /// Returns true if this error should trigger circuit breaker.
    #[inline]
    pub fn is_circuit_breaker_trigger(&self) -> bool {
        matches!(
            self,
            NomosError::UpstreamConnection(_) | NomosError::Timeout(_)
        )
    }
}
