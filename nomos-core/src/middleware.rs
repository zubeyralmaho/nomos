//! Response middleware trait for the healing pipeline.
//!
//! The ResponseMiddleware trait defines the interface for response
//! transformation. Key design considerations:
//!
//! - **Streaming**: Receives body as bytes without forcing full allocation
//! - **Arena allocation**: Uses per-request bumpalo for scratch space
//! - **Zero-copy when possible**: Pass-through path avoids any transformation

use bytes::Bytes;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;
use crate::schema::{RouteKey, SchemaStore};

/// Context passed to middleware during request processing.
///
/// Contains all the shared state needed for healing decisions.
/// All fields are Send + Sync to allow async execution across threads.
///
/// Note: Arena allocation is handled via thread-local storage in the
/// middleware implementations, not passed through context. This ensures
/// the context remains Send + Sync.
pub struct MiddlewareContext<'a> {
    /// Route key for schema lookup
    pub route: &'a RouteKey,

    /// Reference to the global schema store
    pub schema_store: &'a SchemaStore,

    /// Request timestamp (for metrics)
    pub request_start_ns: u64,
}

/// Result of middleware processing.
///
/// Designed to avoid allocation in the pass-through case.
pub enum MiddlewareResult {
    /// Pass through unchanged (zero-copy)
    PassThrough,

    /// Response was transformed
    Transformed {
        /// The transformed body bytes
        body: Bytes,
        /// Healing operations applied (for X-Nomos-Healing-Ops header)
        ops_count: u32,
        /// Confidence score (for X-Nomos-Confidence header)
        confidence: f32,
    },

    /// Skip healing due to low confidence (pass through raw)
    LowConfidence {
        /// The confidence score that triggered skip
        confidence: f32,
    },
}

impl MiddlewareResult {
    /// Returns true if the response was healed.
    #[inline]
    pub fn was_healed(&self) -> bool {
        matches!(self, MiddlewareResult::Transformed { .. })
    }

    /// Get the healing operations count, or 0 if not healed.
    #[inline]
    pub fn ops_count(&self) -> u32 {
        match self {
            MiddlewareResult::Transformed { ops_count, .. } => *ops_count,
            _ => 0,
        }
    }
}

/// Response middleware trait for the healing pipeline.
///
/// Implementations of this trait handle the core healing logic:
/// 1. Schema fingerprint check
/// 2. Structural diff (if fingerprint mismatch)
/// 3. Semantic matching (if drift detected)
/// 4. JSON transformation (if healing needed)
///
/// # Design Principles
///
/// - **Non-blocking**: All operations must be async-safe
/// - **Arena allocation**: Use `ctx.arena` for scratch space, not heap
/// - **Fail-open**: On error, return PassThrough rather than failing the request
///
/// # Example
///
/// ```ignore
/// struct HealingMiddleware;
///
/// impl ResponseMiddleware for HealingMiddleware {
///     fn process<'a>(
///         &'a self,
///         ctx: &'a MiddlewareContext<'a>,
///         body: &'a [u8],
///     ) -> Pin<Box<dyn Future<Output = Result<MiddlewareResult>> + Send + 'a>> {
///         Box::pin(async move {
///             // Fast path: check fingerprint
///             // ... implementation ...
///             Ok(MiddlewareResult::PassThrough)
///         })
///     }
/// }
/// ```
pub trait ResponseMiddleware: Send + Sync {
    /// Process a response body.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Middleware context with schema store and arena
    /// * `body` - Response body bytes (borrowed slice, no allocation)
    ///
    /// # Returns
    ///
    /// * `Ok(MiddlewareResult)` - Processing result
    /// * `Err(_)` - Processing failed (caller should pass through raw)
    ///
    /// # Performance Requirements
    ///
    /// - **Pass-through path**: < 100ns
    /// - **Fingerprint check**: < 1µs
    /// - **Full healing**: < 100µs (excluding WASM)
    fn process<'a>(
        &'a self,
        ctx: &'a MiddlewareContext<'a>,
        body: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<MiddlewareResult>> + Send + 'a>>;
}

/// No-op middleware that always passes through.
///
/// Used as the default when no healing is configured.
/// This establishes the baseline latency.
pub struct PassThroughMiddleware;

impl ResponseMiddleware for PassThroughMiddleware {
    fn process<'a>(
        &'a self,
        _ctx: &'a MiddlewareContext<'a>,
        _body: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<MiddlewareResult>> + Send + 'a>> {
        Box::pin(async move { Ok(MiddlewareResult::PassThrough) })
    }
}

/// Middleware that performs fingerprint checking only.
///
/// This is useful for development/testing to measure the
/// overhead of schema validation without full healing.
pub struct FingerprintCheckMiddleware;

impl ResponseMiddleware for FingerprintCheckMiddleware {
    fn process<'a>(
        &'a self,
        ctx: &'a MiddlewareContext<'a>,
        body: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<MiddlewareResult>> + Send + 'a>> {
        Box::pin(async move {
            use crate::schema::SchemaFingerprint;

            // Get expected schema for this route
            let Some(expected) = ctx.schema_store.get(ctx.route) else {
                // No schema registered - pass through
                return Ok(MiddlewareResult::PassThrough);
            };

            // Compute actual fingerprint
            let actual_fp = SchemaFingerprint::from_json_keys(body)
                .unwrap_or(SchemaFingerprint(0));

            if actual_fp == expected.fingerprint {
                // Fingerprints match - pass through
                Ok(MiddlewareResult::PassThrough)
            } else {
                // Drift detected - for now, just pass through
                // Full healing implementation will go here
                tracing::debug!(
                    expected = expected.fingerprint.0,
                    actual = actual_fp.0,
                    "Schema drift detected"
                );
                Ok(MiddlewareResult::PassThrough)
            }
        })
    }
}

/// Middleware chain that runs multiple middlewares in sequence.
///
/// Stops at the first middleware that transforms the response.
pub struct MiddlewareChain {
    middlewares: Vec<Arc<dyn ResponseMiddleware>>,
}

impl MiddlewareChain {
    /// Create a new middleware chain.
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    /// Add a middleware to the chain.
    pub fn add(mut self, middleware: impl ResponseMiddleware + 'static) -> Self {
        self.middlewares.push(Arc::new(middleware));
        self
    }

    /// Get the number of middlewares in the chain.
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseMiddleware for MiddlewareChain {
    fn process<'a>(
        &'a self,
        ctx: &'a MiddlewareContext<'a>,
        body: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<MiddlewareResult>> + Send + 'a>> {
        Box::pin(async move {
            for middleware in &self.middlewares {
                let result = middleware.process(ctx, body).await?;
                if result.was_healed() {
                    return Ok(result);
                }
            }
            Ok(MiddlewareResult::PassThrough)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{HttpMethod, RouteKey};

    fn make_context<'a>(
        route: &'a RouteKey,
        schema_store: &'a SchemaStore,
    ) -> MiddlewareContext<'a> {
        MiddlewareContext {
            route,
            schema_store,
            request_start_ns: 0,
        }
    }

    #[tokio::test]
    async fn test_passthrough_middleware() {
        let route = RouteKey {
            method: HttpMethod::Get,
            path: Arc::from("/test"),
        };
        let store = SchemaStore::new();
        let ctx = make_context(&route, &store);

        let middleware = PassThroughMiddleware;
        let result = middleware.process(&ctx, b"{}").await.unwrap();

        assert!(matches!(result, MiddlewareResult::PassThrough));
    }

    #[tokio::test]
    async fn test_middleware_chain() {
        let route = RouteKey {
            method: HttpMethod::Get,
            path: Arc::from("/test"),
        };
        let store = SchemaStore::new();
        let ctx = make_context(&route, &store);

        let chain = MiddlewareChain::new()
            .add(PassThroughMiddleware)
            .add(FingerprintCheckMiddleware);

        assert_eq!(chain.len(), 2);

        let result = chain.process(&ctx, b"{}").await.unwrap();
        assert!(matches!(result, MiddlewareResult::PassThrough));
    }
}
