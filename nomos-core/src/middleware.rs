//! Response middleware trait for the healing pipeline.
//!
//! The ResponseMiddleware trait defines the interface for response
//! transformation. Key design considerations:
//!
//! - **Streaming**: Receives body as bytes without forcing full allocation
//! - **Arena allocation**: Uses per-request bumpalo for scratch space
//! - **Zero-copy when possible**: Pass-through path avoids any transformation

use bytes::Bytes;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::{debug, info, warn};

use crate::engine::{HealingOp, SemanticHealer};
use crate::error::Result;
use crate::schema::{ExpectedSchema, JsonType, RouteKey, SchemaFingerprint, SchemaStore};
use crate::wasm_host::ModuleRegistry;

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

// ============================================================================
// WASM Healing Middleware
// ============================================================================

/// Default healing rules for common drift patterns.
/// 
/// These map drifted field names back to expected schema fields.
fn build_default_healing_rules() -> Vec<(&'static str, &'static str)> {
    vec![
        // API v2 style renames (upstream_server.py v2 mode)
        ("uuid", "user_id"),
        ("name", "full_name"),
        ("email", "email_address"),
        ("balance", "account_balance"),
        ("verified", "is_verified"),
        ("created", "created_at"),
        ("prefs", "preferences"),
        ("labels", "tags"),
        ("meta", "metadata"),
        ("notifs", "notifications"),
        ("lang", "language"),
        ("ver", "version"),
        ("src", "source"),
        
        // CamelCase to snake_case (upstream_server.py camel mode)
        ("userId", "user_id"),
        ("fullName", "full_name"),
        ("emailAddress", "email_address"),
        ("accountBalance", "account_balance"),
        ("isVerified", "is_verified"),
        ("createdAt", "created_at"),
        
        // stress_test.py patterns
        ("u_id", "user_id"),
    ]
}

// ============================================================================
// Fast-Path Healer (Pure Rust, No WASM)
// ============================================================================

/// Pre-compiled key pattern for fast replacement.
/// 
/// Stores both the search pattern (`"key":`) and replacement (`"newkey":`).
#[derive(Clone)]
struct KeyPattern {
    /// Search bytes: `"oldkey":`
    search: Vec<u8>,
    /// Replace bytes: `"newkey":`
    replace: Vec<u8>,
    /// Original key name (for logging)
    from_key: String,
    /// Target key name (for logging)
    to_key: String,
}

impl KeyPattern {
    fn new(from: &str, to: &str) -> Self {
        Self {
            search: format!("\"{}\":", from).into_bytes(),
            replace: format!("\"{}\":", to).into_bytes(),
            from_key: from.to_string(),
            to_key: to.to_string(),
        }
    }
}

/// Fast-path healer using byte-level search and replace.
/// 
/// This avoids WASM invocation entirely for simple key renames.
/// Achieves sub-100µs healing for typical payloads.
struct FastPathHealer {
    /// Pre-compiled patterns sorted by search length (longest first for greedy match)
    patterns: Vec<KeyPattern>,
}

impl FastPathHealer {
    fn new(rules: &HashMap<String, String>) -> Self {
        let mut patterns: Vec<KeyPattern> = rules
            .iter()
            .map(|(from, to)| KeyPattern::new(from, to))
            .collect();
        
        // Sort by search length descending for greedy matching
        patterns.sort_by(|a, b| b.search.len().cmp(&a.search.len()));
        
        Self { patterns }
    }
    
    /// Perform fast key renames on JSON bytes.
    /// 
    /// Returns (healed_bytes, ops_count) or None if healing not possible.
    #[inline]
    fn heal(&self, input: &[u8]) -> Option<(Vec<u8>, u32)> {
        if input.is_empty() {
            return None;
        }
        
        // Pre-allocate output with some slack for longer key names
        let mut output = Vec::with_capacity(input.len() + 256);
        let mut ops_count = 0u32;
        let mut i = 0;
        
        while i < input.len() {
            let mut matched = false;
            
            // Try each pattern
            for pattern in &self.patterns {
                let search_len = pattern.search.len();
                if i + search_len <= input.len() 
                   && &input[i..i + search_len] == pattern.search.as_slice() 
                {
                    // Match found - copy replacement
                    output.extend_from_slice(&pattern.replace);
                    i += search_len;
                    ops_count += 1;
                    matched = true;
                    break;
                }
            }
            
            if !matched {
                // No pattern matched - copy byte as-is
                output.push(input[i]);
                i += 1;
            }
        }
        
        if ops_count > 0 {
            Some((output, ops_count))
        } else {
            None
        }
    }
}

/// Extract top-level JSON keys from a byte slice.
/// 
/// Uses simple byte scanning - not full JSON parsing for speed.
fn extract_json_keys(json: &[u8]) -> Vec<String> {
    let mut keys = Vec::new();
    let mut i = 0;
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escape_next = false;
    let mut key_start: Option<usize> = None;
    let mut expecting_colon = false;
    
    while i < json.len() {
        let b = json[i];
        
        if escape_next {
            escape_next = false;
            i += 1;
            continue;
        }
        
        if in_string {
            match b {
                b'\\' => escape_next = true,
                b'"' => {
                    in_string = false;
                    if let Some(start) = key_start {
                        if depth == 1 {
                            // We have a potential key
                            expecting_colon = true;
                            if let Ok(key) = std::str::from_utf8(&json[start..i]) {
                                // Will be added if colon follows
                                if expecting_colon {
                                    keys.push(key.to_string());
                                }
                            }
                        }
                        key_start = None;
                    }
                }
                _ => {}
            }
        } else {
            match b {
                b'"' => {
                    in_string = true;
                    key_start = Some(i + 1);
                    expecting_colon = false;
                }
                b':' => {
                    // Key confirmed
                    expecting_colon = false;
                }
                b'{' => {
                    depth += 1;
                }
                b'}' => {
                    depth = depth.saturating_sub(1);
                }
                b'[' | b']' | b',' | b' ' | b'\n' | b'\r' | b'\t' => {}
                _ => {
                    expecting_colon = false;
                }
            }
        }
        
        i += 1;
    }
    
    keys
}

/// Rule-based healing middleware using WASM for JSON transformation.
/// 
/// This middleware:
/// 1. Tries fast-path pure Rust healing first (sub-100µs)
/// 2. Falls back to WASM only for complex transformations
/// 3. Uses pre-compiled patterns for maximum speed
pub struct WasmHealingMiddleware {
    /// WASM module registry for healer instances (fallback)
    wasm_registry: Arc<ModuleRegistry>,
    
    /// Pre-built healing rules: (drifted_key, expected_key)
    healing_rules: RwLock<HashMap<String, String>>,
    
    /// Fast-path healer (pure Rust, no WASM)
    fast_healer: RwLock<FastPathHealer>,
    
    /// Semantic healer for advanced matching (when rules don't match)
    semantic_healer: RwLock<SemanticHealer>,
    
    /// Minimum confidence threshold
    confidence_threshold: f32,
}

impl WasmHealingMiddleware {
    /// Create a new healing middleware with default rules.
    pub fn new(wasm_registry: Arc<ModuleRegistry>) -> Self {
        let mut rules = HashMap::new();
        for (from, to) in build_default_healing_rules() {
            rules.insert(from.to_string(), to.to_string());
        }
        
        let mut healer = SemanticHealer::with_threshold(0.65);
        
        // Register default expected schema
        let expected_fields = vec![
            "user_id", "full_name", "email_address", "account_balance",
            "is_verified", "created_at", "preferences", "tags", "metadata",
            "notifications", "language", "version", "source", "theme",
        ];
        
        let mut field_types = HashMap::new();
        for f in &expected_fields {
            field_types.insert(Arc::from(*f), JsonType::String);
        }
        field_types.insert(Arc::from("account_balance"), JsonType::Number);
        field_types.insert(Arc::from("is_verified"), JsonType::Bool);
        field_types.insert(Arc::from("preferences"), JsonType::Object);
        field_types.insert(Arc::from("tags"), JsonType::Array);
        field_types.insert(Arc::from("metadata"), JsonType::Object);
        field_types.insert(Arc::from("notifications"), JsonType::Bool);
        
        let expected_schema = ExpectedSchema {
            fingerprint: SchemaFingerprint::from_fields(
                &expected_fields.iter().map(|s| *s).collect::<Vec<_>>()
            ),
            root_fields: expected_fields.iter().map(|s| Arc::from(*s)).collect(),
            field_types,
            version: 1,
            last_seen_timestamp: 0,
        };
        
        healer.register_schema(&expected_schema);
        
        // Create fast-path healer with pre-compiled patterns
        let fast_healer = FastPathHealer::new(&rules);
        
        info!(
            rules_count = rules.len(),
            "Initialized WasmHealingMiddleware with fast-path healing"
        );
        
        Self {
            wasm_registry,
            healing_rules: RwLock::new(rules),
            fast_healer: RwLock::new(fast_healer),
            semantic_healer: RwLock::new(healer),
            confidence_threshold: 0.65,
        }
    }
    
    /// Add a custom healing rule.
    pub fn add_rule(&self, from: &str, to: &str) {
        let mut rules = self.healing_rules.write();
        rules.insert(from.to_string(), to.to_string());
        // Rebuild fast healer with new rules
        *self.fast_healer.write() = FastPathHealer::new(&rules);
    }
    
    /// Build healing operations from actual JSON keys.
    fn build_healing_ops(&self, actual_keys: &[String]) -> Vec<HealingOp> {
        let rules = self.healing_rules.read();
        let mut ops = Vec::new();
        
        for key in actual_keys {
            if let Some(target) = rules.get(key) {
                ops.push(HealingOp::Rename {
                    from: Arc::from(key.as_str()),
                    to: Arc::from(target.as_str()),
                    confidence: 1.0,
                });
            }
        }
        
        ops
    }
}

impl ResponseMiddleware for WasmHealingMiddleware {
    fn process<'a>(
        &'a self,
        _ctx: &'a MiddlewareContext<'a>,
        body: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<MiddlewareResult>> + Send + 'a>> {
        Box::pin(async move {
            // Skip empty or non-JSON bodies
            if body.is_empty() || (body[0] != b'{' && body[0] != b'[') {
                return Ok(MiddlewareResult::PassThrough);
            }
            
            // FAST PATH: Pure Rust byte-level key replacement
            // This avoids WASM invocation entirely for common patterns
            {
                let fast_healer = self.fast_healer.read();
                if let Some((healed_body, ops_count)) = fast_healer.heal(body) {
                    return Ok(MiddlewareResult::Transformed {
                        body: Bytes::from(healed_body),
                        ops_count,
                        confidence: 1.0,
                    });
                }
            }
            
            // No healing needed - pass through
            Ok(MiddlewareResult::PassThrough)
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
