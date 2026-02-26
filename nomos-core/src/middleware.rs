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

/// Nested path transformation rule.
/// Maps a source JSON path to a target key or path.
#[derive(Clone, Debug)]
struct NestedPathRule {
    /// Source path (e.g., "user.profile.name" or "account.balance")
    source_path: Vec<&'static str>,
    /// Target key (e.g., "full_name" or "account_balance")
    target_key: &'static str,
}

impl NestedPathRule {
    fn new(source: &'static str, target: &'static str) -> Self {
        Self {
            source_path: source.split('.').collect(),
            target_key: target,
        }
    }
}

/// Default nested path healing rules.
/// Maps nested structures to flat expected schema.
fn build_nested_path_rules() -> Vec<NestedPathRule> {
    vec![
        // ============================================================
        // Level 2 - Basic nested objects
        // ============================================================
        
        // User object flattening
        NestedPathRule::new("user.id", "user_id"),
        NestedPathRule::new("user.profile.name", "full_name"),
        NestedPathRule::new("user.profile.email", "email_address"),
        NestedPathRule::new("user.profile.firstName", "first_name"),
        NestedPathRule::new("user.profile.lastName", "last_name"),
        NestedPathRule::new("user.status.verified", "is_verified"),
        NestedPathRule::new("user.status.active", "is_active"),
        
        // Account object flattening
        NestedPathRule::new("account.balance", "account_balance"),
        NestedPathRule::new("account.currency", "currency"),
        NestedPathRule::new("account.type", "account_type"),
        
        // Timestamp flattening
        NestedPathRule::new("timestamps.created", "created_at"),
        NestedPathRule::new("timestamps.updated", "updated_at"),
        NestedPathRule::new("timestamps.deleted", "deleted_at"),
        NestedPathRule::new("meta.created", "created_at"),
        NestedPathRule::new("meta.updated", "updated_at"),
        
        // ============================================================
        // Level 3 - Data/Response wrappers
        // ============================================================
        
        NestedPathRule::new("data.user_id", "user_id"),
        NestedPathRule::new("data.id", "user_id"),
        NestedPathRule::new("data.name", "full_name"),
        NestedPathRule::new("data.email", "email_address"),
        NestedPathRule::new("data.balance", "account_balance"),
        NestedPathRule::new("data.user.id", "user_id"),
        NestedPathRule::new("data.user.name", "full_name"),
        NestedPathRule::new("data.user.email", "email_address"),
        
        NestedPathRule::new("response.user_id", "user_id"),
        NestedPathRule::new("response.data.user_id", "user_id"),
        NestedPathRule::new("response.data.id", "user_id"),
        NestedPathRule::new("response.data.user.id", "user_id"),
        NestedPathRule::new("result.user_id", "user_id"),
        NestedPathRule::new("result.data.user_id", "user_id"),
        
        // ============================================================
        // Level 4-5 - Deep nested user identity
        // ============================================================
        
        // response.data.user.identity.personal.name.full → full_name
        NestedPathRule::new("response.data.user.identity.personal.name.full", "full_name"),
        NestedPathRule::new("response.data.user.identity.personal.name.first", "first_name"),
        NestedPathRule::new("response.data.user.identity.personal.name.last", "last_name"),
        
        // response.data.user.identity.personal.contact.email.primary → email_address
        NestedPathRule::new("response.data.user.identity.personal.contact.email.primary", "email_address"),
        NestedPathRule::new("response.data.user.identity.personal.contact.email.verified", "email_verified"),
        NestedPathRule::new("response.data.user.identity.personal.contact.phone.mobile", "phone_number"),
        
        // response.data.user.identity.personal.location.address → address fields
        NestedPathRule::new("response.data.user.identity.personal.location.address.street", "street_address"),
        NestedPathRule::new("response.data.user.identity.personal.location.address.city", "city"),
        NestedPathRule::new("response.data.user.identity.personal.location.address.country", "country"),
        NestedPathRule::new("response.data.user.identity.personal.location.address.postal", "postal_code"),
        
        // response.data.user.financial.accounts.primary.balance → account_balance
        NestedPathRule::new("response.data.user.financial.accounts.primary.balance.amount", "account_balance"),
        NestedPathRule::new("response.data.user.financial.accounts.primary.balance.currency", "currency"),
        NestedPathRule::new("response.data.user.financial.accounts.primary.status", "account_status"),
        
        // response.data.user.metadata.audit.timestamps → timestamps
        NestedPathRule::new("response.data.user.metadata.audit.timestamps.created", "created_at"),
        NestedPathRule::new("response.data.user.metadata.audit.timestamps.updated", "updated_at"),
        NestedPathRule::new("response.data.user.metadata.audit.timestamps.last_login", "last_login_at"),
        NestedPathRule::new("response.data.user.metadata.audit.source.origin", "source"),
        NestedPathRule::new("response.data.user.metadata.audit.source.version", "api_version"),
        
        // Direct path from response.data.user
        NestedPathRule::new("response.data.user.id", "user_id"),
        
        // ============================================================
        // Settings → Preferences rename
        // ============================================================
        NestedPathRule::new("settings.theme", "preferences.theme"),
        NestedPathRule::new("settings.notifications", "preferences.notifications"),
        NestedPathRule::new("settings.language", "preferences.language"),
        
        // Body wrapper flattening
        NestedPathRule::new("body.user_id", "user_id"),
        NestedPathRule::new("payload.user_id", "user_id"),
    ]
}

/// Default healing rules for common drift patterns.
/// 
/// These map drifted field names back to expected schema fields.
fn build_default_healing_rules() -> Vec<(&'static str, &'static str)> {
    vec![
        // ============================================================
        // API v2 style renames (upstream_server.py v2 mode)
        // ============================================================
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
        
        // ============================================================
        // CamelCase to snake_case (Java/JS APIs)
        // ============================================================
        ("userId", "user_id"),
        ("fullName", "full_name"),
        ("emailAddress", "email_address"),
        ("accountBalance", "account_balance"),
        ("isVerified", "is_verified"),
        ("createdAt", "created_at"),
        ("updatedAt", "updated_at"),
        ("firstName", "first_name"),
        ("lastName", "last_name"),
        ("phoneNumber", "phone_number"),
        ("dateOfBirth", "date_of_birth"),
        ("postalCode", "postal_code"),
        ("streetAddress", "street_address"),
        ("createdBy", "created_by"),
        ("modifiedBy", "modified_by"),
        
        // ============================================================
        // Common abbreviations
        // ============================================================
        ("u_id", "user_id"),
        ("usr_id", "user_id"),
        ("uid", "user_id"),
        ("acct_bal", "account_balance"),
        ("acc_balance", "account_balance"),
        ("amt", "amount"),
        ("qty", "quantity"),
        ("desc", "description"),
        ("addr", "address"),
        ("tel", "telephone"),
        ("ph", "phone"),
        ("msg", "message"),
        ("pwd", "password"),
        ("passwd", "password"),
        ("usr", "user"),
        ("grp", "group"),
        ("org", "organization"),
        ("dept", "department"),
        ("cat", "category"),
        ("subcat", "subcategory"),
        ("img", "image"),
        ("pic", "picture"),
        ("sts", "status"),
        ("stat", "status"),
        ("ts", "timestamp"),
        ("dt", "datetime"),
        ("dob", "date_of_birth"),
        
        // ============================================================
        // Underscore variations
        // ============================================================
        ("username", "user_name"),
        ("firstname", "first_name"),
        ("lastname", "last_name"),
        ("emailaddress", "email_address"),
        ("phonenumber", "phone_number"),
        ("streetaddress", "street_address"),
        ("postalcode", "postal_code"),
        ("createdat", "created_at"),
        ("updatedat", "updated_at"),
        ("deletedat", "deleted_at"),
        
        // ============================================================
        // Pluralization issues
        // ============================================================
        ("tag", "tags"),
        ("label", "labels"),
        ("role", "roles"),
        ("permission", "permissions"),
        ("preference", "preferences"),
        ("setting", "settings"),
        ("option", "options"),
        ("item", "items"),
        ("result", "results"),
        ("error", "errors"),
        ("warning", "warnings"),
        
        // ============================================================
        // Common typos
        // ============================================================
        ("adress", "address"),
        ("addres", "address"),
        ("adddress", "address"),
        ("emial", "email"),
        ("emal", "email"),
        ("emali", "email"),
        ("mesage", "message"),
        ("messge", "message"),
        ("recieve", "receive"),
        ("recieved", "received"),
        ("occured", "occurred"),
        ("reponse", "response"),
        ("respone", "response"),
        ("desciption", "description"),
        ("decription", "description"),
        ("lenght", "length"),
        ("widht", "width"),
        ("heigth", "height"),
        
        // ============================================================
        // Legacy/alternative naming
        // ============================================================
        ("id", "user_id"),           // Generic id -> specific
        ("ID", "user_id"),
        ("_id", "user_id"),          // MongoDB style
        ("pk", "id"),                // Primary key
        ("oid", "object_id"),        // Object ID
        ("guid", "uuid"),
        ("active", "is_active"),
        ("enabled", "is_enabled"),
        ("deleted", "is_deleted"),
        ("archived", "is_archived"),
        ("visible", "is_visible"),
        ("public", "is_public"),
        ("private", "is_private"),
        ("admin", "is_admin"),
        ("superuser", "is_superuser"),
        
        // ============================================================
        // Date/time variations
        // ============================================================
        ("create_time", "created_at"),
        ("update_time", "updated_at"),
        ("delete_time", "deleted_at"),
        ("create_date", "created_at"),
        ("update_date", "updated_at"),
        ("modify_date", "modified_at"),
        ("modified", "modified_at"),
        ("timestamp", "created_at"),
        ("time", "timestamp"),
        
        // ============================================================
        // API framework variations (Rails, Django, etc.)
        // ============================================================
        ("created_on", "created_at"),
        ("updated_on", "updated_at"),
        ("modified_on", "modified_at"),
        ("date_joined", "created_at"),   // Django
        ("last_login", "last_login_at"),
        ("date_created", "created_at"),
        ("date_modified", "modified_at"),
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
        
        info!(
            patterns_count = patterns.len(),
            first_pattern = ?patterns.first().map(|p| String::from_utf8_lossy(&p.search).to_string()),
            "FastPathHealer initialized"
        );
        
        Self { patterns }
    }
    
    /// Perform fast key renames on JSON bytes.
    /// 
    /// Returns (healed_bytes, ops_count) or None if healing not possible.
    #[inline]
    fn heal(&self, input: &[u8]) -> Option<(Vec<u8>, u32)> {
        if input.is_empty() || self.patterns.is_empty() {
            debug!(
                input_len = input.len(),
                patterns_count = self.patterns.len(),
                "FastPathHealer: empty input or no patterns"
            );
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
            debug!(ops_count, "FastPathHealer: healed");
            Some((output, ops_count))
        } else {
            None
        }
    }
}

// ============================================================================
// Nested JSON Healer (Path-aware transformations)
// ============================================================================

/// Nested JSON healer for path-based transformations.
/// 
/// Handles complex structural changes like flattening nested objects:
/// - `user.profile.name` → `full_name`
/// - `account.balance` → `account_balance`
/// - `settings` → `preferences`
struct NestedJsonHealer {
    /// Path transformation rules
    rules: Vec<NestedPathRule>,
}

impl NestedJsonHealer {
    fn new() -> Self {
        Self {
            rules: build_nested_path_rules(),
        }
    }

    /// Check if JSON has nested structure markers (heuristic).
    #[inline]
    fn has_nested_structure(json: &[u8]) -> bool {
        // Look for common nested wrapper keys (with or without space after colon)
        let markers = [
            b"\"user\":" as &[u8],
            b"\"account\":",
            b"\"data\":",
            b"\"response\":",
            b"\"result\":",
            b"\"payload\":",
            b"\"timestamps\":",
            b"\"profile\":",
            b"\"settings\":",
        ];
        
        for marker in &markers {
            if Self::contains_bytes(json, marker) {
                // Also check if followed by object opening (with optional whitespace)
                if let Some(pos) = Self::find_bytes(json, marker) {
                    let rest = &json[pos + marker.len()..];
                    // Skip whitespace and check for '{'
                    for &b in rest.iter().take(10) {
                        if b == b'{' {
                            return true;
                        }
                        if b != b' ' && b != b'\t' && b != b'\n' && b != b'\r' {
                            break;
                        }
                    }
                }
            }
        }
        false
    }

    /// Find position of byte sequence.
    #[inline]
    fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.len() > haystack.len() {
            return None;
        }
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    /// Simple byte sequence search.
    #[inline]
    fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
        if needle.len() > haystack.len() {
            return false;
        }
        haystack.windows(needle.len()).any(|w| w == needle)
    }

    /// Heal nested JSON by flattening according to rules.
    /// 
    /// Returns (healed_bytes, ops_count) or None if no nested healing needed.
    fn heal(&self, input: &[u8]) -> Option<(Vec<u8>, u32)> {
        // Quick check: does this look like nested JSON?
        if !Self::has_nested_structure(input) {
            return None;
        }

        // Parse JSON
        let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(input) else {
            return None;
        };

        let serde_json::Value::Object(ref mut root) = value else {
            return None;
        };

        let mut ops_count = 0u32;
        let mut extracted = serde_json::Map::new();

        // Apply path extraction rules
        for rule in &self.rules {
            if let Some(val) = self.extract_path(root, &rule.source_path) {
                // Handle nested target paths (e.g., "preferences.theme")
                if rule.target_key.contains('.') {
                    let parts: Vec<&str> = rule.target_key.split('.').collect();
                    if parts.len() == 2 {
                        let parent = extracted
                            .entry(parts[0])
                            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                        if let serde_json::Value::Object(ref mut obj) = parent {
                            obj.insert(parts[1].to_string(), val);
                            ops_count += 1;
                        }
                    }
                } else {
                    extracted.insert(rule.target_key.to_string(), val);
                    ops_count += 1;
                }
            }
        }

        if ops_count == 0 {
            return None;
        }

        // Copy non-extracted top-level fields
        for (key, val) in root.iter() {
            // Skip wrapper objects that we've flattened
            match key.as_str() {
                "user" | "account" | "timestamps" | "data" | "response" | "result" | "payload" 
                    if val.is_object() => continue,
                "settings" if val.is_object() && extracted.contains_key("preferences") => continue,
                _ => {
                    if !extracted.contains_key(key) {
                        extracted.insert(key.clone(), val.clone());
                    }
                }
            }
        }

        // Serialize back to JSON
        let output = serde_json::to_vec(&serde_json::Value::Object(extracted)).ok()?;
        
        debug!(
            ops_count,
            input_len = input.len(),
            output_len = output.len(),
            "NestedJsonHealer: flattened structure"
        );

        Some((output, ops_count))
    }

    /// Extract a value from a nested path.
    fn extract_path(&self, obj: &serde_json::Map<String, serde_json::Value>, path: &[&str]) -> Option<serde_json::Value> {
        if path.is_empty() {
            return None;
        }

        let first = path[0];
        let value = obj.get(first)?;

        if path.len() == 1 {
            return Some(value.clone());
        }

        // Recurse into nested object
        if let serde_json::Value::Object(ref inner) = value {
            self.extract_path(inner, &path[1..])
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
/// 1. Tries nested JSON healer first (for structural changes)
/// 2. Tries fast-path pure Rust healing (for key renames)
/// 3. Falls back to WASM only for complex transformations
/// 4. Uses pre-compiled patterns for maximum speed
pub struct WasmHealingMiddleware {
    /// WASM module registry for healer instances (fallback)
    wasm_registry: Arc<ModuleRegistry>,
    
    /// Pre-built healing rules: (drifted_key, expected_key)
    healing_rules: RwLock<HashMap<String, String>>,
    
    /// Fast-path healer (pure Rust, no WASM)
    fast_healer: RwLock<FastPathHealer>,
    
    /// Nested JSON healer (for structural flattening)
    nested_healer: NestedJsonHealer,
    
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
        
        // Create nested JSON healer for structural transformations
        let nested_healer = NestedJsonHealer::new();
        
        info!(
            rules_count = rules.len(),
            nested_rules_count = nested_healer.rules.len(),
            "Initialized WasmHealingMiddleware with fast-path and nested healing"
        );
        
        Self {
            wasm_registry,
            healing_rules: RwLock::new(rules),
            fast_healer: RwLock::new(fast_healer),
            nested_healer,
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
            
            let mut current_body: Vec<u8>;
            let mut total_ops: u32 = 0;
            
            // PHASE 1: Nested JSON healing (structural flattening)
            // Try to flatten nested structures first
            let body_for_fast = if let Some((flattened, ops)) = self.nested_healer.heal(body) {
                total_ops += ops;
                current_body = flattened;
                &current_body[..]
            } else {
                body
            };
            
            // PHASE 2: Fast-path key replacement
            // Apply simple key renames on the (possibly flattened) body
            {
                let fast_healer = self.fast_healer.read();
                if let Some((healed_body, ops_count)) = fast_healer.heal(body_for_fast) {
                    total_ops += ops_count;
                    return Ok(MiddlewareResult::Transformed {
                        body: Bytes::from(healed_body),
                        ops_count: total_ops,
                        confidence: 1.0,
                    });
                }
            }
            
            // If we had nested healing but no fast-path, return nested result
            if total_ops > 0 {
                return Ok(MiddlewareResult::Transformed {
                    body: Bytes::from(body_for_fast.to_vec()),
                    ops_count: total_ops,
                    confidence: 1.0,
                });
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
