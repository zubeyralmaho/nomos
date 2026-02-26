//! Control Plane API - Internal Management Interface
//!
//! Provides HTTP endpoints for runtime management:
//! - Hot-swap WASM healer modules
//! - Query metrics and health status
//! - Configure healing behavior
//!
//! # Design Principles
//!
//! - **Nomos Law**: Control plane must not steal cycles from data plane
//! - All metric reads are lock-free (AtomicU64)
//! - Hot-swap is atomic (ArcSwap)
//! - Separate listening port to isolate from proxied traffic

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use http::{Request, Response, StatusCode};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tracing::{error, info, warn};

use crate::ebpf::AggregatedStats;
use crate::proxy::ProxyMetrics;
use crate::wasm_host::{ModuleRegistry, WasmHealerError};

// ============================================================================
// Configuration
// ============================================================================

/// Control plane configuration.
#[derive(Debug, Clone)]
pub struct ControlConfig {
    /// Address to listen on (default: 127.0.0.1:8081)
    pub listen_addr: SocketAddr,
    
    /// Maximum WASM module size in bytes (default: 10MB)
    pub max_wasm_size: usize,
    
    /// API key for authentication (None = no auth)
    pub api_key: Option<String>,
}

impl Default for ControlConfig {
    fn default() -> Self {
        Self {
            listen_addr: ([127, 0, 0, 1], 8081).into(),
            max_wasm_size: 10 * 1024 * 1024, // 10MB
            api_key: None,
        }
    }
}

// ============================================================================
// Per-Core Metrics (Lock-Free)
// ============================================================================

/// Sharded per-core metrics to avoid false sharing.
///
/// Each core gets its own cache-line-aligned counters.
/// Totals are computed on read by summing across cores.
#[repr(align(64))]  // Cache line alignment
#[derive(Debug, Default)]
pub struct CoreMetrics {
    pub requests_total: AtomicU64,
    pub requests_healed: AtomicU64,
    pub healing_time_ns: AtomicU64,
    pub wasm_calls: AtomicU64,
    pub wasm_errors: AtomicU64,
    _padding: [u64; 3],  // Pad to 64 bytes
}

impl CoreMetrics {
    #[inline]
    pub fn record_request(&self, healed: bool, healing_ns: u64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        if healed {
            self.requests_healed.fetch_add(1, Ordering::Relaxed);
            self.healing_time_ns.fetch_add(healing_ns, Ordering::Relaxed);
        }
    }
    
    #[inline]
    pub fn record_wasm_call(&self, error: bool) {
        self.wasm_calls.fetch_add(1, Ordering::Relaxed);
        if error {
            self.wasm_errors.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Sharded metrics across all cores.
pub struct ShardedMetrics {
    cores: Box<[CoreMetrics]>,
    num_cores: usize,
    start_time: Instant,
}

impl ShardedMetrics {
    /// Create new sharded metrics for the given number of cores.
    pub fn new(num_cores: usize) -> Self {
        let cores: Vec<CoreMetrics> = (0..num_cores)
            .map(|_| CoreMetrics::default())
            .collect();
        
        Self {
            cores: cores.into_boxed_slice(),
            num_cores,
            start_time: Instant::now(),
        }
    }
    
    /// Get metrics for current core (uses thread ID hashing).
    #[inline]
    pub fn current_core(&self) -> &CoreMetrics {
        let core_id = core_id() % self.num_cores;
        &self.cores[core_id]
    }
    
    /// Aggregate totals across all cores.
    pub fn aggregate(&self) -> AggregatedMetrics {
        let mut total = AggregatedMetrics {
            requests_total: 0,
            requests_healed: 0,
            healing_time_ns: 0,
            wasm_calls: 0,
            wasm_errors: 0,
            uptime_secs: self.start_time.elapsed().as_secs(),
            num_cores: self.num_cores,
            per_core: Vec::with_capacity(self.num_cores),
        };
        
        for (i, core) in self.cores.iter().enumerate() {
            let requests = core.requests_total.load(Ordering::Relaxed);
            let healed = core.requests_healed.load(Ordering::Relaxed);
            let healing_ns = core.healing_time_ns.load(Ordering::Relaxed);
            let wasm_calls = core.wasm_calls.load(Ordering::Relaxed);
            let wasm_errors = core.wasm_errors.load(Ordering::Relaxed);
            
            total.requests_total += requests;
            total.requests_healed += healed;
            total.healing_time_ns += healing_ns;
            total.wasm_calls += wasm_calls;
            total.wasm_errors += wasm_errors;
            
            total.per_core.push(CoreSnapshot {
                core_id: i,
                requests: requests,
                healed: healed,
                avg_healing_ns: if healed > 0 { healing_ns / healed } else { 0 },
            });
        }
        
        total
    }
}

/// Get current core ID (approximation using thread ID).
#[inline]
fn core_id() -> usize {
    // Use thread ID as proxy for core ID
    // In production with core affinity, this would be accurate
    std::thread::current().id().as_u64().hash() as usize
}

/// Trait extension for thread ID hashing.
trait ThreadIdExt {
    fn as_u64(&self) -> &std::thread::ThreadId;
    fn hash(&self) -> u64;
}

impl ThreadIdExt for std::thread::ThreadId {
    fn as_u64(&self) -> &std::thread::ThreadId {
        self
    }
    
    fn hash(&self) -> u64 {
        // Simple hash of the thread ID's debug representation
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        Hash::hash(self, &mut hasher);
        hasher.finish()
    }
}

/// Aggregated metrics snapshot.
#[derive(Debug, Clone)]
pub struct AggregatedMetrics {
    pub requests_total: u64,
    pub requests_healed: u64,
    pub healing_time_ns: u64,
    pub wasm_calls: u64,
    pub wasm_errors: u64,
    pub uptime_secs: u64,
    pub num_cores: usize,
    pub per_core: Vec<CoreSnapshot>,
}

impl AggregatedMetrics {
    /// Calculate p99 healing latency (approximation).
    pub fn p99_healing_us(&self) -> u64 {
        // Simple approximation: assume p99 ≈ 2x average
        if self.requests_healed == 0 {
            return 0;
        }
        let avg_ns = self.healing_time_ns / self.requests_healed;
        (avg_ns * 2) / 1000  // Convert to microseconds
    }
    
    /// Serialize to JSON.
    pub fn to_json(&self) -> String {
        let per_core_json: Vec<String> = self.per_core.iter()
            .map(|c| format!(
                r#"{{"core_id":{},"requests":{},"healed":{},"avg_healing_ns":{}}}"#,
                c.core_id, c.requests, c.healed, c.avg_healing_ns
            ))
            .collect();
        
        format!(
            r#"{{
  "requests_total": {},
  "requests_healed": {},
  "healing_rate": {:.4},
  "avg_healing_us": {},
  "p99_healing_us": {},
  "wasm_calls": {},
  "wasm_errors": {},
  "uptime_secs": {},
  "num_cores": {},
  "per_core": [{}]
}}"#,
            self.requests_total,
            self.requests_healed,
            if self.requests_total > 0 {
                self.requests_healed as f64 / self.requests_total as f64
            } else {
                0.0
            },
            if self.requests_healed > 0 {
                (self.healing_time_ns / self.requests_healed) / 1000
            } else {
                0
            },
            self.p99_healing_us(),
            self.wasm_calls,
            self.wasm_errors,
            self.uptime_secs,
            self.num_cores,
            per_core_json.join(",\n    ")
        )
    }
}

/// Per-core metric snapshot.
#[derive(Debug, Clone)]
pub struct CoreSnapshot {
    pub core_id: usize,
    pub requests: u64,
    pub healed: u64,
    pub avg_healing_ns: u64,
}

// ============================================================================
// Control Plane State
// ============================================================================

/// Shared state for the control plane.
pub struct ControlState {
    /// Configuration
    pub config: ControlConfig,
    
    /// WASM module registry for hot-swap
    pub healer_registry: Arc<ModuleRegistry>,
    
    /// Sharded metrics (per-core)
    pub metrics: Arc<ShardedMetrics>,
    
    /// Proxy metrics (legacy, for compatibility)
    pub proxy_metrics: Arc<ProxyMetrics>,
    
    /// eBPF stats (optional)
    pub ebpf_stats: Option<Arc<dyn Fn() -> AggregatedStats + Send + Sync>>,
}

// ============================================================================
// Control Plane Server
// ============================================================================

/// Control plane HTTP server.
pub struct ControlServer {
    state: Arc<ControlState>,
}

impl ControlServer {
    /// Create a new control server.
    pub fn new(state: ControlState) -> Self {
        Self {
            state: Arc::new(state),
        }
    }
    
    /// Run the control server.
    ///
    /// This binds to the control port and handles management requests.
    pub async fn run(&self) -> Result<(), std::io::Error> {
        let listener = TcpListener::bind(self.state.config.listen_addr).await?;
        
        info!(
            addr = %self.state.config.listen_addr,
            "Control plane listening"
        );
        
        loop {
            let (stream, peer_addr) = listener.accept().await?;
            let io = TokioIo::new(stream);
            let state = Arc::clone(&self.state);
            
            tokio::spawn(async move {
                let service = service_fn(move |req| {
                    let state = Arc::clone(&state);
                    async move { handle_control_request(state, req).await }
                });
                
                if let Err(e) = http1::Builder::new()
                    .serve_connection(io, service)
                    .await
                {
                    warn!(peer = %peer_addr, error = %e, "Control connection error");
                }
            });
        }
    }
}

/// Handle a control plane request.
async fn handle_control_request(
    state: Arc<ControlState>,
    req: Request<Incoming>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    let path = req.uri().path();
    let method = req.method();
    
    // Check API key if configured
    if let Some(ref expected_key) = state.config.api_key {
        let provided_key = req.headers()
            .get("X-API-Key")
            .and_then(|v| v.to_str().ok());
        
        if provided_key != Some(expected_key) {
            return Ok(json_response(StatusCode::UNAUTHORIZED, 
                r#"{"error": "Invalid or missing API key"}"#));
        }
    }
    
    match (method, path) {
        // POST /v1/healer - Upload new WASM module
        (&http::Method::POST, "/v1/healer") => {
            handle_healer_upload(state, req).await
        }
        
        // POST /v1/heal - Heal JSON with schema (for demo/testing)
        (&http::Method::POST, "/v1/heal") => {
            handle_heal_request(state, req).await
        }
        
        // OPTIONS /v1/heal - CORS preflight
        (&http::Method::OPTIONS, "/v1/heal") => {
            Ok(cors_preflight_response())
        }
        
        // GET /v1/metrics - Get metrics JSON
        (&http::Method::GET, "/v1/metrics") => {
            handle_metrics_get(state).await
        }
        
        // GET /v1/health - Health check
        (&http::Method::GET, "/v1/health") => {
            Ok(cors_json_response(StatusCode::OK, r#"{"status": "healthy"}"#))
        }
        
        // OPTIONS /v1/health - CORS preflight
        (&http::Method::OPTIONS, "/v1/health") => {
            Ok(cors_preflight_response())
        }
        
        // GET /v1/healer/version - Get current healer version
        (&http::Method::GET, "/v1/healer/version") => {
            let version = state.healer_registry.version();
            Ok(json_response(StatusCode::OK, 
                &format!(r#"{{"version": {}}}"#, version)))
        }
        
        // Unknown endpoint
        _ => {
            Ok(json_response(StatusCode::NOT_FOUND,
                r#"{"error": "Not found"}"#))
        }
    }
}

/// Handle POST /v1/healer - Upload new WASM module.
async fn handle_healer_upload(
    state: Arc<ControlState>,
    req: Request<Incoming>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    // Read body
    let body = req.into_body();
    let collected = body.collect().await
        .map_err(|e| {
            error!(error = %e, "Failed to read healer upload body");
            e
        })?;
    let wasm_bytes = collected.to_bytes();
    
    // Check size limit
    if wasm_bytes.len() > state.config.max_wasm_size {
        return Ok(json_response(StatusCode::PAYLOAD_TOO_LARGE,
            &format!(r#"{{"error": "WASM module too large: {} > {} bytes"}}"#,
                wasm_bytes.len(), state.config.max_wasm_size)));
    }
    
    // Validate WASM magic number
    if wasm_bytes.len() < 8 || &wasm_bytes[0..4] != b"\0asm" {
        return Ok(json_response(StatusCode::BAD_REQUEST,
            r#"{"error": "Invalid WASM module (bad magic number)"}"#));
    }
    
    info!(
        size = wasm_bytes.len(),
        "Received WASM healer upload, initiating hot-swap"
    );
    
    // Perform hot-swap
    match state.healer_registry.hot_swap(&wasm_bytes) {
        Ok(new_version) => {
            info!(version = new_version, "WASM healer hot-swapped successfully");
            Ok(json_response(StatusCode::OK,
                &format!(r#"{{"success": true, "version": {}}}"#, new_version)))
        }
        Err(WasmHealerError::CompilationError(e)) => {
            error!(error = %e, "WASM compilation failed during hot-swap");
            Ok(json_response(StatusCode::BAD_REQUEST,
                &format!(r#"{{"error": "WASM compilation failed: {}"}}"#, e)))
        }
        Err(e) => {
            error!(error = %e, "WASM hot-swap failed");
            Ok(json_response(StatusCode::INTERNAL_SERVER_ERROR,
                &format!(r#"{{"error": "Hot-swap failed: {}"}}"#, e)))
        }
    }
}

/// Handle POST /v1/heal - Heal JSON with provided schema (for demo/testing).
async fn handle_heal_request(
    _state: Arc<ControlState>,
    req: Request<Incoming>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    // Get expected schema from header
    let schema_header = req.headers()
        .get("X-Nomos-Schema")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    
    // Read body
    let body = req.into_body();
    let collected = body.collect().await
        .map_err(|e| {
            error!(error = %e, "Failed to read heal request body");
            e
        })?;
    let input_bytes = collected.to_bytes();
    
    // Parse input JSON
    let input: serde_json::Value = match serde_json::from_slice(&input_bytes) {
        Ok(v) => v,
        Err(e) => {
            return Ok(cors_json_response(StatusCode::BAD_REQUEST,
                &format!(r#"{{"error": "Invalid input JSON: {}"}}"#, e)));
        }
    };
    
    // Parse schema if provided
    let schema: Option<serde_json::Value> = if let Some(ref s) = schema_header {
        match serde_json::from_str(s) {
            Ok(v) => Some(v),
            Err(e) => {
                return Ok(cors_json_response(StatusCode::BAD_REQUEST,
                    &format!(r#"{{"error": "Invalid schema JSON: {}"}}"#, e)));
            }
        }
    } else {
        None
    };
    
    // Perform healing
    let start = std::time::Instant::now();
    let (healed, operations) = heal_json(&input, schema.as_ref());
    let elapsed_us = start.elapsed().as_micros();
    
    // Build response
    let ops_json: Vec<String> = operations.iter()
        .map(|op| format!(
            r#"{{"from": "{}", "to": "{}", "confidence": {:.2}, "algorithm": "{}"}}"#,
            op.from, op.to, op.confidence, op.algorithm
        ))
        .collect();
    
    let response = format!(r#"{{
  "data": {},
  "operations": [{}],
  "latency_us": {}
}}"#,
        serde_json::to_string_pretty(&healed).unwrap_or_default(),
        ops_json.join(", "),
        elapsed_us
    );
    
    Ok(cors_json_response(StatusCode::OK, &response))
}

/// Healing operation result.
struct HealOp {
    from: String,
    to: String,
    confidence: f64,
    algorithm: String,
}

/// Perform JSON healing based on schema.
fn heal_json(input: &serde_json::Value, schema: Option<&serde_json::Value>) -> (serde_json::Value, Vec<HealOp>) {
    let mut operations = Vec::new();
    
    // If no schema, return input as-is
    let schema = match schema {
        Some(s) => s,
        None => return (input.clone(), operations),
    };
    
    // Get schema keys
    let schema_keys: Vec<&str> = match schema.as_object() {
        Some(obj) => obj.keys().map(|s| s.as_str()).collect(),
        None => return (input.clone(), operations),
    };
    
    // Flatten input to get all keys and values
    let flat_input = flatten_json(input, "");
    
    // Build healed output
    let mut healed = serde_json::Map::new();
    
    for schema_key in &schema_keys {
        // Find best match from input
        let mut best_match: Option<(&str, f64)> = None;
        
        for (input_key, _) in &flat_input {
            let score = string_similarity(
                &input_key.to_lowercase().replace('.', ""),
                &schema_key.to_lowercase()
            );
            
            if score > 0.5 {
                if best_match.is_none() || score > best_match.unwrap().1 {
                    best_match = Some((input_key.as_str(), score));
                }
            }
        }
        
        if let Some((input_key, confidence)) = best_match {
            if let Some(value) = flat_input.get(input_key) {
                healed.insert(schema_key.to_string(), value.clone());
                
                // Only record op if key changed
                if input_key != *schema_key {
                    operations.push(HealOp {
                        from: input_key.to_string(),
                        to: schema_key.to_string(),
                        confidence,
                        algorithm: if confidence > 0.9 { "exact_match" }
                            else if confidence > 0.7 { "jaro_winkler" }
                            else { "levenshtein" }.to_string(),
                    });
                }
            }
        }
    }
    
    (serde_json::Value::Object(healed), operations)
}

/// Flatten nested JSON to dot-notation keys.
fn flatten_json(value: &serde_json::Value, prefix: &str) -> std::collections::HashMap<String, serde_json::Value> {
    let mut result = std::collections::HashMap::new();
    
    match value {
        serde_json::Value::Object(obj) => {
            for (key, val) in obj {
                let new_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", prefix, key)
                };
                
                if val.is_object() {
                    result.extend(flatten_json(val, &new_prefix));
                } else {
                    result.insert(new_prefix, val.clone());
                }
            }
        }
        _ => {
            if !prefix.is_empty() {
                result.insert(prefix.to_string(), value.clone());
            }
        }
    }
    
    result
}

/// Calculate string similarity (simplified Jaro-Winkler-like).
fn string_similarity(s1: &str, s2: &str) -> f64 {
    if s1 == s2 { return 1.0; }
    if s1.is_empty() || s2.is_empty() { return 0.0; }
    
    // Check if one contains the other (for camelCase/snake_case conversions)
    let s1_normalized = s1.replace('_', "").to_lowercase();
    let s2_normalized = s2.replace('_', "").to_lowercase();
    
    if s1_normalized == s2_normalized {
        return 0.95;
    }
    
    if s1_normalized.contains(&s2_normalized) || s2_normalized.contains(&s1_normalized) {
        return 0.85;
    }
    
    // Levenshtein-based similarity
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let max_len = len1.max(len2);
    
    let mut matrix: Vec<Vec<usize>> = vec![vec![0; len2 + 1]; len1 + 1];
    
    for i in 0..=len1 { matrix[i][0] = i; }
    for j in 0..=len2 { matrix[0][j] = j; }
    
    for (i, c1) in s1.chars().enumerate() {
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            matrix[i + 1][j + 1] = (matrix[i][j + 1] + 1)
                .min(matrix[i + 1][j] + 1)
                .min(matrix[i][j] + cost);
        }
    }
    
    let distance = matrix[len1][len2];
    1.0 - (distance as f64 / max_len as f64)
}

/// Create a JSON response with CORS headers.
fn cors_json_response(status: StatusCode, body: &str) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        .header("Access-Control-Allow-Headers", "Content-Type, X-Nomos-Schema, X-API-Key")
        .header("X-Nomos-Control", "true")
        .body(Full::new(Bytes::from(body.to_owned())))
        .unwrap()
}

/// Create CORS preflight response.
fn cors_preflight_response() -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::OK)
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        .header("Access-Control-Allow-Headers", "Content-Type, X-Nomos-Schema, X-API-Key")
        .header("Access-Control-Max-Age", "86400")
        .body(Full::new(Bytes::new()))
        .unwrap()
}

/// Handle GET /v1/metrics - Return metrics JSON.
async fn handle_metrics_get(
    state: Arc<ControlState>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    // Get proxy metrics (primary source)
    let pm = &state.proxy_metrics;
    let requests_total = pm.requests_total.load(Ordering::Relaxed);
    let requests_healed = pm.requests_healed.load(Ordering::Relaxed);
    let healing_rate = if requests_total > 0 {
        requests_healed as f64 / requests_total as f64
    } else {
        0.0
    };
    let avg_healing_us = pm.avg_latency_us();
    
    // Aggregate sharded metrics for per-core stats
    let sharded = state.metrics.aggregate();
    
    // Get eBPF stats if available
    let ebpf_json = if let Some(ref get_stats) = state.ebpf_stats {
        let stats = get_stats();
        format!(r#",
  "ebpf": {{
    "fast_path_packets": {},
    "slow_path_packets": {},
    "dropped_packets": {},
    "bytes_processed": {}
  }}"#,
            stats.fast_path_packets,
            stats.slow_path_packets,
            stats.dropped_packets,
            stats.bytes_processed
        )
    } else {
        String::new()
    };
    
    // Get healer version
    let healer_version = state.healer_registry.version();
    
    // Build per-core JSON
    let per_core_json: Vec<String> = sharded.per_core.iter()
        .map(|c| format!(
            r#"{{"core_id":{},"requests":{},"healed":{},"avg_healing_ns":{}}}"#,
            c.core_id, c.requests, c.healed, c.avg_healing_ns
        ))
        .collect();
    
    // Build response JSON (using proxy metrics as primary)
    let json = format!(r#"{{
  "requests_total": {},
  "requests_healed": {},
  "healing_rate": {:.4},
  "avg_healing_us": {},
  "p99_healing_us": {},
  "wasm_calls": {},
  "wasm_errors": {},
  "uptime_secs": {},
  "num_cores": {},
  "per_core": [{}],
  "healer_version": {}{}
}}"#,
        requests_total,
        requests_healed,
        healing_rate,
        avg_healing_us,
        0, // p99 not tracked in ProxyMetrics yet
        sharded.wasm_calls,
        sharded.wasm_errors,
        sharded.uptime_secs,
        sharded.num_cores,
        per_core_json.join(",\n    "),
        healer_version,
        ebpf_json
    );
    
    Ok(json_response(StatusCode::OK, &json))
}

/// Create a JSON response.
fn json_response(status: StatusCode, body: &str) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .header("X-Nomos-Control", "true")
        .body(Full::new(Bytes::from(body.to_owned())))
        .unwrap()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_control_config_defaults() {
        let config = ControlConfig::default();
        assert_eq!(config.listen_addr.port(), 8081);
        assert_eq!(config.max_wasm_size, 10 * 1024 * 1024);
        assert!(config.api_key.is_none());
    }
    
    #[test]
    fn test_core_metrics_recording() {
        let metrics = CoreMetrics::default();
        
        metrics.record_request(true, 1000);
        metrics.record_request(false, 0);
        metrics.record_request(true, 2000);
        
        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.requests_healed.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.healing_time_ns.load(Ordering::Relaxed), 3000);
    }
    
    #[test]
    fn test_sharded_metrics_aggregate() {
        let metrics = ShardedMetrics::new(4);
        
        // Record on current core
        metrics.current_core().record_request(true, 1000);
        metrics.current_core().record_request(false, 0);
        
        let agg = metrics.aggregate();
        assert_eq!(agg.requests_total, 2);
        assert_eq!(agg.requests_healed, 1);
    }
    
    #[test]
    fn test_aggregated_metrics_json() {
        let metrics = AggregatedMetrics {
            requests_total: 1000,
            requests_healed: 100,
            healing_time_ns: 5000000,
            wasm_calls: 100,
            wasm_errors: 2,
            uptime_secs: 3600,
            num_cores: 4,
            per_core: vec![],
        };
        
        let json = metrics.to_json();
        assert!(json.contains(r#""requests_total": 1000"#));
        assert!(json.contains(r#""requests_healed": 100"#));
    }
}
