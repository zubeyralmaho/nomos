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
        
        // GET /v1/metrics - Get metrics JSON
        (&http::Method::GET, "/v1/metrics") => {
            handle_metrics_get(state).await
        }
        
        // GET /v1/health - Health check
        (&http::Method::GET, "/v1/health") => {
            Ok(json_response(StatusCode::OK, r#"{"status": "healthy"}"#))
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
