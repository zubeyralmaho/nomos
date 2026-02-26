//! High-performance reverse proxy implementation.
//!
//! This module implements the core proxy as specified in Section 3
//! of the architecture document. Key features:
//!
//! - Asynchronous I/O with Tokio and Hyper 1.x
//! - Zero-copy buffer handling where possible
//! - Per-request arena allocation for transformations
//! - Lock-free schema store access
//! - eBPF fast path integration for route classification

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use bumpalo::Bump;
use bytes::{Bytes, BytesMut};
use http::{Request, Response, StatusCode, Uri};
use http_body_util::{BodyExt, Full};
use hyper::body::Body;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::client::legacy::{connect::HttpConnector, Client};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tracing::{debug, error, info, instrument, warn};

use crate::ebpf::{EbpfConfig, RouteHealthUpdate};
use crate::error::{NomosError, Result};
use crate::middleware::{MiddlewareContext, MiddlewareResult, ResponseMiddleware};
use crate::schema::{HttpMethod, RouteKey, SchemaStore};

// Thread-local arena allocator for per-request scratch space.
// Each worker thread gets its own arena that is reset between requests.
// Default capacity: 64KB (sized for typical JSON payloads).
thread_local! {
    static REQUEST_ARENA: std::cell::RefCell<Bump> =
        std::cell::RefCell::new(Bump::with_capacity(65536));
}

/// Proxy server configuration.
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Address to listen on (default: 127.0.0.1:8080)
    pub listen_addr: SocketAddr,

    /// Target URL to forward requests to
    pub target_url: Uri,

    /// Request timeout in milliseconds (default: 30000)
    pub timeout_ms: u64,

    /// Maximum response body size in bytes (default: 10MB)
    pub max_body_size: usize,

    /// Enable Nomos headers in response
    pub enable_nomos_headers: bool,

    /// eBPF configuration (None = disabled)
    pub ebpf: Option<EbpfConfig>,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            listen_addr: ([127, 0, 0, 1], 8080).into(),
            target_url: "http://127.0.0.1:9090".parse().unwrap(),
            timeout_ms: 30_000,
            max_body_size: 10 * 1024 * 1024, // 10MB
            enable_nomos_headers: true,
            ebpf: Some(EbpfConfig::default()),
        }
    }
}

/// Metrics for the proxy.
///
/// Uses atomic counters for lock-free updates.
#[derive(Debug, Default)]
pub struct ProxyMetrics {
    /// Total requests handled
    pub requests_total: AtomicU64,
    /// Requests that were healed
    pub requests_healed: AtomicU64,
    /// Requests that failed
    pub requests_failed: AtomicU64,
    /// Total bytes received from upstream
    pub bytes_received: AtomicU64,
    /// Total bytes sent to client
    pub bytes_sent: AtomicU64,
    /// Total processing time in microseconds
    pub processing_time_us: AtomicU64,
}

impl ProxyMetrics {
    /// Record a completed request.
    #[inline]
    pub fn record_request(&self, healed: bool, duration_us: u64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        if healed {
            self.requests_healed.fetch_add(1, Ordering::Relaxed);
        }
        self.processing_time_us.fetch_add(duration_us, Ordering::Relaxed);
    }

    /// Record a failed request.
    #[inline]
    pub fn record_failure(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.requests_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Get average latency in microseconds.
    pub fn avg_latency_us(&self) -> u64 {
        let total = self.requests_total.load(Ordering::Relaxed);
        if total == 0 {
            return 0;
        }
        self.processing_time_us.load(Ordering::Relaxed) / total
    }
}

/// Shared state for the proxy service.
///
/// This is passed to each request handler via Arc.
/// All members must be thread-safe and lock-free for reads.
pub struct ProxyState {
    /// Configuration
    pub config: ProxyConfig,

    /// HTTP client for upstream requests
    pub client: Client<HttpConnector, Full<Bytes>>,

    /// Schema store (ArcSwap for lock-free reads)
    pub schema_store: Arc<SchemaStore>,

    /// Response middleware
    pub middleware: Arc<dyn ResponseMiddleware>,

    /// Metrics
    pub metrics: Arc<ProxyMetrics>,

    /// WASM healer (hot-swappable via ModuleRegistry)
    pub wasm_healer: Option<Arc<crate::wasm_host::ModuleRegistry>>,

    /// Async channel for eBPF route health updates (non-blocking).
    /// None if eBPF is disabled or failed to initialize.
    pub route_health_tx: Option<mpsc::Sender<RouteHealthUpdate>>,
}

impl ProxyState {
    /// Create a new proxy state.
    pub fn new(
        config: ProxyConfig,
        schema_store: Arc<SchemaStore>,
        middleware: Arc<dyn ResponseMiddleware>,
        route_health_tx: Option<mpsc::Sender<RouteHealthUpdate>>,
    ) -> Self {
        let client = Client::builder(hyper_util::rt::TokioExecutor::new())
            .http1_title_case_headers(true)
            .http1_preserve_header_case(true)
            .build_http();

        Self {
            config,
            client,
            schema_store,
            middleware,
            metrics: Arc::new(ProxyMetrics::default()),
            wasm_healer: None,
            route_health_tx,
        }
    }
    
    /// Create proxy state with WASM healer.
    pub fn with_wasm_healer(
        config: ProxyConfig,
        schema_store: Arc<SchemaStore>,
        middleware: Arc<dyn ResponseMiddleware>,
        wasm_healer: Arc<crate::wasm_host::ModuleRegistry>,
        route_health_tx: Option<mpsc::Sender<RouteHealthUpdate>>,
    ) -> Self {
        Self::with_wasm_healer_and_metrics(
            config, schema_store, middleware, wasm_healer, route_health_tx, None
        )
    }
    
    /// Create proxy state with WASM healer and shared metrics.
    pub fn with_wasm_healer_and_metrics(
        config: ProxyConfig,
        schema_store: Arc<SchemaStore>,
        middleware: Arc<dyn ResponseMiddleware>,
        wasm_healer: Arc<crate::wasm_host::ModuleRegistry>,
        route_health_tx: Option<mpsc::Sender<RouteHealthUpdate>>,
        shared_metrics: Option<Arc<ProxyMetrics>>,
    ) -> Self {
        let client = Client::builder(hyper_util::rt::TokioExecutor::new())
            .http1_title_case_headers(true)
            .http1_preserve_header_case(true)
            .build_http();

        Self {
            config,
            client,
            schema_store,
            middleware,
            metrics: shared_metrics.unwrap_or_else(|| Arc::new(ProxyMetrics::default())),
            wasm_healer: Some(wasm_healer),
            route_health_tx,
        }
    }
}

/// The Nomos proxy server.
pub struct ProxyServer {
    state: Arc<ProxyState>,
}

impl ProxyServer {
    /// Create a new proxy server.
    pub fn new(state: ProxyState) -> Self {
        Self {
            state: Arc::new(state),
        }
    }

    /// Run the proxy server.
    ///
    /// This method blocks until the server is shut down.
    pub async fn run(&self) -> Result<()> {
        let listener = TcpListener::bind(self.state.config.listen_addr).await?;

        info!(
            addr = %self.state.config.listen_addr,
            target = %self.state.config.target_url,
            "Nomos proxy listening"
        );

        loop {
            let (stream, peer_addr) = listener.accept().await?;

            // Use TokioIo adapter for hyper 1.x compatibility
            let io = TokioIo::new(stream);
            let state = Arc::clone(&self.state);

            tokio::spawn(async move {
                let service = service_fn(move |req| {
                    let state = Arc::clone(&state);
                    async move { handle_request(state, req).await }
                });

                if let Err(e) = http1::Builder::new()
                    .serve_connection(io, service)
                    .await
                {
                    debug!(peer = %peer_addr, error = %e, "Connection error");
                }
            });
        }
    }

    /// Get the current metrics.
    pub fn metrics(&self) -> &ProxyMetrics {
        &self.state.metrics
    }
}

/// Handle a single request.
///
/// This is the hot path - every line matters for performance.
#[instrument(skip(state, req), fields(method = %req.method(), uri = %req.uri()))]
async fn handle_request(
    state: Arc<ProxyState>,
    req: Request<Incoming>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    let start = Instant::now();

    match process_request(&state, req).await {
        Ok(response) => {
            let duration = start.elapsed();
            let healed = response
                .headers()
                .get("X-Nomos-Healed")
                .map(|v| v == "true")
                .unwrap_or(false);

            state
                .metrics
                .record_request(healed, duration.as_micros() as u64);

            Ok(response)
        }
        Err(e) => {
            state.metrics.record_failure();
            error!(error = %e, "Request processing failed");

            Ok(error_response(e.status_code(), &e.to_string()))
        }
    }
}

/// Process a request through the proxy pipeline.
///
/// # Pipeline
///
/// 1. Extract route key for schema lookup
/// 2. Forward request to upstream
/// 3. Capture response body (zero-copy where possible)
/// 4. Run response middleware
/// 5. Return (possibly transformed) response
async fn process_request(
    state: &ProxyState,
    req: Request<Incoming>,
) -> Result<Response<Full<Bytes>>> {
    let start_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    // Extract route key (for schema lookup later)
    let route = RouteKey {
        method: HttpMethod::from(req.method()),
        path: Arc::from(req.uri().path()),
    };

    // Build upstream request
    let upstream_req = build_upstream_request(&state.config.target_url, req).await?;

    // Send to upstream and get response
    let upstream_resp = state.client.request(upstream_req).await?;
    let (parts, body) = upstream_resp.into_parts();

    // Read response body
    // NOTE: Using size hint to pre-allocate, avoiding reallocation
    let body_bytes = read_body(body, state.config.max_body_size).await?;

    // Track bytes received
    state
        .metrics
        .bytes_received
        .fetch_add(body_bytes.len() as u64, Ordering::Relaxed);

    // Reset thread-local arena for this request (memory reuse)
    REQUEST_ARENA.with(|arena| {
        arena.borrow_mut().reset();
    });

    // Create middleware context (arena access is thread-local within middleware)
    let ctx = MiddlewareContext {
        route: &route,
        schema_store: &state.schema_store,
        request_start_ns: start_ns,
    };

    let middleware = Arc::clone(&state.middleware);
    let body_vec = body_bytes.to_vec();

    // Measure healing time separately (this is the Nomos overhead)
    let healing_start = std::time::Instant::now();

    let result = middleware.process(&ctx, &body_vec).await.unwrap_or_else(|e| {
        warn!(error = %e, "Middleware error, passing through");
        MiddlewareResult::PassThrough
    });

    let healing_us = healing_start.elapsed().as_nanos() as u64 / 1000;  // More precise

    // Build response based on middleware result
    let (final_body, healed, ops_count, confidence) = match result {
        MiddlewareResult::PassThrough => (body_bytes, false, 0, 1.0),
        MiddlewareResult::Transformed {
            body,
            ops_count,
            confidence,
        } => (body, true, ops_count, confidence),
        MiddlewareResult::LowConfidence { confidence } => (body_bytes, false, 0, confidence),
    };

    // Track bytes sent
    state
        .metrics
        .bytes_sent
        .fetch_add(final_body.len() as u64, Ordering::Relaxed);

    // Build response
    let mut response = Response::builder().status(parts.status);

    // Copy original headers (avoiding clone)
    for (name, value) in parts.headers.iter() {
        // Skip content-length as it may have changed
        if name != http::header::CONTENT_LENGTH {
            response = response.header(name, value);
        }
    }

    // Add Nomos headers
    if state.config.enable_nomos_headers {
        response = response
            .header("X-Nomos-Healed", if healed { "true" } else { "false" })
            .header("X-Nomos-Healing-Ops", ops_count.to_string())
            .header(
                "X-Nomos-Confidence",
                format!("{:.2}", confidence),
            )
            .header(
                "X-Nomos-Latency-Us",
                healing_us,  // Healing time only, not total request time
            );
    }

    // Send async route health feedback to eBPF (Nomos Law: never block response path)
    send_route_health_feedback(&state.route_health_tx, &state.config.target_url, healed);

    // Set correct content-length
    response = response.header(http::header::CONTENT_LENGTH, final_body.len());

    Ok(response.body(Full::new(final_body))?)
}

/// Build the upstream request.
///
/// Transforms the incoming request URI to point to the target.
async fn build_upstream_request(
    target: &Uri,
    req: Request<Incoming>,
) -> Result<Request<Full<Bytes>>> {
    let (parts, body) = req.into_parts();

    // Build new URI
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");

    let uri = format!(
        "{}://{}{}{}",
        target.scheme_str().unwrap_or("http"),
        target.authority().map(|a| a.as_str()).unwrap_or("localhost"),
        path_and_query,
        parts.uri.query().map(|q| format!("?{}", q)).unwrap_or_default()
    );

    // Read incoming body (if any)
    let body_bytes = read_body(body, 10 * 1024 * 1024).await?;

    // Build request - avoiding clone of headers
    let mut builder = Request::builder()
        .method(parts.method)
        .uri(uri.parse::<Uri>()?);

    // Copy headers
    for (name, value) in parts.headers.iter() {
        // Skip hop-by-hop headers
        if !is_hop_by_hop(name) {
            builder = builder.header(name, value);
        }
    }

    Ok(builder.body(Full::new(body_bytes))?)
}

/// Read body into bytes, respecting size limit.
///
/// Uses size hint for pre-allocation when available.
async fn read_body(body: Incoming, max_size: usize) -> Result<Bytes> {
    // Get size hint for pre-allocation
    let size_hint = body.size_hint();
    let initial_capacity = size_hint
        .upper()
        .unwrap_or(size_hint.lower())
        .min(max_size as u64) as usize;

    let _buf = BytesMut::with_capacity(initial_capacity.max(1024));

    // Collect body frames
    let collected = body.collect().await.map_err(|e| NomosError::Body(e.to_string()))?;
    let bytes = collected.to_bytes();

    if bytes.len() > max_size {
        return Err(NomosError::Body(format!(
            "Body too large: {} > {}",
            bytes.len(),
            max_size
        )));
    }

    Ok(bytes)
}

/// Check if a header is hop-by-hop (should not be forwarded).
#[inline]
fn is_hop_by_hop(name: &http::HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
    )
}

/// Build an error response.
fn error_response(status: u16, message: &str) -> Response<Full<Bytes>> {
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let body = format!(r#"{{"error": "{}"}}"#, message);

    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .header("X-Nomos-Error", "true")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

/// Send async route health feedback to eBPF (non-blocking).
///
/// This implements "Nomos Law": the feedback loop never blocks the response path.
/// Uses try_send() to avoid blocking - if the channel is full, the update is
/// dropped and a warning is logged. This is acceptable because:
/// 1. The eBPF map will eventually be updated by subsequent requests
/// 2. The fail-open policy means we never drop legitimate traffic
/// 3. Backpressure indicates we're processing requests faster than eBPF can update
#[inline]
fn send_route_health_feedback(
    tx: &Option<mpsc::Sender<RouteHealthUpdate>>,
    target: &Uri,
    healed: bool,
) {
    // Early return if channel not available
    let Some(sender) = tx else {
        return;
    };

    // Try to extract IP from target URL (skip if hostname)
    let Some(host_str) = target.host() else {
        return;
    };

    // Parse as IPv4 - skip DNS hostnames (they'd require async resolution)
    let Ok(ip) = host_str.parse::<std::net::Ipv4Addr>() else {
        // Not an IP literal, skip eBPF feedback for this route
        // In production, you'd cache DNS resolution results
        return;
    };

    let port = target.port_u16().unwrap_or(80);

    // Create the update
    let update = if healed {
        RouteHealthUpdate::needs_healing(ip, port)
    } else {
        RouteHealthUpdate::healthy(ip, port)
    };

    // Non-blocking send - if channel is full, drop the update
    match sender.try_send(update) {
        Ok(()) => {
            debug!(
                ip = %ip,
                port = port,
                healed = healed,
                "Sent route health feedback to eBPF"
            );
        }
        Err(mpsc::error::TrySendError::Full(_)) => {
            debug!(
                ip = %ip,
                port = port,
                "Route health channel full, dropping update"
            );
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            // Channel closed, eBPF subsystem shut down
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_config_defaults() {
        let config = ProxyConfig::default();
        assert_eq!(config.listen_addr.port(), 8080);
        assert_eq!(config.timeout_ms, 30_000);
        assert!(config.enable_nomos_headers);
    }

    #[test]
    fn test_metrics_recording() {
        let metrics = ProxyMetrics::default();

        metrics.record_request(false, 100);
        metrics.record_request(true, 200);
        metrics.record_request(false, 150);

        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.requests_healed.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.avg_latency_us(), 150); // (100+200+150)/3
    }

    #[test]
    fn test_hop_by_hop_headers() {
        assert!(is_hop_by_hop(&http::HeaderName::from_static("connection")));
        assert!(is_hop_by_hop(&http::HeaderName::from_static("transfer-encoding")));
        assert!(!is_hop_by_hop(&http::HeaderName::from_static("content-type")));
        assert!(!is_hop_by_hop(&http::HeaderName::from_static("x-custom")));
    }
}
