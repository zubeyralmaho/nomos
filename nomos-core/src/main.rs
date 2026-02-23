//! Nomos - Zero-latency autonomous schema-healing proxy
//!
//! This is the main entry point for the Nomos proxy server.
//!
//! # Usage
//!
//! ```bash
//! # Start with default settings (localhost:8080 -> localhost:9090)
//! nomos-core
//!
//! # Set target URL via environment
//! TARGET_URL=http://api.example.com:80 nomos-core
//!
//! # Set listen address
//! LISTEN_ADDR=0.0.0.0:8080 nomos-core
//!
//! # Set control plane address
//! CONTROL_ADDR=0.0.0.0:8081 nomos-core
//! ```

use std::env;
use std::sync::Arc;

use nomos_core::control::{ControlConfig, ControlServer, ControlState, ShardedMetrics};
use nomos_core::ebpf::{create_route_health_channel, spawn_feedback_loop, EbpfConfig, EbpfManager};
use nomos_core::middleware::WasmHealingMiddleware;
use nomos_core::proxy::ProxyMetrics;
use nomos_core::runtime::{build_runtime, RuntimeConfig};
use nomos_core::schema::SchemaStore;
use nomos_core::wasm_host::ModuleRegistry;
use nomos_core::{ProxyConfig, ProxyServer, ProxyState};
use tracing::{error, info, warn};
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::EnvFilter;

/// Embedded default WASM healer binary (built from nomos-healer-guest)
static DEFAULT_HEALER_WASM: &[u8] = include_bytes!(
    "../../nomos-healer-guest/target/wasm32-wasip1/release/nomos_healer_guest.wasm"
);

/// Environment variable for target URL
const ENV_TARGET_URL: &str = "TARGET_URL";

/// Environment variable for listen address
const ENV_LISTEN_ADDR: &str = "LISTEN_ADDR";

/// Environment variable for control plane address
const ENV_CONTROL_ADDR: &str = "CONTROL_ADDR";

/// Environment variable for worker threads
const ENV_WORKER_THREADS: &str = "WORKER_THREADS";

/// Environment variable for CPU pinning
const ENV_CPU_PINNING: &str = "CPU_PINNING";

fn main() {
    // Initialize tracing
    init_tracing();

    // Build runtime config from environment
    let runtime_config = RuntimeConfig {
        worker_threads: env::var(ENV_WORKER_THREADS)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            }),
        enable_cpu_pinning: env::var(ENV_CPU_PINNING)
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true),
        ..Default::default()
    };

    info!(
        worker_threads = runtime_config.worker_threads,
        cpu_pinning = runtime_config.enable_cpu_pinning,
        "Initializing Nomos runtime"
    );

    // Build the runtime
    let runtime = match build_runtime(runtime_config) {
        Ok(rt) => rt,
        Err(e) => {
            error!(error = %e, "Failed to build runtime");
            std::process::exit(1);
        }
    };

    // Run the proxy server
    runtime.block_on(async {
        if let Err(e) = run_proxy().await {
            error!(error = %e, "Proxy server failed");
            std::process::exit(1);
        }
    });
}

/// Initialize the tracing subscriber.
fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("nomos_core=debug,info")
    });

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_span_events(FmtSpan::CLOSE)
        .with_target(true)
        .with_thread_ids(true)
        .with_level(true)
        .init();
}

/// Run the proxy server.
async fn run_proxy() -> nomos_core::Result<()> {
    // Build proxy configuration
    let config = build_config();
    
    // Get number of worker threads for metrics sharding
    let num_cores = env::var(ENV_WORKER_THREADS)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

    info!(
        listen = %config.listen_addr,
        target = %config.target_url,
        "Starting Nomos proxy"
    );

    // Initialize eBPF with graceful fallback
    let route_health_tx = initialize_ebpf(&config).await;

    // Initialize WASM healer with embedded default module
    let wasm_healer = match ModuleRegistry::new(DEFAULT_HEALER_WASM) {
        Ok(registry) => {
            info!(
                size_bytes = DEFAULT_HEALER_WASM.len(),
                "Loaded embedded WASM healer"
            );
            Arc::new(registry)
        }
        Err(e) => {
            error!(error = %e, "Failed to initialize WASM healer");
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("WASM healer initialization failed: {}", e)
            ).into());
        }
    };

    // Create shared metrics (lock-free, per-core)
    let sharded_metrics = Arc::new(ShardedMetrics::new(num_cores));
    let proxy_metrics = Arc::new(ProxyMetrics::default());
    
    // Create shared state
    let schema_store = Arc::new(SchemaStore::new());
    
    // Create healing middleware with WASM healer
    let middleware = Arc::new(WasmHealingMiddleware::new(wasm_healer.clone()));

    let state = ProxyState::with_wasm_healer_and_metrics(
        config, 
        schema_store, 
        middleware, 
        wasm_healer.clone(),
        route_health_tx,
        Some(proxy_metrics.clone()),  // Share metrics with control plane
    );
    let server = ProxyServer::new(state);

    // Start control plane server
    let control_config = build_control_config();
    let control_state = ControlState {
        config: control_config.clone(),
        healer_registry: wasm_healer,
        metrics: sharded_metrics,
        proxy_metrics,
        ebpf_stats: None, // TODO: Wire up eBPF stats
    };
    let control_server = ControlServer::new(control_state);
    
    // Spawn control server on separate task
    tokio::spawn(async move {
        if let Err(e) = control_server.run().await {
            error!(error = %e, "Control plane server failed");
        }
    });

    info!(
        control_addr = %control_config.listen_addr,
        "Control plane started"
    );

    // Print startup banner
    print_banner();

    // Run the proxy server (blocking)
    server.run().await
}

/// Build control plane configuration from environment.
fn build_control_config() -> ControlConfig {
    let listen_addr = env::var(ENV_CONTROL_ADDR)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| ([127, 0, 0, 1], 8081).into());
    
    ControlConfig {
        listen_addr,
        ..Default::default()
    }
}

/// Initialize eBPF/XDP with graceful fallback.
///
/// Returns Some(sender) if eBPF initialized successfully, None otherwise.
/// The proxy continues to function without eBPF - it's an optimization, not a requirement.
async fn initialize_ebpf(
    config: &ProxyConfig,
) -> Option<tokio::sync::mpsc::Sender<nomos_core::ebpf::RouteHealthUpdate>> {
    // Check if eBPF is enabled in config
    let ebpf_config = match &config.ebpf {
        Some(cfg) if cfg.enabled => cfg.clone(),
        Some(_) => {
            info!("eBPF disabled in configuration");
            return None;
        }
        None => {
            info!("No eBPF configuration, using default");
            EbpfConfig::default()
        }
    };

    info!(
        path = ebpf_config.ebpf_object_path.as_deref().unwrap_or("default"),
        interface = ebpf_config.interface.as_deref().unwrap_or("auto-detect"),
        mode = ?ebpf_config.xdp_mode,
        "Initializing eBPF subsystem"
    );

    // Create the eBPF manager
    let manager = Arc::new(EbpfManager::new(ebpf_config));

    // Attempt to load and attach
    match manager.load_and_attach() {
        Ok(()) => {
            info!(
                interface = manager.attached_interface().as_deref().unwrap_or("unknown"),
                "eBPF/XDP program attached successfully"
            );

            // Create channel and spawn feedback loop
            let (tx, rx) = create_route_health_channel();
            let _handle = spawn_feedback_loop(manager.clone(), rx);

            // Spawn periodic stats reporter
            spawn_stats_reporter(manager);

            Some(tx)
        }
        Err(e) => {
            warn!(
                error = %e,
                "Failed to initialize eBPF - continuing without fast path"
            );
            warn!("This is normal if running without root/CAP_BPF or if the eBPF program is not compiled");
            warn!("The proxy will function correctly, but without kernel-level acceleration");
            None
        }
    }
}

/// Spawn a background task to periodically log eBPF statistics.
fn spawn_stats_reporter(manager: Arc<EbpfManager>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;

            match manager.get_stats() {
                Ok(stats) => {
                    info!(
                        fast_path = stats.fast_path_packets,
                        slow_path = stats.slow_path_packets,
                        dropped = stats.dropped_packets,
                        bytes = stats.bytes_processed,
                        "eBPF packet statistics"
                    );
                }
                Err(e) => {
                    warn!(error = %e, "Failed to read eBPF statistics");
                }
            }

            // Also log route count
            match manager.route_count() {
                Ok(count) => {
                    info!(routes = count, "Active routes in ROUTE_HEALTH map");
                }
                Err(e) => {
                    warn!(error = %e, "Failed to read route count");
                }
            }
        }
    });
}

/// Build proxy configuration from environment.
fn build_config() -> ProxyConfig {
    let listen_addr = env::var(ENV_LISTEN_ADDR)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| ([127, 0, 0, 1], 8080).into());

    let target_url = env::var(ENV_TARGET_URL)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| "http://127.0.0.1:9090".parse().unwrap());

    ProxyConfig {
        listen_addr,
        target_url,
        enable_nomos_headers: true,
        ..Default::default()
    }
}

/// Print the startup banner.
fn print_banner() {
    eprintln!(
        r#"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███╗   ██╗ ██████╗ ███╗   ███╗ ██████╗ ███████╗            ║
║   ████╗  ██║██╔═══██╗████╗ ████║██╔═══██╗██╔════╝            ║
║   ██╔██╗ ██║██║   ██║██╔████╔██║██║   ██║███████╗            ║
║   ██║╚██╗██║██║   ██║██║╚██╔╝██║██║   ██║╚════██║            ║
║   ██║ ╚████║╚██████╔╝██║ ╚═╝ ██║╚██████╔╝███████║            ║
║   ╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝ ╚═════╝ ╚══════╝            ║
║                                                               ║
║   Zero-latency autonomous schema-healing proxy                ║
║   It never stops.                                             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"#
    );
}
