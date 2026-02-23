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
//! ```

use std::env;
use std::sync::Arc;

use nomos_core::middleware::PassThroughMiddleware;
use nomos_core::runtime::{build_runtime, RuntimeConfig};
use nomos_core::schema::SchemaStore;
use nomos_core::{ProxyConfig, ProxyServer, ProxyState};
use tracing::{error, info};
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::EnvFilter;

/// Environment variable for target URL
const ENV_TARGET_URL: &str = "TARGET_URL";

/// Environment variable for listen address
const ENV_LISTEN_ADDR: &str = "LISTEN_ADDR";

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

    info!(
        listen = %config.listen_addr,
        target = %config.target_url,
        "Starting Nomos proxy"
    );

    // Create shared state
    let schema_store = Arc::new(SchemaStore::new());
    let middleware = Arc::new(PassThroughMiddleware);

    let state = ProxyState::new(config, schema_store, middleware);
    let server = ProxyServer::new(state);

    // Print startup banner
    print_banner();

    // Run the server
    server.run().await
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
