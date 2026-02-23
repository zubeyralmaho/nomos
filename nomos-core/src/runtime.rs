//! Tokio runtime configuration with thread pinning for optimal cache locality.
//!
//! As specified in Section 4 of the architecture document, we configure
//! the runtime to:
//! - Pin workers to CPU cores for NUMA locality
//! - Use work-stealing for load balancing
//! - Limit blocking threads to avoid interference

use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::runtime::{Builder, Runtime};
use tracing::{info, warn};

/// Thread ID counter for naming workers
static WORKER_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Configuration for the Nomos runtime.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Number of worker threads (default: number of CPU cores)
    pub worker_threads: usize,
    /// Maximum blocking threads (default: 4)
    pub max_blocking_threads: usize,
    /// Enable thread pinning to CPU cores
    pub enable_cpu_pinning: bool,
    /// Thread stack size in bytes (default: 2MB)
    pub thread_stack_size: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus(),
            max_blocking_threads: 4,
            enable_cpu_pinning: true,
            thread_stack_size: 2 * 1024 * 1024, // 2MB
        }
    }
}

/// Returns the number of available CPU cores.
#[inline]
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Build the Tokio runtime with Nomos-specific optimizations.
///
/// # Configuration
///
/// - Multi-threaded work-stealing scheduler
/// - Thread pinning to CPU cores (if enabled)
/// - Limited blocking thread pool
/// - Named threads for debugging
///
/// # Example
///
/// ```
/// use nomos_core::runtime::build_runtime;
///
/// let rt = build_runtime(Default::default()).expect("runtime build failed");
/// rt.block_on(async {
///     println!("Nomos runtime initialized");
/// });
/// ```
pub fn build_runtime(config: RuntimeConfig) -> std::io::Result<Runtime> {
    let core_ids = if config.enable_cpu_pinning {
        core_affinity::get_core_ids().unwrap_or_default()
    } else {
        Vec::new()
    };

    let has_core_ids = !core_ids.is_empty();
    let core_ids_clone = core_ids.clone();

    info!(
        worker_threads = config.worker_threads,
        blocking_threads = config.max_blocking_threads,
        cpu_pinning = has_core_ids,
        "Building Nomos runtime"
    );

    let mut builder = Builder::new_multi_thread();

    builder
        .worker_threads(config.worker_threads)
        .max_blocking_threads(config.max_blocking_threads)
        .thread_stack_size(config.thread_stack_size)
        .enable_all()
        .thread_name_fn(move || {
            let id = WORKER_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("nomos-worker-{}", id)
        })
        .on_thread_start(move || {
            let worker_id = WORKER_COUNTER.load(Ordering::Relaxed).saturating_sub(1);

            // Pin thread to CPU core if cores are available
            if has_core_ids && worker_id < core_ids_clone.len() {
                let core_id = core_ids_clone[worker_id];
                if core_affinity::set_for_current(core_id) {
                    tracing::debug!(
                        worker_id = worker_id,
                        core_id = core_id.id,
                        "Worker thread pinned to core"
                    );
                } else {
                    warn!(
                        worker_id = worker_id,
                        core_id = core_id.id,
                        "Failed to pin worker to core"
                    );
                }
            }
        });

    builder.build()
}

/// Build a lightweight runtime for testing (single-threaded, no pinning).
#[cfg(test)]
pub fn build_test_runtime() -> Runtime {
    Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime build failed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RuntimeConfig::default();
        assert!(config.worker_threads >= 1);
        assert_eq!(config.max_blocking_threads, 4);
    }

    #[test]
    fn test_runtime_builds() {
        let config = RuntimeConfig {
            worker_threads: 2,
            max_blocking_threads: 1,
            enable_cpu_pinning: false,
            ..Default::default()
        };
        let rt = build_runtime(config).expect("runtime should build");
        rt.block_on(async {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        });
    }
}
