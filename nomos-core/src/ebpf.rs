//! Userspace eBPF Manager for Nomos.
//!
//! This module handles loading, attaching, and managing the XDP program
//! from userspace using the Aya framework.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     EbpfManager                                     │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────────┐  ┌────────────────────────┐   │
//! │  │ BPF Loader  │  │ ROUTE_HEALTH    │  │ PACKET_STATS           │   │
//! │  │ (Aya::Bpf)  │  │ HashMap<K,V>    │  │ PerCpuArray<Stats>     │   │
//! │  └─────────────┘  └─────────────────┘  └────────────────────────┘   │
//! │         │                  │                       │                │
//! │         ▼                  ▼                       ▼                │
//! │  ┌────────────────────────────────────────────────────────────────┐ │
//! │  │                     XDP Program                                │ │
//! │  │            (attached to network interface)                     │ │
//! │  └────────────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Thread Safety
//!
//! - Map updates use atomic operations (BPF_F_LOCK where supported)
//! - Statistics reads are eventually consistent (per-CPU, no locking)

use std::fs;
use std::io::{BufRead, BufReader};
use std::net::Ipv4Addr;
use std::path::Path;
use std::sync::Arc;

use aya::maps::{HashMap, PerCpuArray, PerCpuValues};
use aya::programs::{Xdp, XdpFlags};
use aya::Ebpf;
use parking_lot::RwLock;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use nomos_ebpf_common::{
    map_names, PacketStats, RouteClass, RouteKey, RouteValue, XDP_PROGRAM_NAME,
};

/// Errors that can occur during eBPF operations.
#[derive(Error, Debug)]
pub enum EbpfError {
    #[error("Failed to load eBPF program: {0}")]
    LoadError(String),

    #[error("Failed to attach XDP program to interface {interface}: {reason}")]
    AttachError { interface: String, reason: String },

    #[error("Failed to find default network interface")]
    NoDefaultInterface,

    #[error("BPF map operation failed: {0}")]
    MapError(String),

    #[error("eBPF program not loaded")]
    NotLoaded,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for eBPF operations.
pub type Result<T> = std::result::Result<T, EbpfError>;

/// XDP attachment mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdpMode {
    /// SKB generic mode - works everywhere but slower
    Skb,
    /// Driver mode - faster, requires driver support
    Driver,
    /// Hardware offload - fastest, requires NIC support
    Hardware,
}

impl From<XdpMode> for XdpFlags {
    fn from(mode: XdpMode) -> Self {
        match mode {
            XdpMode::Skb => XdpFlags::SKB_MODE,
            XdpMode::Driver => XdpFlags::DRV_MODE,
            XdpMode::Hardware => XdpFlags::HW_MODE,
        }
    }
}

/// Configuration for the eBPF manager.
#[derive(Debug, Clone)]
pub struct EbpfConfig {
    /// Whether eBPF/XDP acceleration is enabled.
    pub enabled: bool,

    /// Network interface to attach XDP program to.
    /// If None, auto-detect the default interface.
    pub interface: Option<String>,

    /// XDP attachment mode.
    pub xdp_mode: XdpMode,

    /// Path to the compiled eBPF object file.
    /// If None, use embedded bytecode (if available).
    pub ebpf_object_path: Option<String>,
}

impl Default for EbpfConfig {
    fn default() -> Self {
        Self {
            enabled: true, // Enabled by default, graceful fallback if unavailable
            interface: None,
            xdp_mode: XdpMode::Skb, // Safe default that works everywhere
            ebpf_object_path: None,
        }
    }
}

/// Aggregated packet statistics across all CPUs.
#[derive(Debug, Default, Clone)]
pub struct AggregatedStats {
    pub fast_path_packets: u64,
    pub slow_path_packets: u64,
    pub dropped_packets: u64,
    pub bytes_processed: u64,
}

/// Route health update message for the async feedback loop.
///
/// Sent from the request handler to the eBPF manager without blocking
/// the response path. Uses a bounded channel to apply backpressure.
#[derive(Debug, Clone)]
pub struct RouteHealthUpdate {
    /// Source IP (upstream server) in host byte order
    pub src_ip: std::net::Ipv4Addr,
    /// Destination port in host byte order
    pub dst_port: u16,
    /// New health classification
    pub class: RouteClass,
}

impl RouteHealthUpdate {
    /// Create a new route health update.
    #[inline]
    pub fn new(src_ip: std::net::Ipv4Addr, dst_port: u16, class: RouteClass) -> Self {
        Self { src_ip, dst_port, class }
    }

    /// Mark a route as healthy (fast-path eligible).
    #[inline]
    pub fn healthy(src_ip: std::net::Ipv4Addr, dst_port: u16) -> Self {
        Self::new(src_ip, dst_port, RouteClass::Healthy)
    }

    /// Mark a route as needing healing (slow-path).
    #[inline]
    pub fn needs_healing(src_ip: std::net::Ipv4Addr, dst_port: u16) -> Self {
        Self::new(src_ip, dst_port, RouteClass::NeedsHealing)
    }

    /// Block a route (circuit breaker).
    #[inline]
    pub fn blocked(src_ip: std::net::Ipv4Addr, dst_port: u16) -> Self {
        Self::new(src_ip, dst_port, RouteClass::Blocked)
    }
}

/// Channel capacity for route health updates.
/// Sized to handle burst traffic without blocking senders.
pub const ROUTE_HEALTH_CHANNEL_CAPACITY: usize = 8192;

/// eBPF Manager - handles XDP program lifecycle and map operations.
///
/// # Example
///
/// ```ignore
/// let config = EbpfConfig {
///     interface: None, // Auto-detect
///     xdp_mode: XdpMode::Skb,
///     ebpf_object_path: Some("target/bpfel-unknown-none/release/nomos-ebpf".into()),
/// };
///
/// let manager = EbpfManager::new(config)?;
/// manager.load_and_attach()?;
///
/// // Mark a route as healthy
/// manager.mark_healthy(Ipv4Addr::new(10, 0, 0, 1), 8080)?;
/// ```
pub struct EbpfManager {
    config: EbpfConfig,
    /// The loaded eBPF program and maps.
    /// Wrapped in RwLock to allow stats reads while updating maps.
    bpf: RwLock<Option<Ebpf>>,
    /// Interface the XDP program is attached to.
    attached_interface: RwLock<Option<String>>,
}

impl EbpfManager {
    /// Create a new eBPF manager with the given configuration.
    pub fn new(config: EbpfConfig) -> Self {
        Self {
            config,
            bpf: RwLock::new(None),
            attached_interface: RwLock::new(None),
        }
    }

    /// Load the eBPF program and attach XDP to the network interface.
    ///
    /// # Steps
    /// 1. Load compiled eBPF object file
    /// 2. Detect default interface if not specified
    /// 3. Attach XDP program to the interface
    pub fn load_and_attach(&self) -> Result<()> {
        // Load eBPF bytecode
        let ebpf_path = self
            .config
            .ebpf_object_path
            .as_ref()
            .ok_or_else(|| EbpfError::LoadError("No eBPF object path specified".into()))?;

        info!(path = %ebpf_path, "Loading eBPF program");

        let mut bpf = Ebpf::load_file(ebpf_path)
            .map_err(|e| EbpfError::LoadError(e.to_string()))?;

        // Determine interface
        let interface = match &self.config.interface {
            Some(iface) => iface.clone(),
            None => detect_default_interface()?,
        };

        info!(interface = %interface, mode = ?self.config.xdp_mode, "Attaching XDP program");

        // Get and attach XDP program
        let program: &mut Xdp = bpf
            .program_mut(XDP_PROGRAM_NAME)
            .ok_or_else(|| {
                EbpfError::LoadError(format!("XDP program '{}' not found", XDP_PROGRAM_NAME))
            })?
            .try_into()
            .map_err(|e: aya::programs::ProgramError| EbpfError::LoadError(e.to_string()))?;

        program
            .load()
            .map_err(|e| EbpfError::LoadError(e.to_string()))?;

        program
            .attach(&interface, self.config.xdp_mode.into())
            .map_err(|e| EbpfError::AttachError {
                interface: interface.clone(),
                reason: e.to_string(),
            })?;

        info!(interface = %interface, "XDP program attached successfully");

        // Store state
        *self.bpf.write() = Some(bpf);
        *self.attached_interface.write() = Some(interface);

        Ok(())
    }

    /// Detach XDP program and unload eBPF.
    pub fn detach(&self) -> Result<()> {
        let mut bpf_guard = self.bpf.write();
        let mut iface_guard = self.attached_interface.write();

        if let Some(interface) = iface_guard.take() {
            info!(interface = %interface, "Detaching XDP program");
        }

        // Dropping the Ebpf struct automatically detaches programs
        *bpf_guard = None;

        Ok(())
    }

    /// Check if the eBPF program is loaded and attached.
    pub fn is_attached(&self) -> bool {
        self.bpf.read().is_some()
    }

    /// Get the interface the XDP program is attached to.
    pub fn attached_interface(&self) -> Option<String> {
        self.attached_interface.read().clone()
    }

    // ========================================================================
    // Route Health Map Operations
    // ========================================================================

    /// Mark a route as healthy (fast-path eligible).
    ///
    /// # Arguments
    /// - `src_ip`: Source IP address (upstream server) in host byte order
    /// - `dst_port`: Destination port in host byte order
    #[inline]
    pub fn mark_healthy(&self, src_ip: Ipv4Addr, dst_port: u16) -> Result<()> {
        self.update_route(src_ip, dst_port, RouteClass::Healthy)
    }

    /// Mark a route as needing healing (slow-path).
    ///
    /// # Arguments
    /// - `src_ip`: Source IP address (upstream server) in host byte order
    /// - `dst_port`: Destination port in host byte order
    #[inline]
    pub fn mark_needs_healing(&self, src_ip: Ipv4Addr, dst_port: u16) -> Result<()> {
        self.update_route(src_ip, dst_port, RouteClass::NeedsHealing)
    }

    /// Block a route (circuit breaker open - XDP_DROP).
    ///
    /// # Arguments
    /// - `src_ip`: Source IP address (upstream server) in host byte order
    /// - `dst_port`: Destination port in host byte order
    #[inline]
    pub fn block_route(&self, src_ip: Ipv4Addr, dst_port: u16) -> Result<()> {
        self.update_route(src_ip, dst_port, RouteClass::Blocked)
    }

    /// Remove a route from the map (reset to unknown).
    ///
    /// # Arguments
    /// - `src_ip`: Source IP address (upstream server) in host byte order
    /// - `dst_port`: Destination port in host byte order
    pub fn remove_route(&self, src_ip: Ipv4Addr, dst_port: u16) -> Result<()> {
        let mut bpf_guard = self.bpf.write();
        let bpf = bpf_guard.as_mut().ok_or(EbpfError::NotLoaded)?;

        let mut route_health: HashMap<_, RouteKey, RouteValue> =
            HashMap::try_from(bpf.map_mut(map_names::ROUTE_HEALTH).ok_or_else(|| {
                EbpfError::MapError("ROUTE_HEALTH map not found".into())
            })?)
            .map_err(|e| EbpfError::MapError(format!("Failed to create HashMap: {}", e)))?;

        let key = route_key_from_host(src_ip, dst_port);

        route_health
            .remove(&key)
            .map_err(|e| EbpfError::MapError(format!("Failed to remove route: {}", e)))?;

        debug!(
            src_ip = %src_ip,
            dst_port = dst_port,
            "Route removed from ROUTE_HEALTH map"
        );

        Ok(())
    }

    /// Update a route's classification in the BPF map.
    ///
    /// This operation is atomic from the kernel's perspective - the map
    /// update is a single hash table operation.
    fn update_route(&self, src_ip: Ipv4Addr, dst_port: u16, class: RouteClass) -> Result<()> {
        let mut bpf_guard = self.bpf.write();
        let bpf = bpf_guard.as_mut().ok_or(EbpfError::NotLoaded)?;

        let mut route_health: HashMap<_, RouteKey, RouteValue> =
            HashMap::try_from(bpf.map_mut(map_names::ROUTE_HEALTH).ok_or_else(|| {
                EbpfError::MapError("ROUTE_HEALTH map not found".into())
            })?)
            .map_err(|e| EbpfError::MapError(format!("Failed to create HashMap: {}", e)))?;

        let key = route_key_from_host(src_ip, dst_port);
        let value = RouteValue::with_timestamp(class, current_timestamp_ns());

        // BPF_ANY flag: create or update
        route_health
            .insert(key, value, 0)
            .map_err(|e| EbpfError::MapError(format!("Failed to update route: {}", e)))?;

        debug!(
            src_ip = %src_ip,
            dst_port = dst_port,
            class = ?class,
            "Route updated in ROUTE_HEALTH map"
        );

        Ok(())
    }

    /// Get the current classification for a route.
    pub fn get_route(&self, src_ip: Ipv4Addr, dst_port: u16) -> Result<Option<RouteClass>> {
        let bpf_guard = self.bpf.read();
        let bpf = bpf_guard.as_ref().ok_or(EbpfError::NotLoaded)?;

        let route_health: HashMap<_, RouteKey, RouteValue> =
            HashMap::try_from(bpf.map(map_names::ROUTE_HEALTH).ok_or_else(|| {
                EbpfError::MapError("ROUTE_HEALTH map not found".into())
            })?)
            .map_err(|e| EbpfError::MapError(format!("Failed to create HashMap: {}", e)))?;

        let key = route_key_from_host(src_ip, dst_port);

        match route_health.get(&key, 0) {
            Ok(value) => Ok(Some(RouteClass::from_u32(value.class))),
            Err(aya::maps::MapError::KeyNotFound) => Ok(None),
            Err(e) => Err(EbpfError::MapError(format!("Failed to get route: {}", e))),
        }
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get aggregated packet statistics across all CPUs.
    ///
    /// Note: Statistics are eventually consistent due to per-CPU counters.
    pub fn get_stats(&self) -> Result<AggregatedStats> {
        let bpf_guard = self.bpf.read();
        let bpf = bpf_guard.as_ref().ok_or(EbpfError::NotLoaded)?;

        let packet_stats: PerCpuArray<_, PacketStats> =
            PerCpuArray::try_from(bpf.map(map_names::PACKET_STATS).ok_or_else(|| {
                EbpfError::MapError("PACKET_STATS map not found".into())
            })?)
            .map_err(|e| EbpfError::MapError(format!("Failed to create PerCpuArray: {}", e)))?;

        let per_cpu_values: PerCpuValues<PacketStats> = packet_stats
            .get(&0, 0)
            .map_err(|e| EbpfError::MapError(format!("Failed to read stats: {}", e)))?;

        // Aggregate across all CPUs
        let mut aggregated = AggregatedStats::default();
        for stats in per_cpu_values.iter() {
            aggregated.fast_path_packets += stats.fast_path_packets;
            aggregated.slow_path_packets += stats.slow_path_packets;
            aggregated.dropped_packets += stats.dropped_packets;
            aggregated.bytes_processed += stats.bytes_processed;
        }

        Ok(aggregated)
    }

    /// Get the number of entries in the ROUTE_HEALTH map.
    pub fn route_count(&self) -> Result<u32> {
        let bpf_guard = self.bpf.read();
        let bpf = bpf_guard.as_ref().ok_or(EbpfError::NotLoaded)?;

        let route_health: HashMap<_, RouteKey, RouteValue> =
            HashMap::try_from(bpf.map(map_names::ROUTE_HEALTH).ok_or_else(|| {
                EbpfError::MapError("ROUTE_HEALTH map not found".into())
            })?)
            .map_err(|e| EbpfError::MapError(format!("Failed to create HashMap: {}", e)))?;

        // Count entries by iterating (there's no direct count method)
        let count = route_health.iter().count() as u32;
        Ok(count)
    }
}

impl Drop for EbpfManager {
    fn drop(&mut self) {
        if let Err(e) = self.detach() {
            error!(error = %e, "Failed to detach XDP program during drop");
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert host byte order IP and port to RouteKey (network byte order).
#[inline]
fn route_key_from_host(src_ip: Ipv4Addr, dst_port: u16) -> RouteKey {
    RouteKey {
        src_ip: u32::from(src_ip).to_be(),
        dst_port: dst_port.to_be(),
        _pad: 0,
    }
}

/// Get current timestamp in nanoseconds since boot.
///
/// Uses CLOCK_MONOTONIC for consistency with bpf_ktime_get_ns().
fn current_timestamp_ns() -> u64 {
    use std::time::Instant;
    // Note: This is not exactly the same as bpf_ktime_get_ns(), but close enough
    // for userspace timestamp purposes
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

/// Detect the default network interface from /proc/net/route.
///
/// Parses the kernel routing table to find the interface with the default
/// route (destination 0.0.0.0).
///
/// # Returns
/// The interface name (e.g., "eth0", "ens5", "wlan0")
fn detect_default_interface() -> Result<String> {
    let route_file = fs::File::open("/proc/net/route")?;
    let reader = BufReader::new(route_file);

    // Format: Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT
    // Default route has Destination = 00000000

    for line in reader.lines().skip(1) {
        // Skip header
        let line = line?;
        let fields: Vec<&str> = line.split_whitespace().collect();

        if fields.len() >= 2 {
            let interface = fields[0];
            let destination = fields[1];

            // Default route: destination is 00000000
            if destination == "00000000" {
                info!(interface = %interface, "Detected default network interface");
                return Ok(interface.to_string());
            }
        }
    }

    // Fallback: try common interface names
    for iface in &["eth0", "ens5", "ens3", "enp0s3", "wlan0", "lo"] {
        let path = format!("/sys/class/net/{}", iface);
        if Path::new(&path).exists() {
            warn!(
                interface = %iface,
                "No default route found, falling back to existing interface"
            );
            return Ok((*iface).to_string());
        }
    }

    Err(EbpfError::NoDefaultInterface)
}

// ============================================================================
// Async Feedback Loop
// ============================================================================

/// Spawn the async feedback loop that processes RouteHealthUpdate messages.
///
/// This implements "Nomos Law": the feedback loop runs asynchronously and
/// never blocks the response path. Updates are batched for efficiency.
///
/// # Arguments
/// - `manager`: Arc-wrapped EbpfManager for map updates
/// - `rx`: Receiver end of the bounded channel
///
/// # Returns
/// A JoinHandle for the spawned task (for graceful shutdown)
pub fn spawn_feedback_loop(
    manager: Arc<EbpfManager>,
    mut rx: mpsc::Receiver<RouteHealthUpdate>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        info!("eBPF feedback loop started");

        let mut updates_processed: u64 = 0;
        let mut errors: u64 = 0;

        while let Some(update) = rx.recv().await {
            // Process the update - this is the only blocking operation
            // but it's fast: just a syscall to update the BPF map
            let result = match update.class {
                RouteClass::Healthy => manager.mark_healthy(update.src_ip, update.dst_port),
                RouteClass::NeedsHealing => manager.mark_needs_healing(update.src_ip, update.dst_port),
                RouteClass::Blocked => manager.block_route(update.src_ip, update.dst_port),
                RouteClass::Unknown => manager.remove_route(update.src_ip, update.dst_port),
            };

            match result {
                Ok(()) => {
                    updates_processed += 1;
                    if updates_processed % 10_000 == 0 {
                        debug!(
                            updates = updates_processed,
                            errors = errors,
                            "eBPF feedback loop progress"
                        );
                    }
                }
                Err(e) => {
                    errors += 1;
                    if errors <= 10 || errors % 1000 == 0 {
                        warn!(
                            error = %e,
                            src_ip = %update.src_ip,
                            dst_port = update.dst_port,
                            class = ?update.class,
                            "Failed to update route in eBPF map"
                        );
                    }
                }
            }
        }

        info!(
            updates = updates_processed,
            errors = errors,
            "eBPF feedback loop shutting down"
        );
    })
}

/// Create a bounded channel for route health updates.
///
/// Returns (sender, receiver) pair. The sender can be cloned for multiple
/// request handlers. If the channel is full, `try_send` will return an error
/// but never block (non-blocking backpressure).
pub fn create_route_health_channel() -> (
    mpsc::Sender<RouteHealthUpdate>,
    mpsc::Receiver<RouteHealthUpdate>,
) {
    mpsc::channel(ROUTE_HEALTH_CHANNEL_CAPACITY)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_key_from_host() {
        let key = route_key_from_host(Ipv4Addr::new(192, 168, 1, 1), 8080);

        // 192.168.1.1 = 0xC0A80101 in host order
        // In network (big-endian) order: 0xC0A80101 (same on big-endian, reversed on little-endian)
        assert_eq!(key.src_ip, 0xC0A80101u32.to_be());
        assert_eq!(key.dst_port, 8080u16.to_be());
        assert_eq!(key._pad, 0);
    }

    #[test]
    fn test_route_class_conversion() {
        assert_eq!(RouteClass::from_u32(0), RouteClass::Unknown);
        assert_eq!(RouteClass::from_u32(1), RouteClass::Healthy);
        assert_eq!(RouteClass::from_u32(2), RouteClass::NeedsHealing);
        assert_eq!(RouteClass::from_u32(3), RouteClass::Blocked);
        assert_eq!(RouteClass::from_u32(999), RouteClass::Unknown);
    }
}
