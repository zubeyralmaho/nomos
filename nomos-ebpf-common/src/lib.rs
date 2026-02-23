//! Nomos eBPF Common - Shared kernel/userspace types
//!
//! This crate defines the data structures shared between the XDP program
//! running in kernel space and the Aya-based userspace loader.
//!
//! # Memory Layout
//!
//! All structures are `#[repr(C)]` with explicit padding to 8-byte boundaries.
//! This ensures identical layout in both BPF bytecode and userspace Rust.
//!
//! # Safety
//!
//! These types are used in BPF maps which require:
//! - Fixed, predictable size (`Copy` + `Pod`-like)
//! - No pointers or references
//! - Consistent endianness (network byte order for IP/port)

#![no_std]

/// Route health classification for the ROUTE_HEALTH BPF map.
///
/// Determines packet handling at XDP level:
/// - `Unknown`: First-time route, send to slow path for schema learning
/// - `Healthy`: Schema matches expected, fast-path eligible
/// - `NeedsHealing`: Schema drifted, requires transformation
/// - `Blocked`: Circuit breaker open, drop at kernel level
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RouteClass {
    /// Route not yet classified - pass to userspace for learning
    Unknown = 0,
    /// Schema healthy - eligible for fast-path bypass
    Healthy = 1,
    /// Schema drifted - needs healing transformation
    NeedsHealing = 2,
    /// Circuit breaker open - drop packets (XDP_DROP)
    Blocked = 3,
}

impl RouteClass {
    /// Convert from raw u32 value (for BPF map reads).
    #[inline(always)]
    pub const fn from_u32(v: u32) -> Self {
        match v {
            1 => RouteClass::Healthy,
            2 => RouteClass::NeedsHealing,
            3 => RouteClass::Blocked,
            _ => RouteClass::Unknown,
        }
    }
}

/// Key for the ROUTE_HEALTH BPF map.
///
/// Identifies a route by source IP (upstream server) and destination port.
/// 
/// # Memory Layout (8 bytes total, 4-byte aligned)
///
/// ```text
/// ┌────────────────┬────────────────┬────────────────┐
/// │  src_ip (4B)   │  dst_port (2B) │  _pad (2B)     │
/// └────────────────┴────────────────┴────────────────┘
/// ```
///
/// # Byte Order
///
/// - `src_ip`: Network byte order (big-endian) as received from packet
/// - `dst_port`: Network byte order (big-endian) as received from packet
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RouteKey {
    /// Source IP address (network byte order)
    /// This identifies the upstream API server
    pub src_ip: u32,
    /// Destination port (network byte order)
    /// Typically 8080 for the Nomos proxy
    pub dst_port: u16,
    /// Explicit padding to 8-byte boundary
    /// MUST be zero for consistent hashing
    pub _pad: u16,
}

impl RouteKey {
    /// Create a new route key.
    ///
    /// # Arguments
    /// - `src_ip`: Source IP in network byte order
    /// - `dst_port`: Destination port in network byte order
    #[inline(always)]
    pub const fn new(src_ip: u32, dst_port: u16) -> Self {
        Self {
            src_ip,
            dst_port,
            _pad: 0,
        }
    }

    /// Create from host byte order values (for userspace convenience).
    #[inline(always)]
    #[cfg(feature = "userspace")]
    pub fn from_host(src_ip: u32, dst_port: u16) -> Self {
        Self {
            src_ip: src_ip.to_be(),
            dst_port: dst_port.to_be(),
            _pad: 0,
        }
    }
}

/// Value for the ROUTE_HEALTH BPF map.
///
/// Contains route classification and metadata for observability.
///
/// # Memory Layout (16 bytes total, 8-byte aligned)
///
/// ```text
/// ┌────────────────┬────────────────┬────────────────────────────────┐
/// │  class (4B)    │  _pad1 (4B)    │  last_update_ns (8B)           │
/// └────────────────┴────────────────┴────────────────────────────────┘
/// ```
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RouteValue {
    /// Route classification (determines XDP action)
    pub class: u32,
    /// Explicit padding to 8-byte alignment
    pub _pad1: u32,
    /// Timestamp of last update (nanoseconds since boot, from bpf_ktime_get_ns)
    /// Used for stale entry detection
    pub last_update_ns: u64,
}

impl RouteValue {
    /// Create a new route value with the given classification.
    #[inline(always)]
    pub const fn new(class: RouteClass) -> Self {
        Self {
            class: class as u32,
            _pad1: 0,
            last_update_ns: 0,
        }
    }

    /// Create with timestamp (for userspace updates).
    #[inline(always)]
    pub const fn with_timestamp(class: RouteClass, timestamp_ns: u64) -> Self {
        Self {
            class: class as u32,
            _pad1: 0,
            last_update_ns: timestamp_ns,
        }
    }

    /// Get the route class.
    #[inline(always)]
    pub const fn get_class(&self) -> RouteClass {
        RouteClass::from_u32(self.class)
    }
}

/// Per-CPU packet statistics for observability.
///
/// # Memory Layout (32 bytes, 8-byte aligned)
///
/// ```text
/// ┌────────────────────────────────┬────────────────────────────────┐
/// │  fast_path_packets (8B)        │  slow_path_packets (8B)        │
/// ├────────────────────────────────┼────────────────────────────────┤
/// │  dropped_packets (8B)          │  bytes_processed (8B)          │
/// └────────────────────────────────┴────────────────────────────────┘
/// ```
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PacketStats {
    /// Packets that hit the fast path (Healthy routes)
    pub fast_path_packets: u64,
    /// Packets that went to slow path (Unknown/NeedsHealing)
    pub slow_path_packets: u64,
    /// Packets dropped (Blocked routes - circuit breaker)
    pub dropped_packets: u64,
    /// Total bytes processed through XDP
    pub bytes_processed: u64,
}

/// BPF map names - must match exactly between kernel and userspace.
pub mod map_names {
    /// LRU hash map for route health classification
    pub const ROUTE_HEALTH: &str = "ROUTE_HEALTH";
    /// Per-CPU array for packet statistics
    pub const PACKET_STATS: &str = "PACKET_STATS";
}

/// XDP program name - must match the #[xdp] attribute in kernel code.
pub const XDP_PROGRAM_NAME: &str = "nomos_xdp";

/// Default XDP listen port (proxy port).
pub const DEFAULT_XDP_PORT: u16 = 8080;

/// Maximum entries in the ROUTE_HEALTH map.
/// Sized for ~64K concurrent upstream routes.
pub const ROUTE_HEALTH_MAX_ENTRIES: u32 = 65536;

// Compile-time size assertions to ensure layout consistency
const _: () = {
    assert!(core::mem::size_of::<RouteKey>() == 8);
    assert!(core::mem::align_of::<RouteKey>() == 4);
    assert!(core::mem::size_of::<RouteValue>() == 16);
    assert!(core::mem::align_of::<RouteValue>() == 8);
    assert!(core::mem::size_of::<PacketStats>() == 32);
    assert!(core::mem::align_of::<PacketStats>() == 8);
};

#[cfg(feature = "userspace")]
mod userspace_impls {
    use super::*;
    use core::fmt;

    impl fmt::Debug for RouteKey {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let ip_bytes = self.src_ip.to_be_bytes();
            f.debug_struct("RouteKey")
                .field(
                    "src_ip",
                    &format_args!(
                        "{}.{}.{}.{}",
                        ip_bytes[0], ip_bytes[1], ip_bytes[2], ip_bytes[3]
                    ),
                )
                .field("dst_port", &u16::from_be(self.dst_port))
                .finish()
        }
    }

    impl fmt::Debug for RouteValue {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("RouteValue")
                .field("class", &self.get_class())
                .field("last_update_ns", &self.last_update_ns)
                .finish()
        }
    }

    impl fmt::Debug for RouteClass {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                RouteClass::Unknown => write!(f, "Unknown"),
                RouteClass::Healthy => write!(f, "Healthy"),
                RouteClass::NeedsHealing => write!(f, "NeedsHealing"),
                RouteClass::Blocked => write!(f, "Blocked"),
            }
        }
    }

    impl fmt::Debug for PacketStats {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("PacketStats")
                .field("fast_path", &self.fast_path_packets)
                .field("slow_path", &self.slow_path_packets)
                .field("dropped", &self.dropped_packets)
                .field("bytes", &self.bytes_processed)
                .finish()
        }
    }

    // SAFETY: These types are #[repr(C)], contain only primitive types,
    // have no padding that could leak data, and satisfy all Pod requirements:
    // - Copy (derived)
    // - 'static (no references)
    // - No interior mutability
    // - All bit patterns are valid
    unsafe impl aya::Pod for RouteKey {}
    unsafe impl aya::Pod for RouteValue {}
    unsafe impl aya::Pod for PacketStats {}
}
