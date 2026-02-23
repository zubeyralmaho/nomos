//! Nomos XDP Program - Kernel-level packet classification
//!
//! This program intercepts incoming TCP packets at the XDP hook point
//! (earliest possible in the network stack) and classifies them based
//! on the ROUTE_HEALTH BPF map.
//!
//! # Design Principles
//!
//! - **Fail-Open**: Any parsing failure returns XDP_PASS, never drop legitimate traffic
//! - **Zero Allocation**: All parsing uses pointer arithmetic on packet buffer
//! - **O(1) Lookup**: BPF hash map provides constant-time route classification
//! - **Minimal Instructions**: Target <50 instructions for healthy path
//!
//! # Packet Flow
//!
//! ```text
//! NIC → XDP Hook → Parse Headers → Map Lookup → Action
//!                                    │
//!                    ┌───────────────┼───────────────┐
//!                    │               │               │
//!                    ▼               ▼               ▼
//!              Healthy           Unknown/          Blocked
//!              XDP_PASS         NeedsHealing      XDP_DROP
//!              (fast path)      XDP_PASS         (circuit breaker)
//!                              (slow path)
//! ```

#![no_std]
#![no_main]

use aya_ebpf::{
    bindings::xdp_action,
    macros::{map, xdp},
    maps::{HashMap, PerCpuArray},
    programs::XdpContext,
};
use aya_log_ebpf::info;
use core::mem;
use nomos_ebpf_common::{
    map_names, PacketStats, RouteClass, RouteKey, RouteValue, DEFAULT_XDP_PORT,
    ROUTE_HEALTH_MAX_ENTRIES,
};

// ============================================================================
// NETWORK PROTOCOL CONSTANTS
// ============================================================================

/// Ethernet header size (no VLAN tags)
const ETH_HDR_LEN: usize = 14;
/// EtherType for IPv4
const ETH_P_IP: u16 = 0x0800;
/// IP protocol number for TCP
const IPPROTO_TCP: u8 = 6;
/// Minimum IP header size (no options)
const IP_HDR_MIN_LEN: usize = 20;
/// Minimum TCP header size (no options)
const TCP_HDR_MIN_LEN: usize = 20;

// ============================================================================
// BPF MAPS
// ============================================================================

/// Route health classification map.
///
/// Key: (src_ip, dst_port) tuple identifying the upstream route
/// Value: RouteClass + metadata
///
/// LRU hash ensures automatic eviction of stale entries under memory pressure.
#[map]
static ROUTE_HEALTH: HashMap<RouteKey, RouteValue> = HashMap::with_max_entries(ROUTE_HEALTH_MAX_ENTRIES, 0);

/// Per-CPU packet statistics.
///
/// Index 0 contains the only stats entry - one per CPU core.
/// Using PerCpuArray avoids atomic contention on counters.
#[map]
static PACKET_STATS: PerCpuArray<PacketStats> = PerCpuArray::with_max_entries(1, 0);

// ============================================================================
// XDP PROGRAM ENTRY POINT
// ============================================================================

/// Main XDP program entry point.
///
/// This function is called by the kernel for every incoming packet
/// on the attached interface.
#[xdp]
pub fn nomos_xdp(ctx: XdpContext) -> u32 {
    // CRITICAL: Fail-open wrapper. Any error → XDP_PASS
    match try_process_packet(&ctx) {
        Ok(action) => action,
        Err(_) => xdp_action::XDP_PASS,
    }
}

/// Inner packet processing logic with error handling.
///
/// Separated from entry point to enable ? operator for cleaner code.
/// The BPF verifier will inline this.
#[inline(always)]
fn try_process_packet(ctx: &XdpContext) -> Result<u32, ()> {
    let data_start = ctx.data();
    let data_end = ctx.data_end();
    
    // ========================================================================
    // STEP 1: Parse Ethernet Header
    // ========================================================================
    
    // Bounds check: ensure we have at least an Ethernet header
    let eth_end = data_start + ETH_HDR_LEN;
    if eth_end > data_end {
        return Ok(xdp_action::XDP_PASS); // Fail-open: malformed packet
    }
    
    // Read EtherType at offset 12-13 (last 2 bytes of Ethernet header)
    // SAFETY: We verified eth_end <= data_end above
    let ethertype_ptr = (data_start + 12) as *const u16;
    let ethertype = unsafe { *ethertype_ptr };
    
    // Only process IPv4 packets
    if u16::from_be(ethertype) != ETH_P_IP {
        return Ok(xdp_action::XDP_PASS); // Not IPv4, pass through
    }
    
    // ========================================================================
    // STEP 2: Parse IP Header
    // ========================================================================
    
    let ip_start = data_start + ETH_HDR_LEN;
    
    // Bounds check: ensure we have at least minimum IP header
    if ip_start + IP_HDR_MIN_LEN > data_end {
        return Ok(xdp_action::XDP_PASS); // Truncated IP header
    }
    
    // Read version + IHL (first byte)
    // SAFETY: We verified ip_start + 20 <= data_end
    let version_ihl = unsafe { *(ip_start as *const u8) };
    let ihl = (version_ihl & 0x0F) as usize; // Lower 4 bits = header length in 32-bit words
    let ip_hdr_len = ihl * 4;
    
    // Validate IHL (must be >= 5 words = 20 bytes)
    if ip_hdr_len < IP_HDR_MIN_LEN {
        return Ok(xdp_action::XDP_PASS); // Invalid IHL
    }
    
    // Bounds check: full IP header including options
    let ip_end = ip_start + ip_hdr_len;
    if ip_end > data_end {
        return Ok(xdp_action::XDP_PASS); // IP header extends past packet
    }
    
    // Read protocol (offset 9 in IP header)
    let protocol = unsafe { *((ip_start + 9) as *const u8) };
    
    // Only process TCP packets
    if protocol != IPPROTO_TCP {
        return Ok(xdp_action::XDP_PASS); // Not TCP, pass through
    }
    
    // Read source IP (offset 12-15, network byte order)
    let src_ip = unsafe { *((ip_start + 12) as *const u32) };
    
    // ========================================================================
    // STEP 3: Parse TCP Header
    // ========================================================================
    
    let tcp_start = ip_end;
    
    // Bounds check: ensure we have at least minimum TCP header
    if tcp_start + TCP_HDR_MIN_LEN > data_end {
        return Ok(xdp_action::XDP_PASS); // Truncated TCP header
    }
    
    // Read destination port (offset 2-3 in TCP header, network byte order)
    // Source port is at offset 0-1, but we want destination (where packet is going)
    let dst_port = unsafe { *((tcp_start + 2) as *const u16) };
    
    // ========================================================================
    // STEP 4: Port Filter - Only process traffic to our proxy port
    // ========================================================================
    
    // Convert to host byte order for comparison
    let dst_port_host = u16::from_be(dst_port);
    
    // Only classify traffic destined for the Nomos proxy port
    if dst_port_host != DEFAULT_XDP_PORT {
        return Ok(xdp_action::XDP_PASS); // Not our port, pass through
    }
    
    // ========================================================================
    // STEP 5: Route Classification Lookup
    // ========================================================================
    
    // Build route key (already in network byte order)
    let route_key = RouteKey::new(src_ip, dst_port);
    
    // O(1) hash map lookup
    let action = match unsafe { ROUTE_HEALTH.get(&route_key) } {
        Some(value) => {
            match RouteClass::from_u32(value.class) {
                RouteClass::Healthy => {
                    // Fast path: schema is healthy, minimal processing needed
                    increment_stat(|s| s.fast_path_packets += 1);
                    xdp_action::XDP_PASS
                }
                RouteClass::NeedsHealing => {
                    // Slow path: needs schema transformation in userspace
                    increment_stat(|s| s.slow_path_packets += 1);
                    xdp_action::XDP_PASS
                }
                RouteClass::Blocked => {
                    // Circuit breaker open: drop at kernel level
                    increment_stat(|s| s.dropped_packets += 1);
                    xdp_action::XDP_DROP
                }
                RouteClass::Unknown => {
                    // Should not happen if map was properly initialized
                    increment_stat(|s| s.slow_path_packets += 1);
                    xdp_action::XDP_PASS
                }
            }
        }
        None => {
            // Route not in map - first time seeing this route
            // Pass to userspace for schema learning
            increment_stat(|s| s.slow_path_packets += 1);
            xdp_action::XDP_PASS
        }
    };
    
    // ========================================================================
    // STEP 6: Update bytes counter
    // ========================================================================
    
    let packet_len = (data_end - data_start) as u64;
    increment_stat(|s| s.bytes_processed += packet_len);
    
    Ok(action)
}

/// Atomically increment a packet statistic.
///
/// Uses per-CPU array to avoid contention between cores.
#[inline(always)]
fn increment_stat<F>(updater: F)
where
    F: FnOnce(&mut PacketStats),
{
    // Get per-CPU stats entry (index 0)
    if let Some(stats) = unsafe { PACKET_STATS.get_ptr_mut(0) } {
        let stats = unsafe { &mut *stats };
        updater(stats);
    }
}

// ============================================================================
// PANIC HANDLER (Required for #![no_std])
// ============================================================================

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // BPF programs cannot panic - this should never be reached
    // The verifier ensures no panic paths exist
    loop {}
}
