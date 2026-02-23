//! Core healer implementation - WASM exports.
//!
//! This module exports the functions defined in the WIT interface.
//! Uses raw exports compatible with Wasmtime's component model.

use alloc::string::String;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::slice;

use crate::json_transform::{apply_healing_ops, HealingInstruction, Operation};

// ============================================================================
// Static Buffers & State
// ============================================================================

/// Maximum buffer sizes (match architecture spec)
const MAX_INPUT_SIZE: usize = 512 * 1024;   // 512KB
const MAX_OUTPUT_SIZE: usize = 512 * 1024;  // 512KB

/// Healer state - initialized once via `init()`
struct HealerState {
    input_ptr: *const u8,
    input_cap: usize,
    output_ptr: *mut u8,
    output_cap: usize,
    initialized: bool,
}

impl Default for HealerState {
    fn default() -> Self {
        Self {
            input_ptr: core::ptr::null(),
            input_cap: 0,
            output_ptr: core::ptr::null_mut(),
            output_cap: 0,
            initialized: false,
        }
    }
}

// WASM is single-threaded, so we can safely use static mut wrapped in UnsafeCell
struct SyncState(UnsafeCell<HealerState>);
unsafe impl Sync for SyncState {}

impl SyncState {
    #[inline]
    fn get(&self) -> &mut HealerState {
        // SAFETY: WASM is single-threaded
        unsafe { &mut *self.0.get() }
    }
}

static STATE: SyncState = SyncState(UnsafeCell::new(HealerState {
    input_ptr: core::ptr::null(),
    input_cap: 0,
    output_ptr: core::ptr::null_mut(),
    output_cap: 0,
    initialized: false,
}));

// ============================================================================
// WIT Interface Exports
// ============================================================================

/// Error codes matching WIT enum
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum ErrorCode {
    Ok = 0,
    ParseError = 1,
    BufferOverflow = 2,
    FieldNotFound = 3,
    CoercionFailed = 4,
    InternalError = 5,
}

/// Result struct matching WIT record (packed for FFI)
#[repr(C)]
pub struct HealingResult {
    pub success: u32,          // bool as u32
    pub output_len: u32,       // bytes written to output
    pub error_code: u32,       // ErrorCode
    pub ops_applied: u32,      // number of operations
    pub processing_ns: u64,    // timing (filled by host)
}

/// Initialize the healer with shared memory regions.
///
/// # Safety
/// Pointers must be valid and within WASM linear memory bounds.
#[no_mangle]
pub extern "C" fn healer_init(
    input_ptr: u32,
    input_cap: u32,
    output_ptr: u32,
    output_cap: u32,
) {
    let s = STATE.get();
    s.input_ptr = input_ptr as *const u8;
    s.input_cap = (input_cap as usize).min(MAX_INPUT_SIZE);
    s.output_ptr = output_ptr as *mut u8;
    s.output_cap = (output_cap as usize).min(MAX_OUTPUT_SIZE);
    s.initialized = true;
}

/// Execute healing transformations on JSON in shared memory.
///
/// # Arguments
/// - `input_len`: Actual length of input JSON
/// - `instructions_ptr`: Pointer to serialized instructions
/// - `instructions_len`: Length of instructions buffer
///
/// # Returns
/// Pointer to HealingResult in linear memory (guest allocates)
#[no_mangle]
pub extern "C" fn healer_heal(
    input_len: u32,
    instructions_ptr: u32,
    instructions_len: u32,
) -> u64 {
    // Pack result into u64: high 32 bits = output_len, low 32 = error_code | success
    let result = heal_impl(input_len, instructions_ptr, instructions_len);
    
    // Encode: success(1 bit) | error_code(7 bits) | ops_applied(8 bits) | output_len(16 bits)
    // Then extend to u64 for more space
    let success_bit = if result.success != 0 { 1u64 } else { 0u64 };
    let error_bits = (result.error_code as u64 & 0x7F) << 1;
    let ops_bits = (result.ops_applied as u64 & 0xFF) << 8;
    let len_bits = (result.output_len as u64 & 0xFFFF) << 16;
    
    success_bit | error_bits | ops_bits | len_bits
}

fn heal_impl(input_len: u32, instructions_ptr: u32, instructions_len: u32) -> HealingResult {
    let s = STATE.get();
    
    if !s.initialized {
        return HealingResult {
            success: 0,
            output_len: 0,
            error_code: ErrorCode::InternalError as u32,
            ops_applied: 0,
            processing_ns: 0,
        };
    }
    
    let input_len = input_len as usize;
    let instructions_len = instructions_len as usize;
    
    // Bounds check
    if input_len > s.input_cap {
        return HealingResult {
            success: 0,
            output_len: 0,
            error_code: ErrorCode::BufferOverflow as u32,
            ops_applied: 0,
            processing_ns: 0,
        };
    }
    
    // SAFETY: Host guarantees valid pointers within linear memory
    let input_slice = unsafe { slice::from_raw_parts(s.input_ptr, input_len) };
    let output_slice = unsafe { slice::from_raw_parts_mut(s.output_ptr, s.output_cap) };
    
    // Parse instructions
    let instructions_slice = unsafe {
        slice::from_raw_parts(instructions_ptr as *const u8, instructions_len)
    };
    
    let instructions = match parse_instructions(instructions_slice) {
        Ok(i) => i,
        Err(_) => {
            return HealingResult {
                success: 0,
                output_len: 0,
                error_code: ErrorCode::ParseError as u32,
                ops_applied: 0,
                processing_ns: 0,
            };
        }
    };
    
    // Apply healing
    match apply_healing_ops(input_slice, &instructions, output_slice) {
        Ok((written, ops_count)) => HealingResult {
            success: 1,
            output_len: written as u32,
            error_code: ErrorCode::Ok as u32,
            ops_applied: ops_count as u32,
            processing_ns: 0,
        },
        Err(e) => HealingResult {
            success: 0,
            output_len: 0,
            error_code: e as u32,
            ops_applied: 0,
            processing_ns: 0,
        },
    }
}

/// Parse serialized instructions from binary format.
///
/// Format: [count: u32][instructions...]
/// Each instruction: [op: u8][path_len: u16][path: bytes][source_len: u16][source: bytes][target_len: u16][target: bytes]
fn parse_instructions(data: &[u8]) -> Result<Vec<HealingInstruction>, ErrorCode> {
    if data.len() < 4 {
        return Ok(Vec::new()); // No instructions
    }
    
    let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let mut instructions = Vec::with_capacity(count);
    let mut offset = 4;
    
    for _ in 0..count {
        if offset >= data.len() {
            break;
        }
        
        // Operation type
        let op = match data.get(offset).copied() {
            Some(0) => Operation::Rename,
            Some(1) => Operation::CoerceType,
            Some(2) => Operation::SetDefault,
            Some(3) => Operation::Delete,
            _ => return Err(ErrorCode::ParseError),
        };
        offset += 1;
        
        // Source field name
        if offset + 2 > data.len() {
            return Err(ErrorCode::ParseError);
        }
        let source_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;
        
        if offset + source_len > data.len() {
            return Err(ErrorCode::ParseError);
        }
        let source = core::str::from_utf8(&data[offset..offset + source_len])
            .map_err(|_| ErrorCode::ParseError)?;
        offset += source_len;
        
        // Target field name
        if offset + 2 > data.len() {
            return Err(ErrorCode::ParseError);
        }
        let target_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;
        
        if offset + target_len > data.len() {
            return Err(ErrorCode::ParseError);
        }
        let target = core::str::from_utf8(&data[offset..offset + target_len])
            .map_err(|_| ErrorCode::ParseError)?;
        offset += target_len;
        
        instructions.push(HealingInstruction {
            op,
            source: String::from(source),
            target: String::from(target),
        });
    }
    
    Ok(instructions)
}

/// Validate JSON fingerprint (fast path check).
///
/// Returns 1 if fingerprint matches, 0 otherwise.
#[no_mangle]
pub extern "C" fn healer_validate_fingerprint(input_len: u32, expected_fingerprint: u64) -> u32 {
    let s = STATE.get();
    
    if !s.initialized || (input_len as usize) > s.input_cap {
        return 0;
    }
    
    let input_slice = unsafe { slice::from_raw_parts(s.input_ptr, input_len as usize) };
    let computed = compute_fingerprint_impl(input_slice);
    
    if computed == expected_fingerprint { 1 } else { 0 }
}

/// Get protocol version string.
#[no_mangle]
pub extern "C" fn healer_version() -> u64 {
    // Return pointer and length packed into u64
    // Version: "1.0.0"
    static VERSION: &[u8] = b"1.0.0";
    let ptr = VERSION.as_ptr() as u64;
    let len = VERSION.len() as u64;
    (ptr << 32) | len
}

// ============================================================================
// Schema-Aware Interface (Optional)
// ============================================================================

/// Compute schema fingerprint from JSON keys.
///
/// Uses xxHash-like algorithm for fast fingerprinting.
#[no_mangle]
pub extern "C" fn schema_compute_fingerprint(input_len: u32) -> u64 {
    let s = STATE.get();
    
    if !s.initialized || (input_len as usize) > s.input_cap {
        return 0;
    }
    
    let input_slice = unsafe { slice::from_raw_parts(s.input_ptr, input_len as usize) };
    compute_fingerprint_impl(input_slice)
}

/// Simple fingerprint computation - hash of JSON keys only.
fn compute_fingerprint_impl(json: &[u8]) -> u64 {
    // FNV-1a hash of keys (simplified - full impl would parse JSON)
    const FNV_PRIME: u64 = 0x100000001b3;
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    
    let mut hash = FNV_OFFSET;
    let mut in_key = false;
    let mut depth = 0u32;
    
    for &b in json {
        match b {
            b'{' => depth += 1,
            b'}' => depth = depth.saturating_sub(1),
            b'"' if depth > 0 => {
                in_key = !in_key;
                if in_key {
                    // Start of key - include depth in hash
                    hash ^= depth as u64;
                    hash = hash.wrapping_mul(FNV_PRIME);
                }
            }
            _ if in_key => {
                hash ^= b as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            b':' => in_key = false,
            _ => {}
        }
    }
    
    hash
}

// ============================================================================
// Memory Allocator for WASM
// ============================================================================

/// Simple bump allocator for guest-side allocations.
/// Used for returning strings/buffers to host.
static mut BUMP_PTR: usize = 0x120000;  // After instruction buffer
const BUMP_END: usize = 0x140000;       // 128KB for allocations

/// Allocate memory for guest use (called by host if needed).
#[no_mangle]
pub extern "C" fn guest_alloc(size: u32) -> u32 {
    unsafe {
        let aligned_size = ((size as usize) + 7) & !7;  // 8-byte alignment
        if BUMP_PTR + aligned_size > BUMP_END {
            return 0;  // OOM
        }
        let ptr = BUMP_PTR;
        BUMP_PTR += aligned_size;
        ptr as u32
    }
}

/// Reset bump allocator (call between requests).
#[no_mangle]
pub extern "C" fn guest_reset() {
    unsafe {
        BUMP_PTR = 0x120000;
    }
}
