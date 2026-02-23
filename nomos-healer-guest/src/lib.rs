//! Nomos Healer Guest - WASM Implementation
//!
//! This crate provides a sandboxed JSON healer that runs in WebAssembly.
//! It implements the WIT interface defined in `nomos-core/wit/healer.wit`.
//!
//! # Design Principles
//!
//! 1. **Shared Memory**: Input/output buffers are in linear memory to avoid
//!    serialization overhead. Host writes JSON directly, guest reads in-place.
//!
//! 2. **Zero-Copy Where Possible**: Operate on byte slices, avoid String allocs.
//!
//! 3. **Minimal Dependencies**: Only serde_json for parsing. No heavy runtime.
//!
//! # Memory Layout (compact version for default WASM memory ~1MB)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    WASM LINEAR MEMORY                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  0x10000: INPUT_BUFFER  (256KB) ← Host writes JSON here         │
//! │  0x50000: OUTPUT_BUFFER (256KB) ← Guest writes healed output    │
//! │  0x90000: INSTR_BUFFER  (64KB)  ← Healing instructions          │
//! │  0xA0000: HEAP          (Remaining) ← Guest heap allocation     │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

#![no_std]

extern crate alloc;

mod healer;
mod json_transform;

pub use healer::*;

// ============================================================================
// Global Allocator for no_std WASM
// ============================================================================

use core::alloc::{GlobalAlloc, Layout};

/// Simple bump allocator for WASM - uses reserved memory region
struct WasmBumpAlloc;

// Allocator state - starts after instruction buffer
#[allow(dead_code)]  // Used conceptually for documentation
static mut HEAP_START: usize = 0xA0000;  // After instruction buffer
static mut HEAP_PTR: usize = 0xA0000;
const HEAP_END: usize = 0x110000;  // Up to ~1.1MB boundary

unsafe impl GlobalAlloc for WasmBumpAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let align = layout.align();
        let size = layout.size();
        
        // Align the current pointer
        let aligned_ptr = (HEAP_PTR + align - 1) & !(align - 1);
        
        if aligned_ptr + size > HEAP_END {
            return core::ptr::null_mut();  // OOM
        }
        
        HEAP_PTR = aligned_ptr + size;
        aligned_ptr as *mut u8
    }
    
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Bump allocator never deallocates
        // Memory is reclaimed when WASM instance is reset
    }
}

#[global_allocator]
static ALLOC: WasmBumpAlloc = WasmBumpAlloc;

// ============================================================================
// Panic Handler
// ============================================================================

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // In WASM, we can't really recover from panic
    // Just trap (unreachable instruction)
    core::arch::wasm32::unreachable()
}
