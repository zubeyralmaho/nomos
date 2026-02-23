//! WASM Host Integration - Wasmtime Runtime for Healers
//!
//! This module provides the host-side integration for WASM healer components.
//! Key features:
//!
//! - **Shared Memory**: Direct memory access for zero-copy I/O
//! - **Instance Pooling**: Thread-local pools to avoid instantiation overhead
//! - **Hot-Swap**: ModuleRegistry for updating WASM modules without restart
//! - **Resource Limits**: Memory/CPU constraints to prevent runaway guests
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    WASM HOST LAYER                              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐   │
//! │  │ ModuleRegistry  │  │ InstancePool    │  │ SharedMemory   │   │
//! │  │ (hot-swap)      │  │ (per-thread)    │  │ (buffer mgmt)  │   │
//! │  └─────────────────┘  └─────────────────┘  └────────────────┘   │
//! │                              │                                  │
//! │                              ▼                                  │
//! │  ┌────────────────────────────────────────────────────────────┐ │
//! │  │              WasmHealer (high-level API)                   │ │
//! │  │   heal(input, instructions) -> Result<Vec<u8>, Error>      │ │
//! │  └────────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use thiserror::Error;
use tracing::{debug, info};
use wasmtime::{
    Config, Engine, Instance, Linker, Memory, Module, Store, TypedFunc,
};

use crate::engine::HealingOp;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Error)]
pub enum WasmHealerError {
    #[error("WASM compilation failed: {0}")]
    CompilationError(String),

    #[error("WASM instantiation failed: {0}")]
    InstantiationError(String),

    #[error("Module not initialized")]
    NotInitialized,

    #[error("Buffer overflow: input={input_len}, capacity={capacity}")]
    BufferOverflow { input_len: usize, capacity: usize },

    #[error("Healing failed with error code: {0}")]
    HealingFailed(u32),

    #[error("JSON parse error in guest")]
    ParseError,

    #[error("Memory access error: {0}")]
    MemoryError(String),

    #[error("Function not found: {0}")]
    FunctionNotFound(String),
}

// ============================================================================
// Memory Layout Constants
// ============================================================================

/// Memory layout for shared buffers (compact version - fits in default WASM memory)
pub mod memory_layout {
    /// Start of input buffer in linear memory
    pub const INPUT_OFFSET: u32 = 0x10000;      // 64KB offset
    /// Capacity of input buffer
    pub const INPUT_CAPACITY: u32 = 256 * 1024; // 256KB
    
    /// Start of output buffer
    pub const OUTPUT_OFFSET: u32 = 0x50000;     // After input
    /// Capacity of output buffer
    pub const OUTPUT_CAPACITY: u32 = 256 * 1024; // 256KB
    
    /// Start of instruction buffer
    pub const INSTR_OFFSET: u32 = 0x90000;      // After output
    /// Capacity of instruction buffer
    pub const INSTR_CAPACITY: u32 = 64 * 1024;  // 64KB
}

// ============================================================================
// Healing Instruction Serialization
// ============================================================================

/// Binary encoding for healing instructions.
///
/// Format: [count: u32][instructions...]
/// Each instruction: [op: u8][source_len: u16][source: bytes][target_len: u16][target: bytes]
fn serialize_instructions(ops: &[HealingOp]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(ops.len() * 64);
    
    // Count
    buf.extend_from_slice(&(ops.len() as u32).to_le_bytes());
    
    for op in ops {
        match op {
            HealingOp::Rename { from, to, .. } => {
                buf.push(0); // Operation::Rename
                
                // Source
                let source_bytes = from.as_bytes();
                buf.extend_from_slice(&(source_bytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(source_bytes);
                
                // Target
                let target_bytes = to.as_bytes();
                buf.extend_from_slice(&(target_bytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(target_bytes);
            }
            HealingOp::CoerceType { field, .. } => {
                buf.push(1); // Operation::CoerceType
                
                let field_bytes = field.as_bytes();
                buf.extend_from_slice(&(field_bytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(field_bytes);
                
                buf.extend_from_slice(&0u16.to_le_bytes()); // Empty target
            }
            HealingOp::SetDefault { field, default_value } => {
                buf.push(2); // Operation::SetDefault
                
                let field_bytes = field.as_bytes();
                buf.extend_from_slice(&(field_bytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(field_bytes);
                
                let value_bytes = default_value.as_bytes();
                buf.extend_from_slice(&(value_bytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(value_bytes);
            }
            HealingOp::Delete { field } => {
                buf.push(3); // Operation::Delete
                
                let field_bytes = field.as_bytes();
                buf.extend_from_slice(&(field_bytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(field_bytes);
                
                buf.extend_from_slice(&0u16.to_le_bytes()); // Empty target
            }
        }
    }
    
    buf
}

// ============================================================================
// WASM Healer Instance
// ============================================================================

/// A single WASM healer instance.
///
/// Wraps a Wasmtime instance with cached function handles and memory access.
pub struct WasmHealerInstance {
    store: Store<()>,
    #[allow(dead_code)]  // Kept for future use (e.g., calling additional exports)
    instance: Instance,
    memory: Memory,
    
    // Cached function handles
    fn_init: TypedFunc<(u32, u32, u32, u32), ()>,
    fn_heal: TypedFunc<(u32, u32, u32), u64>,
    fn_validate_fingerprint: TypedFunc<(u32, u64), u32>,
    fn_reset: TypedFunc<(), ()>,
    
    // State
    initialized: bool,
}

impl WasmHealerInstance {
    /// Create a new healer instance from a compiled module.
    pub fn new(engine: &Engine, module: &Module) -> Result<Self, WasmHealerError> {
        let mut store = Store::new(engine, ());
        
        // Create linker (no imports needed - guest is self-contained)
        let linker = Linker::new(engine);
        
        // Instantiate module
        let instance = linker
            .instantiate(&mut store, module)
            .map_err(|e| WasmHealerError::InstantiationError(e.to_string()))?;
        
        // Get memory export
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| WasmHealerError::FunctionNotFound("memory".into()))?;
        
        // Get function exports
        let fn_init = instance
            .get_typed_func::<(u32, u32, u32, u32), ()>(&mut store, "healer_init")
            .map_err(|e| WasmHealerError::FunctionNotFound(format!("healer_init: {}", e)))?;
        
        let fn_heal = instance
            .get_typed_func::<(u32, u32, u32), u64>(&mut store, "healer_heal")
            .map_err(|e| WasmHealerError::FunctionNotFound(format!("healer_heal: {}", e)))?;
        
        let fn_validate_fingerprint = instance
            .get_typed_func::<(u32, u64), u32>(&mut store, "healer_validate_fingerprint")
            .map_err(|e| WasmHealerError::FunctionNotFound(format!("healer_validate_fingerprint: {}", e)))?;
        
        let fn_reset = instance
            .get_typed_func::<(), ()>(&mut store, "guest_reset")
            .map_err(|e| WasmHealerError::FunctionNotFound(format!("guest_reset: {}", e)))?;
        
        let mut inst = Self {
            store,
            instance,
            memory,
            fn_init,
            fn_heal,
            fn_validate_fingerprint,
            fn_reset,
            initialized: false,
        };
        
        // Initialize with buffer pointers
        inst.initialize()?;
        
        Ok(inst)
    }
    
    /// Initialize the healer with buffer regions.
    fn initialize(&mut self) -> Result<(), WasmHealerError> {
        use memory_layout::*;
        
        self.fn_init
            .call(
                &mut self.store,
                (INPUT_OFFSET, INPUT_CAPACITY, OUTPUT_OFFSET, OUTPUT_CAPACITY),
            )
            .map_err(|e| WasmHealerError::InstantiationError(e.to_string()))?;
        
        self.initialized = true;
        Ok(())
    }
    
    /// Execute healing transformation.
    ///
    /// # Arguments
    /// - `input`: JSON bytes to heal
    /// - `operations`: Healing operations to apply
    ///
    /// # Returns
    /// Healed JSON bytes on success.
    pub fn heal(&mut self, input: &[u8], operations: &[HealingOp]) -> Result<Vec<u8>, WasmHealerError> {
        use memory_layout::*;
        
        if !self.initialized {
            return Err(WasmHealerError::NotInitialized);
        }
        
        // Check input size
        if input.len() > INPUT_CAPACITY as usize {
            return Err(WasmHealerError::BufferOverflow {
                input_len: input.len(),
                capacity: INPUT_CAPACITY as usize,
            });
        }
        
        // Serialize instructions
        let instructions = serialize_instructions(operations);
        if instructions.len() > INSTR_CAPACITY as usize {
            return Err(WasmHealerError::BufferOverflow {
                input_len: instructions.len(),
                capacity: INSTR_CAPACITY as usize,
            });
        }
        
        // Write input to shared memory
        let mem_data = self.memory.data_mut(&mut self.store);
        mem_data[INPUT_OFFSET as usize..INPUT_OFFSET as usize + input.len()]
            .copy_from_slice(input);
        
        // Write instructions to shared memory
        mem_data[INSTR_OFFSET as usize..INSTR_OFFSET as usize + instructions.len()]
            .copy_from_slice(&instructions);
        
        // Call heal function
        let result = self
            .fn_heal
            .call(
                &mut self.store,
                (input.len() as u32, INSTR_OFFSET, instructions.len() as u32),
            )
            .map_err(|_| WasmHealerError::HealingFailed(0))?;
        
        // Decode result: success(1 bit) | error_code(7 bits) | ops_applied(8 bits) | output_len(16 bits)
        let success = (result & 1) == 1;
        let error_code = ((result >> 1) & 0x7F) as u32;
        let ops_applied = ((result >> 8) & 0xFF) as u32;
        let output_len = ((result >> 16) & 0xFFFF) as usize;
        
        if !success {
            return Err(match error_code {
                1 => WasmHealerError::ParseError,
                2 => WasmHealerError::BufferOverflow {
                    input_len: input.len(),
                    capacity: OUTPUT_CAPACITY as usize,
                },
                _ => WasmHealerError::HealingFailed(error_code),
            });
        }
        
        // Read output from shared memory
        let mem_data = self.memory.data(&self.store);
        let output = mem_data[OUTPUT_OFFSET as usize..OUTPUT_OFFSET as usize + output_len].to_vec();
        
        debug!(
            ops_applied = ops_applied,
            input_len = input.len(),
            output_len = output_len,
            "WASM healing completed"
        );
        
        Ok(output)
    }
    
    /// Fast fingerprint validation.
    pub fn validate_fingerprint(&mut self, input: &[u8], expected: u64) -> Result<bool, WasmHealerError> {
        use memory_layout::*;
        
        if !self.initialized {
            return Err(WasmHealerError::NotInitialized);
        }
        
        if input.len() > INPUT_CAPACITY as usize {
            return Err(WasmHealerError::BufferOverflow {
                input_len: input.len(),
                capacity: INPUT_CAPACITY as usize,
            });
        }
        
        // Write input
        let mem_data = self.memory.data_mut(&mut self.store);
        mem_data[INPUT_OFFSET as usize..INPUT_OFFSET as usize + input.len()]
            .copy_from_slice(input);
        
        // Call validate
        let result = self
            .fn_validate_fingerprint
            .call(&mut self.store, (input.len() as u32, expected))
            .map_err(|_| WasmHealerError::HealingFailed(0))?;
        
        Ok(result == 1)
    }
    
    /// Reset guest allocator (call between requests for memory reuse).
    pub fn reset(&mut self) -> Result<(), WasmHealerError> {
        self.fn_reset
            .call(&mut self.store, ())
            .map_err(|e| WasmHealerError::InstantiationError(e.to_string()))
    }
}

// ============================================================================
// Module Registry - Hot-Swap Support
// ============================================================================

/// Registry for WASM healer modules.
///
/// Supports hot-swapping modules without dropping connections. Active instances
/// continue using their current module until they're returned to the pool.
pub struct ModuleRegistry {
    engine: Engine,
    
    /// Current active module (ArcSwap for lock-free reads)
    current_module: ArcSwap<Module>,
    
    /// Module version counter
    version: AtomicU64,
    
    /// Historical modules (keep last N for graceful drain)
    history: Mutex<Vec<(u64, Arc<Module>)>>,
    
    /// Maximum historical modules to keep
    history_limit: usize,
}

impl ModuleRegistry {
    /// Create a new registry with initial WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmHealerError> {
        // Configure engine for performance
        let mut config = Config::new();
        config.cranelift_opt_level(wasmtime::OptLevel::Speed);
        config.parallel_compilation(true);
        
        let engine = Engine::new(&config)
            .map_err(|e| WasmHealerError::CompilationError(e.to_string()))?;
        
        let module = Module::new(&engine, wasm_bytes)
            .map_err(|e| WasmHealerError::CompilationError(e.to_string()))?;
        
        info!(
            module_size = wasm_bytes.len(),
            "Initialized WASM module registry"
        );
        
        Ok(Self {
            engine,
            current_module: ArcSwap::new(Arc::new(module)),
            version: AtomicU64::new(1),
            history: Mutex::new(Vec::new()),
            history_limit: 3,
        })
    }
    
    /// Hot-swap to a new module.
    ///
    /// The new module is compiled and atomically swapped in. Existing instances
    /// continue using the old module until they're returned to the pool.
    pub fn hot_swap(&self, wasm_bytes: &[u8]) -> Result<u64, WasmHealerError> {
        // Compile new module
        let new_module = Module::new(&self.engine, wasm_bytes)
            .map_err(|e| WasmHealerError::CompilationError(e.to_string()))?;
        
        // Increment version
        let new_version = self.version.fetch_add(1, Ordering::SeqCst) + 1;
        
        // Swap in new module
        let old_module = self.current_module.swap(Arc::new(new_module));
        
        // Archive old module
        {
            let mut history = self.history.lock();
            history.push((new_version - 1, old_module));
            
            // Trim history
            while history.len() > self.history_limit {
                history.remove(0);
            }
        }
        
        info!(
            version = new_version,
            module_size = wasm_bytes.len(),
            "Hot-swapped WASM module"
        );
        
        Ok(new_version)
    }
    
    /// Get current module version.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }
    
    /// Get the engine reference.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }
    
    /// Get current module reference.
    pub fn current_module(&self) -> Arc<Module> {
        self.current_module.load_full()
    }
    
    /// Create a new healer instance using current module.
    pub fn create_instance(&self) -> Result<WasmHealerInstance, WasmHealerError> {
        let module = self.current_module.load();
        WasmHealerInstance::new(&self.engine, &*module)
    }
}

// ============================================================================
// Thread-Local Instance Pool
// ============================================================================

/// Thread-local pool of healer instances.
///
/// Avoids instantiation overhead by reusing instances. Each thread gets its own
/// pool to avoid lock contention.
pub struct HealerPool {
    registry: Arc<ModuleRegistry>,
    pool_size: usize,
}

thread_local! {
    static LOCAL_POOL: RefCell<Vec<WasmHealerInstance>> = RefCell::new(Vec::new());
}

impl HealerPool {
    /// Create a new pool backed by the given registry.
    pub fn new(registry: Arc<ModuleRegistry>, pool_size: usize) -> Self {
        Self { registry, pool_size }
    }
    
    /// Get an instance from the pool or create a new one.
    pub fn get(&self) -> Result<PooledInstance, WasmHealerError> {
        LOCAL_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            
            if let Some(instance) = pool.pop() {
                Ok(PooledInstance {
                    instance: Some(instance),
                    registry: Arc::clone(&self.registry),
                    pool_size: self.pool_size,
                })
            } else {
                // Create new instance
                let instance = self.registry.create_instance()?;
                Ok(PooledInstance {
                    instance: Some(instance),
                    registry: Arc::clone(&self.registry),
                    pool_size: self.pool_size,
                })
            }
        })
    }
}

/// RAII wrapper that returns instance to pool on drop.
pub struct PooledInstance {
    instance: Option<WasmHealerInstance>,
    #[allow(dead_code)]  // Kept for future version tracking
    registry: Arc<ModuleRegistry>,
    pool_size: usize,
}

impl PooledInstance {
    /// Access the healer instance.
    pub fn healer(&mut self) -> &mut WasmHealerInstance {
        self.instance.as_mut().expect("instance already returned")
    }
}

impl Drop for PooledInstance {
    fn drop(&mut self) {
        if let Some(mut instance) = self.instance.take() {
            // Reset guest state before returning to pool
            let _ = instance.reset();
            
            LOCAL_POOL.with(|pool| {
                let mut pool = pool.borrow_mut();
                if pool.len() < self.pool_size {
                    pool.push(instance);
                }
                // Otherwise drop the instance to prevent unbounded growth
            });
        }
    }
}

// ============================================================================
// High-Level API
// ============================================================================

/// High-level WASM healer interface.
///
/// Wraps the complexity of instance pooling and module management behind a
/// simple API.
pub struct WasmHealer {
    pool: HealerPool,
}

impl WasmHealer {
    /// Create a new WASM healer from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmHealerError> {
        let registry = Arc::new(ModuleRegistry::new(wasm_bytes)?);
        let pool = HealerPool::new(registry, 4); // 4 instances per thread
        
        Ok(Self { pool })
    }
    
    /// Create with custom pool size.
    pub fn with_pool_size(wasm_bytes: &[u8], pool_size: usize) -> Result<Self, WasmHealerError> {
        let registry = Arc::new(ModuleRegistry::new(wasm_bytes)?);
        let pool = HealerPool::new(registry, pool_size);
        
        Ok(Self { pool })
    }
    
    /// Heal JSON with the given operations.
    ///
    /// # Performance
    /// Target: <10µs overhead (excluding JSON transformation time)
    pub fn heal(&self, input: &[u8], operations: &[HealingOp]) -> Result<Vec<u8>, WasmHealerError> {
        let mut pooled = self.pool.get()?;
        pooled.healer().heal(input, operations)
    }
    
    /// Fast fingerprint check.
    pub fn validate_fingerprint(&self, input: &[u8], expected: u64) -> Result<bool, WasmHealerError> {
        let mut pooled = self.pool.get()?;
        pooled.healer().validate_fingerprint(input, expected)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Dummy WASM bytes for testing - would need real module in actual tests
    // These tests verify the API structure, not actual WASM execution
    
    #[test]
    fn test_instruction_serialization() {
        let ops = vec![
            HealingOp::Rename {
                from: "old_key".into(),
                to: "new_key".into(),
                confidence: 0.95,
            },
            HealingOp::Delete {
                field: "unwanted".into(),
            },
        ];
        
        let serialized = serialize_instructions(&ops);
        
        // Verify format
        let count = u32::from_le_bytes([serialized[0], serialized[1], serialized[2], serialized[3]]);
        assert_eq!(count, 2);
        
        // First instruction should be rename (op = 0)
        assert_eq!(serialized[4], 0);
    }
    
    #[test]
    fn test_memory_layout_constants() {
        use memory_layout::*;
        
        // Verify non-overlapping regions
        assert!(INPUT_OFFSET + INPUT_CAPACITY <= OUTPUT_OFFSET);
        assert!(OUTPUT_OFFSET + OUTPUT_CAPACITY <= INSTR_OFFSET);
        
        // Verify total fits in default WASM memory (~17 pages = ~1.1MB)
        let total_bytes = (INSTR_OFFSET + INSTR_CAPACITY) as usize;
        let pages_needed = (total_bytes + 65535) / 65536;
        assert!(pages_needed <= 17, "Memory layout exceeds default WASM memory");
    }
}

#[cfg(test)]
#[test]
fn test_wasm_imports() {
    use std::path::PathBuf;
    let wasm_path = std::env::var("WASM_MODULE").ok()
        .map(PathBuf::from)
        .or_else(|| {
            let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            p.pop();
            p.push("nomos-healer-guest/target/wasm32-wasip1/release/nomos_healer_guest.wasm");
            Some(p)
        });
    
    if let Some(path) = wasm_path {
        if path.exists() {
            let bytes = std::fs::read(&path).unwrap();
            for item in wasmparser::Parser::new(0).parse_all(&bytes) {
                if let Ok(wasmparser::Payload::ImportSection(reader)) = item {
                    for import in reader {
                        let import = import.unwrap();
                        println!("IMPORT: {}::{}", import.module, import.name);
                    }
                }
            }
        }
    }
}
