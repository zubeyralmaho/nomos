//! Lock-free Schema Store using ArcSwap.
//!
//! As specified in Section 4 of the architecture document, the Schema Store
//! uses ArcSwap for lock-free reads. This is critical because:
//!
//! - Reads happen on every request (hot path)
//! - Writes are rare (schema updates)
//! - We cannot afford Mutex contention under load

use arc_swap::{ArcSwap, Guard};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use xxhash_rust::xxh64::xxh64;

/// Route key for schema lookup.
///
/// Routes are identified by method + path pattern.
/// Using a compact representation to minimize memory.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RouteKey {
    /// HTTP method as a compact enum
    pub method: HttpMethod,
    /// Path pattern (e.g., "/api/v1/users")
    pub path: Arc<str>,
}

/// Compact HTTP method representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HttpMethod {
    Get = 0,
    Post = 1,
    Put = 2,
    Delete = 3,
    Patch = 4,
    Head = 5,
    Options = 6,
    Other = 255,
}

impl From<&http::Method> for HttpMethod {
    #[inline]
    fn from(method: &http::Method) -> Self {
        match *method {
            http::Method::GET => HttpMethod::Get,
            http::Method::POST => HttpMethod::Post,
            http::Method::PUT => HttpMethod::Put,
            http::Method::DELETE => HttpMethod::Delete,
            http::Method::PATCH => HttpMethod::Patch,
            http::Method::HEAD => HttpMethod::Head,
            http::Method::OPTIONS => HttpMethod::Options,
            _ => HttpMethod::Other,
        }
    }
}

/// Schema fingerprint for quick equality checks.
///
/// Computed as XxHash64 of the JSON key structure.
/// This allows O(1) drift detection before full validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SchemaFingerprint(pub u64);

impl SchemaFingerprint {
    /// Compute fingerprint from a list of field names.
    pub fn from_fields(fields: &[&str]) -> Self {
        let mut sorted = fields.to_vec();
        sorted.sort_unstable();
        let concatenated = sorted.join("\0");
        Self(xxh64(concatenated.as_bytes(), 0))
    }

    /// Compute fingerprint from raw JSON bytes (keys only).
    ///
    /// This is a fast path using simd-json tape iteration.
    pub fn from_json_keys(json: &[u8]) -> Option<Self> {
        // For now, use a simple byte-based hash
        // In production, we'd iterate the simd-json tape
        Some(Self(xxh64(json, 0)))
    }
}

/// Expected schema for a route.
///
/// Contains the structural information needed to detect drift
/// and compute healing maps.
#[derive(Debug, Clone)]
pub struct ExpectedSchema {
    /// Unique fingerprint of this schema
    pub fingerprint: SchemaFingerprint,
    /// Field names at the root level
    pub root_fields: Vec<Arc<str>>,
    /// Field type hints (for type coercion healing)
    pub field_types: HashMap<Arc<str>, JsonType>,
    /// Schema version (for tracking changes)
    pub version: u64,
    /// When this schema was last seen
    pub last_seen_timestamp: u64,
}

/// JSON type representation for schema fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum JsonType {
    Null = 0,
    Bool = 1,
    Number = 2,
    String = 3,
    Array = 4,
    Object = 5,
}

/// Internal state of the Schema Store.
///
/// This struct is immutable once created. Updates create a new
/// instance and atomically swap the pointer.
#[derive(Debug, Default)]
pub struct SchemaStoreInner {
    /// Route -> Expected Schema mapping
    schemas: HashMap<RouteKey, Arc<ExpectedSchema>>,
    /// Fingerprint -> Expected Schema (for reverse lookup)
    by_fingerprint: HashMap<SchemaFingerprint, Arc<ExpectedSchema>>,
    /// Monotonic version counter
    version: u64,
}

impl Clone for SchemaStoreInner {
    fn clone(&self) -> Self {
        Self {
            schemas: self.schemas.clone(),
            by_fingerprint: self.by_fingerprint.clone(),
            version: self.version,
        }
    }
}

/// Lock-free Schema Store.
///
/// Provides O(1) read access to schema definitions without locking.
/// Updates use copy-on-write semantics via ArcSwap.
///
/// # Thread Safety
///
/// - Reads: Lock-free, wait-free
/// - Writes: Atomic swap (brief contention on swap)
///
/// # Memory Model
///
/// Readers always see a consistent snapshot. During updates,
/// old versions are kept alive until all readers release them.
pub struct SchemaStore {
    inner: ArcSwap<SchemaStoreInner>,
}

impl SchemaStore {
    /// Create a new empty Schema Store.
    pub fn new() -> Self {
        Self {
            inner: ArcSwap::from_pointee(SchemaStoreInner::default()),
        }
    }

    /// Get the schema for a route.
    ///
    /// This is the hot path - must be lock-free.
    ///
    /// # Performance
    ///
    /// - Time: O(1) hash lookup
    /// - Allocation: Zero (returns Arc reference)
    /// - Contention: None (atomic load only)
    #[inline]
    pub fn get(&self, route: &RouteKey) -> Option<Arc<ExpectedSchema>> {
        let guard = self.inner.load();
        guard.schemas.get(route).map(Arc::clone)
    }

    /// Get schema by fingerprint.
    ///
    /// Used during healing to find the expected schema
    /// for a drifted response.
    #[inline]
    pub fn get_by_fingerprint(&self, fingerprint: SchemaFingerprint) -> Option<Arc<ExpectedSchema>> {
        let guard = self.inner.load();
        guard.by_fingerprint.get(&fingerprint).map(Arc::clone)
    }

    /// Check if a route has a registered schema.
    #[inline]
    pub fn has_schema(&self, route: &RouteKey) -> bool {
        let guard = self.inner.load();
        guard.schemas.contains_key(route)
    }

    /// Get the current store version.
    #[inline]
    pub fn version(&self) -> u64 {
        self.inner.load().version
    }

    /// Get a snapshot guard for extended read operations.
    ///
    /// The guard keeps the current version alive for the
    /// duration of the borrow.
    #[inline]
    pub fn snapshot(&self) -> Guard<Arc<SchemaStoreInner>> {
        self.inner.load()
    }

    /// Register or update a schema for a route.
    ///
    /// This is the cold path - can tolerate brief allocation.
    ///
    /// # Copy-on-Write
    ///
    /// Creates a new inner state with the updated schema,
    /// then atomically swaps the pointer.
    pub fn register(&self, route: RouteKey, schema: ExpectedSchema) {
        let schema = Arc::new(schema);
        let mut new_inner = (*self.inner.load_full()).clone();

        // Update fingerprint index
        new_inner
            .by_fingerprint
            .insert(schema.fingerprint, Arc::clone(&schema));

        // Update route mapping
        new_inner.schemas.insert(route, schema);

        // Increment version
        new_inner.version += 1;

        // Atomic swap
        self.inner.store(Arc::new(new_inner));
    }

    /// Remove a schema from the store.
    pub fn remove(&self, route: &RouteKey) -> Option<Arc<ExpectedSchema>> {
        let old = {
            let guard = self.inner.load();
            guard.schemas.get(route).map(Arc::clone)
        };

        if let Some(ref schema) = old {
            let mut new_inner = (*self.inner.load_full()).clone();
            new_inner.schemas.remove(route);
            new_inner.by_fingerprint.remove(&schema.fingerprint);
            new_inner.version += 1;
            self.inner.store(Arc::new(new_inner));
        }

        old
    }

    /// Get the number of registered schemas.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.load().schemas.len()
    }

    /// Check if the store is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.load().schemas.is_empty()
    }
}

impl Default for SchemaStore {
    fn default() -> Self {
        Self::new()
    }
}

// SchemaStore is Send + Sync because ArcSwap provides thread-safe access
unsafe impl Send for SchemaStore {}
unsafe impl Sync for SchemaStore {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_route(method: HttpMethod, path: &str) -> RouteKey {
        RouteKey {
            method,
            path: Arc::from(path),
        }
    }

    fn make_schema(fields: &[&str]) -> ExpectedSchema {
        let fingerprint = SchemaFingerprint::from_fields(fields);
        ExpectedSchema {
            fingerprint,
            root_fields: fields.iter().map(|s| Arc::from(*s)).collect(),
            field_types: HashMap::new(),
            version: 1,
            last_seen_timestamp: 0,
        }
    }

    #[test]
    fn test_empty_store() {
        let store = SchemaStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.version(), 0);
    }

    #[test]
    fn test_register_and_get() {
        let store = SchemaStore::new();
        let route = make_route(HttpMethod::Get, "/api/users");
        let schema = make_schema(&["id", "name", "email"]);

        store.register(route.clone(), schema);

        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
        assert_eq!(store.version(), 1);

        let retrieved = store.get(&route).expect("schema should exist");
        assert_eq!(retrieved.root_fields.len(), 3);
    }

    #[test]
    fn test_fingerprint_lookup() {
        let store = SchemaStore::new();
        let route = make_route(HttpMethod::Get, "/api/users");
        let schema = make_schema(&["id", "name"]);
        let fingerprint = schema.fingerprint;

        store.register(route, schema);

        let by_fp = store
            .get_by_fingerprint(fingerprint)
            .expect("should find by fingerprint");
        assert_eq!(by_fp.fingerprint, fingerprint);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let store = StdArc::new(SchemaStore::new());

        // Register some schemas
        for i in 0..10 {
            let route = make_route(HttpMethod::Get, &format!("/api/resource/{}", i));
            let schema = make_schema(&["id", "data"]);
            store.register(route, schema);
        }

        // Spawn readers
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let store = StdArc::clone(&store);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        let route = make_route(HttpMethod::Get, "/api/resource/5");
                        let _ = store.get(&route);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
