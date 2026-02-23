//! Semantic Healing Engine for Nomos.
//!
//! This module implements the autonomous schema healing logic as specified
//! in Section 5 of the architecture document. Key principles:
//!
//! - **No ML Runtime**: Field embeddings use character trigrams (deterministic, fast)
//! - **SIMD-Optimized**: AVX2/SSE2 dot product for cosine similarity
//! - **Sub-50µs Target**: Entire healing path must complete in <50µs
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     SEMANTIC HEALER                             │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌───────────────┐  ┌────────────────┐  ┌───────────────────┐   │
//! │  │ Field Embedder│  │ Similarity     │  │ Confidence Scorer │   │
//! │  │ (trigram hash)│  │ (SIMD cosine)  │  │ (type + pattern)  │   │
//! │  └───────────────┘  └────────────────┘  └───────────────────┘   │
//! │                              │                                  │
//! │                              ▼                                  │
//! │  ┌────────────────────────────────────────────────────────────┐ │
//! │  │              Healing Map Generator                         │ │
//! │  │   { renames: [(old, new)], coercions: [...] }              │ │
//! │  └────────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use tracing::{debug, info, warn};

use crate::schema::{ExpectedSchema, JsonType, SchemaFingerprint};

// ============================================================================
// Constants
// ============================================================================

/// Embedding dimension - 64 i8 values = 64 bytes per field.
/// Chosen for balance of accuracy vs. cache efficiency (fits in L1 cache line).
pub const EMBEDDING_DIM: usize = 64;

/// Minimum confidence threshold to accept a semantic match.
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.70;

// ============================================================================
// SIMD Dot Product - The Heart of the Engine
// ============================================================================
//
// Hand-optimized for x86_64 with AVX2/SSE2 fallback.
// On ARM, uses NEON intrinsics.
// Portable fallback for other architectures.

/// SIMD-optimized dot product for INT8 vectors.
///
/// This is the inner loop of the semantic engine - must be fast.
/// Uses AVX2 for 32-wide parallel multiply-accumulate when available.
///
/// # Performance
/// - AVX2: ~30 cycles for 64 elements
/// - SSE2: ~60 cycles for 64 elements
/// - Scalar: ~200 cycles for 64 elements
#[inline]
pub fn simd_dot_product_i8(a: &[i8; EMBEDDING_DIM], b: &[i8; EMBEDDING_DIM]) -> i32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        simd_dot_product_avx2(a, b)
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        simd_dot_product_sse2(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        simd_dot_product_neon(a, b)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        scalar_dot_product(a, b)
    }
}

/// AVX2 implementation - processes 32 i8 values per instruction.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn simd_dot_product_avx2(a: &[i8; EMBEDDING_DIM], b: &[i8; EMBEDDING_DIM]) -> i32 {
    use std::arch::x86_64::*;

    unsafe {
        // Load first 32 elements
        let a0 = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
        let b0 = _mm256_loadu_si256(b.as_ptr() as *const __m256i);

        // Load second 32 elements
        let a1 = _mm256_loadu_si256(a.as_ptr().add(32) as *const __m256i);
        let b1 = _mm256_loadu_si256(b.as_ptr().add(32) as *const __m256i);

        // Convert i8 to i16 and multiply (extending precision)
        // Split into low/high halves for proper sign extension
        let a0_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a0, 0));
        let a0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a0, 1));
        let b0_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b0, 0));
        let b0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b0, 1));

        let a1_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a1, 0));
        let a1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a1, 1));
        let b1_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b1, 0));
        let b1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b1, 1));

        // Multiply and add pairs (i16 * i16 -> i32, then horizontal add)
        let prod0_lo = _mm256_madd_epi16(a0_lo, b0_lo);
        let prod0_hi = _mm256_madd_epi16(a0_hi, b0_hi);
        let prod1_lo = _mm256_madd_epi16(a1_lo, b1_lo);
        let prod1_hi = _mm256_madd_epi16(a1_hi, b1_hi);

        // Sum all products
        let sum01 = _mm256_add_epi32(prod0_lo, prod0_hi);
        let sum23 = _mm256_add_epi32(prod1_lo, prod1_hi);
        let sum = _mm256_add_epi32(sum01, sum23);

        // Horizontal sum of 8 i32 values
        horizontal_sum_epi32_avx2(sum)
    }
}

/// Horizontal sum of 8 i32 values in AVX2 register.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn horizontal_sum_epi32_avx2(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;

    // Add high 128 bits to low 128 bits
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_extracti128_si256(v, 0);
    let sum128 = _mm_add_epi32(lo, hi);

    // Horizontal add within 128-bit register
    let sum64 = _mm_hadd_epi32(sum128, sum128);
    let sum32 = _mm_hadd_epi32(sum64, sum64);

    _mm_cvtsi128_si32(sum32)
}

/// SSE2 implementation - fallback for older x86_64 CPUs.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[inline]
fn simd_dot_product_sse2(a: &[i8; EMBEDDING_DIM], b: &[i8; EMBEDDING_DIM]) -> i32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut sum = _mm_setzero_si128();

        // Process 16 elements at a time
        for i in (0..EMBEDDING_DIM).step_by(16) {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);

            // Unpack to i16 and multiply
            let va_lo = _mm_cvtepi8_epi16(va);
            let vb_lo = _mm_cvtepi8_epi16(vb);
            let va_hi = _mm_cvtepi8_epi16(_mm_srli_si128(va, 8));
            let vb_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vb, 8));

            // madd: multiply pairs and add adjacent results
            let prod_lo = _mm_madd_epi16(va_lo, vb_lo);
            let prod_hi = _mm_madd_epi16(va_hi, vb_hi);

            sum = _mm_add_epi32(sum, prod_lo);
            sum = _mm_add_epi32(sum, prod_hi);
        }

        // Horizontal sum
        let sum64 = _mm_hadd_epi32(sum, sum);
        let sum32 = _mm_hadd_epi32(sum64, sum64);
        _mm_cvtsi128_si32(sum32)
    }
}

/// NEON implementation for ARM64.
#[cfg(target_arch = "aarch64")]
#[inline]
fn simd_dot_product_neon(a: &[i8; EMBEDDING_DIM], b: &[i8; EMBEDDING_DIM]) -> i32 {
    use std::arch::aarch64::*;

    unsafe {
        let mut sum = vdupq_n_s32(0);

        // Process 16 elements at a time
        for i in (0..EMBEDDING_DIM).step_by(16) {
            let va = vld1q_s8(a.as_ptr().add(i));
            let vb = vld1q_s8(b.as_ptr().add(i));

            // Widen to i16
            let va_lo = vmovl_s8(vget_low_s8(va));
            let va_hi = vmovl_s8(vget_high_s8(va));
            let vb_lo = vmovl_s8(vget_low_s8(vb));
            let vb_hi = vmovl_s8(vget_high_s8(vb));

            // Multiply and accumulate
            sum = vmlal_s16(sum, vget_low_s16(va_lo), vget_low_s16(vb_lo));
            sum = vmlal_s16(sum, vget_high_s16(va_lo), vget_high_s16(vb_lo));
            sum = vmlal_s16(sum, vget_low_s16(va_hi), vget_low_s16(vb_hi));
            sum = vmlal_s16(sum, vget_high_s16(va_hi), vget_high_s16(vb_hi));
        }

        // Horizontal sum
        vaddvq_s32(sum)
    }
}

/// Scalar fallback for non-SIMD platforms (x86/ARM with SIMD disabled).
#[allow(dead_code)]
#[inline]
fn scalar_dot_product(a: &[i8; EMBEDDING_DIM], b: &[i8; EMBEDDING_DIM]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32) * (y as i32))
        .sum()
}

// ============================================================================
// Field Embedding - Trigram-based (No ML Runtime)
// ============================================================================

/// Field embedding for semantic matching.
///
/// Uses character trigrams hashed to a fixed-size vector.
/// This is deterministic, fast, and requires no ML runtime.
///
/// # How it works
/// 1. Extract character trigrams from field name (e.g., "user_id" -> ["use", "ser", "er_", ...])
/// 2. Hash each trigram to a bucket in [0, EMBEDDING_DIM)
/// 3. Increment/decrement the bucket based on position parity
/// 4. Normalize to INT8 range
#[derive(Debug, Clone)]
pub struct FieldEmbedding {
    /// Original field name
    pub field_name: Arc<str>,
    /// Quantized embedding vector (INT8)
    pub embedding: [i8; EMBEDDING_DIM],
    /// Precomputed inverse magnitude for cosine similarity
    pub magnitude_inv: f32,
    /// JSON type hint
    pub json_type: JsonType,
}

impl FieldEmbedding {
    /// Create a new field embedding.
    pub fn new(field_name: &str, json_type: JsonType) -> Self {
        let embedding = compute_trigram_embedding(field_name);
        let magnitude = compute_magnitude(&embedding);
        let magnitude_inv = if magnitude > 0.0 {
            1.0 / magnitude
        } else {
            0.0
        };

        Self {
            field_name: Arc::from(field_name),
            embedding,
            magnitude_inv,
            json_type,
        }
    }

    /// Compute cosine similarity with another embedding.
    ///
    /// Uses SIMD-optimized dot product.
    /// Cost: ~50 cycles on AVX2, ~100 cycles on SSE2.
    #[inline]
    pub fn similarity(&self, other: &FieldEmbedding) -> f32 {
        let dot = simd_dot_product_i8(&self.embedding, &other.embedding);
        dot as f32 * self.magnitude_inv * other.magnitude_inv
    }
}

/// Compute trigram-based embedding for a field name.
///
/// Algorithm:
/// 1. Normalize: lowercase, replace separators with '_'
/// 2. Extract trigrams with padding
/// 3. Hash each trigram to bucket, apply sign based on position
/// 4. Clamp to INT8 range
fn compute_trigram_embedding(field_name: &str) -> [i8; EMBEDDING_DIM] {
    let mut counts = [0i16; EMBEDDING_DIM];

    // Normalize field name: lowercase, standardize separators
    let normalized: String = field_name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();

    // Pad for edge trigrams
    let padded = format!("__{normalized}__");
    let chars: Vec<char> = padded.chars().collect();

    // Extract trigrams and hash to buckets
    for (i, window) in chars.windows(3).enumerate() {
        let trigram: String = window.iter().collect();
        let hash = hash_trigram(&trigram);
        let bucket = (hash as usize) % EMBEDDING_DIM;

        // Alternate sign based on position (reduces collisions)
        let sign = if i % 2 == 0 { 1 } else { -1 };
        counts[bucket] = counts[bucket].saturating_add(sign);
    }

    // Also include bigrams for short field names
    for window in chars.windows(2) {
        let bigram: String = window.iter().collect();
        let hash = hash_trigram(&bigram);
        let bucket = (hash as usize) % EMBEDDING_DIM;
        counts[bucket] = counts[bucket].saturating_add(1);
    }

    // Clamp to INT8 range
    let mut embedding = [0i8; EMBEDDING_DIM];
    for (i, &count) in counts.iter().enumerate() {
        embedding[i] = count.clamp(-127, 127) as i8;
    }

    embedding
}

/// Fast hash for trigrams using FNV-1a variant.
#[inline]
fn hash_trigram(s: &str) -> u32 {
    let mut hash: u32 = 2166136261;
    for byte in s.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash
}

/// Compute vector magnitude.
#[inline]
fn compute_magnitude(embedding: &[i8; EMBEDDING_DIM]) -> f32 {
    let sum_sq: i32 = embedding.iter().map(|&x| (x as i32) * (x as i32)).sum();
    (sum_sq as f32).sqrt()
}

// ============================================================================
// Confidence Scoring
// ============================================================================

/// Match confidence combining multiple signals.
#[derive(Debug, Clone, Copy)]
pub struct MatchConfidence {
    /// Vector similarity score (0.0 - 1.0)
    pub vector_similarity: f32,
    /// Type compatibility score (0.0 - 1.0)
    pub type_compatibility: f32,
    /// Combined confidence (weighted geometric mean)
    pub overall: f32,
}

impl MatchConfidence {
    /// Compute confidence for a field match.
    pub fn compute(old_field: &FieldEmbedding, new_field: &FieldEmbedding) -> Self {
        let vector_similarity = old_field.similarity(new_field).max(0.0).min(1.0);
        let type_compatibility = type_compatibility_score(old_field.json_type, new_field.json_type);

        // Weighted geometric mean: vector has more weight
        // sqrt(vector^0.7 * type^0.3)
        let overall = (vector_similarity.powf(0.7) * type_compatibility.powf(0.3)).sqrt();

        Self {
            vector_similarity,
            type_compatibility,
            overall,
        }
    }

    /// Check if confidence exceeds threshold.
    #[inline]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.overall >= threshold
    }
}

/// Compute type compatibility score.
///
/// Higher score for compatible/coercible types.
pub fn type_compatibility_score(old_type: JsonType, new_type: JsonType) -> f32 {
    match (old_type, new_type) {
        // Exact match
        (a, b) if a == b => 1.0,

        // String <-> Number is common (IDs often change representation)
        (JsonType::String, JsonType::Number) => 0.75,
        (JsonType::Number, JsonType::String) => 0.75,

        // Null compatible with anything (optional fields)
        (JsonType::Null, _) => 0.85,
        (_, JsonType::Null) => 0.85,

        // Bool <-> Number (0/1 encoding)
        (JsonType::Bool, JsonType::Number) => 0.6,
        (JsonType::Number, JsonType::Bool) => 0.6,

        // Arrays and Objects are incompatible with primitives
        (JsonType::Array, JsonType::Object) => 0.3,
        (JsonType::Object, JsonType::Array) => 0.3,

        // Everything else is low compatibility
        _ => 0.1,
    }
}

// ============================================================================
// Locality-Sensitive Hashing (LSH) Index
// ============================================================================

/// Number of hyperplanes for LSH hashing.
const LSH_NUM_HYPERPLANES: usize = 16;

/// Threshold field count above which LSH is used.
pub const LSH_THRESHOLD: usize = 100;

/// Locality-Sensitive Hashing index for O(1) candidate filtering.
///
/// For schemas with >100 fields, brute-force O(n) similarity comparison
/// becomes expensive. LSH provides approximate nearest neighbor lookup
/// in O(1) by hashing similar embeddings to the same bucket.
///
/// # How it works
/// 1. Generate 16 random hyperplanes during initialization
/// 2. For each embedding, compute which side of each hyperplane it falls on
/// 3. This gives a 16-bit hash (LSH signature)
/// 4. Similar embeddings have similar hashes (Hamming distance)
/// 5. Lookup: find candidates in same bucket, then verify with exact similarity
#[derive(Clone)]
pub struct LshIndex {
    /// Random hyperplanes for hashing (16 hyperplanes × 64 dimensions)
    hyperplanes: Box<[[f32; EMBEDDING_DIM]; LSH_NUM_HYPERPLANES]>,
    
    /// Hash buckets: LSH hash -> list of field indices
    buckets: HashMap<u16, Vec<usize>>,
    
    /// All indexed embeddings (for exact similarity verification)
    embeddings: Vec<Arc<FieldEmbedding>>,
    
    /// Number of fields indexed
    field_count: usize,
}

impl Default for LshIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl LshIndex {
    /// Create a new LSH index with random hyperplanes.
    pub fn new() -> Self {
        // Generate deterministic "random" hyperplanes using a simple LCG
        // This ensures consistent behavior across restarts
        let mut seed: u64 = 0xDEADBEEF_CAFEBABE;
        let mut hyperplanes = Box::new([[0.0f32; EMBEDDING_DIM]; LSH_NUM_HYPERPLANES]);
        
        for plane in hyperplanes.iter_mut() {
            for val in plane.iter_mut() {
                // Simple LCG: next = (a * seed + c) mod m
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                // Convert to [-1, 1] range
                *val = ((seed >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            }
            // Normalize hyperplane
            let mag: f32 = plane.iter().map(|x| x * x).sum::<f32>().sqrt();
            if mag > 0.0 {
                for val in plane.iter_mut() {
                    *val /= mag;
                }
            }
        }
        
        Self {
            hyperplanes,
            buckets: HashMap::new(),
            embeddings: Vec::new(),
            field_count: 0,
        }
    }
    
    /// Index a list of field embeddings.
    pub fn index_embeddings(&mut self, embeddings: &[FieldEmbedding]) {
        self.buckets.clear();
        self.embeddings.clear();
        self.field_count = embeddings.len();
        
        for (idx, emb) in embeddings.iter().enumerate() {
            let hash = self.compute_hash(&emb.embedding);
            self.buckets.entry(hash).or_default().push(idx);
            self.embeddings.push(Arc::new(emb.clone()));
        }
    }
    
    /// Compute LSH hash for an embedding.
    #[inline]
    fn compute_hash(&self, embedding: &[i8; EMBEDDING_DIM]) -> u16 {
        let mut hash = 0u16;
        
        for (i, hyperplane) in self.hyperplanes.iter().enumerate() {
            // Compute dot product with hyperplane
            let dot: f32 = embedding.iter()
                .zip(hyperplane.iter())
                .map(|(&e, &h)| e as f32 * h)
                .sum();
            
            // Set bit if on positive side of hyperplane
            if dot > 0.0 {
                hash |= 1 << i;
            }
        }
        
        hash
    }
    
    /// Find candidate matches for a query embedding.
    ///
    /// Returns field indices that are likely similar (same LSH bucket).
    /// Also includes neighbors with Hamming distance 1 for better recall.
    pub fn find_candidates(&self, query: &[i8; EMBEDDING_DIM]) -> Vec<usize> {
        let query_hash = self.compute_hash(query);
        let mut candidates = Vec::new();
        
        // Exact bucket match
        if let Some(indices) = self.buckets.get(&query_hash) {
            candidates.extend(indices.iter().copied());
        }
        
        // Also check buckets with Hamming distance 1 (flip one bit)
        for bit in 0..LSH_NUM_HYPERPLANES {
            let neighbor_hash = query_hash ^ (1 << bit);
            if let Some(indices) = self.buckets.get(&neighbor_hash) {
                for &idx in indices {
                    if !candidates.contains(&idx) {
                        candidates.push(idx);
                    }
                }
            }
        }
        
        candidates
    }
    
    /// Get embedding by index.
    #[inline]
    pub fn get_embedding(&self, idx: usize) -> Option<&Arc<FieldEmbedding>> {
        self.embeddings.get(idx)
    }
    
    /// Check if LSH should be used based on field count.
    #[inline]
    pub fn should_use_lsh(&self) -> bool {
        self.field_count >= LSH_THRESHOLD
    }
    
    /// Get number of indexed fields.
    #[inline]
    pub fn field_count(&self) -> usize {
        self.field_count
    }
}

// ============================================================================
// Healing Map
// ============================================================================

/// A single healing operation.
#[derive(Debug, Clone)]
pub enum HealingOp {
    /// Rename a field
    Rename {
        from: Arc<str>,
        to: Arc<str>,
        confidence: f32,
    },
    /// Coerce type (e.g., String -> Number)
    CoerceType {
        field: Arc<str>,
        from_type: JsonType,
        to_type: JsonType,
    },
    /// Set default value for missing field
    SetDefault {
        field: Arc<str>,
        default_value: Arc<str>,
    },
    /// Delete unexpected field
    Delete { field: Arc<str> },
}

impl std::fmt::Display for HealingOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealingOp::Rename { from, to, confidence } => {
                write!(f, "RENAME: {} -> {} | Confidence: {:.2}", from, to, confidence)
            }
            HealingOp::CoerceType {
                field,
                from_type,
                to_type,
            } => {
                write!(f, "COERCE: {} ({:?} -> {:?})", field, from_type, to_type)
            }
            HealingOp::SetDefault { field, default_value } => {
                write!(f, "DEFAULT: {} = {}", field, default_value)
            }
            HealingOp::Delete { field } => {
                write!(f, "DELETE: {}", field)
            }
        }
    }
}

/// Healing map containing all transformations for a schema drift.
#[derive(Debug, Clone, Default)]
pub struct HealingMap {
    /// List of healing operations to apply
    pub operations: Vec<HealingOp>,
    /// Overall confidence in the healing map
    pub confidence: f32,
    /// Source schema fingerprint
    pub source_fingerprint: SchemaFingerprint,
    /// Target schema fingerprint
    pub target_fingerprint: SchemaFingerprint,
}

// ============================================================================
// Semantic Healer
// ============================================================================

/// The Semantic Healer - autonomous schema reconstruction engine.
///
/// # Design Principles
/// - No ML runtime - uses trigram-based embeddings
/// - SIMD-optimized similarity computation
/// - LSH indexing for O(1) lookups on large schemas (>100 fields)
/// - Sub-50µs healing path target
///
/// # Usage
/// ```ignore
/// let healer = SemanticHealer::new();
/// healer.register_expected_schema(route, schema);
///
/// // On drift detection:
/// let healing_map = healer.compute_healing_map(expected, actual)?;
/// ```
pub struct SemanticHealer {
    /// Confidence threshold for accepting matches
    confidence_threshold: f32,
    /// Pre-computed embeddings for expected fields (keyed by schema fingerprint)
    schema_embeddings: HashMap<SchemaFingerprint, Vec<FieldEmbedding>>,
    /// LSH indices for large schemas (keyed by schema fingerprint)
    lsh_indices: HashMap<SchemaFingerprint, LshIndex>,
}

impl Default for SemanticHealer {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticHealer {
    /// Create a new Semantic Healer.
    pub fn new() -> Self {
        Self {
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            schema_embeddings: HashMap::new(),
            lsh_indices: HashMap::new(),
        }
    }

    /// Create with custom confidence threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            confidence_threshold: threshold,
            schema_embeddings: HashMap::new(),
            lsh_indices: HashMap::new(),
        }
    }

    /// Register an expected schema's field embeddings.
    ///
    /// Call this once per schema during startup or schema update.
    /// Embeddings are computed and cached for fast matching.
    /// For schemas with >100 fields, an LSH index is built for O(1) lookups.
    pub fn register_schema(&mut self, schema: &ExpectedSchema) {
        let embeddings: Vec<FieldEmbedding> = schema
            .root_fields
            .iter()
            .map(|field| {
                let json_type = schema
                    .field_types
                    .get(field)
                    .copied()
                    .unwrap_or(JsonType::String);
                FieldEmbedding::new(field, json_type)
            })
            .collect();

        // Build LSH index for large schemas
        if embeddings.len() >= LSH_THRESHOLD {
            let mut lsh = LshIndex::new();
            lsh.index_embeddings(&embeddings);
            info!(
                fingerprint = ?schema.fingerprint,
                field_count = embeddings.len(),
                "Built LSH index for large schema"
            );
            self.lsh_indices.insert(schema.fingerprint, lsh);
        }

        debug!(
            fingerprint = ?schema.fingerprint,
            field_count = embeddings.len(),
            use_lsh = embeddings.len() >= LSH_THRESHOLD,
            "Registered schema embeddings"
        );

        self.schema_embeddings.insert(schema.fingerprint, embeddings);
    }

    /// Compute a healing map for a schema drift.
    ///
    /// # Arguments
    /// - `expected`: The expected schema
    /// - `actual_fields`: Fields found in the actual response
    /// - `actual_types`: Types of actual fields
    ///
    /// # Returns
    /// A HealingMap if matches are found, or None if drift is too severe.
    ///
    /// # Performance
    /// Target: <50µs for typical drift (2-3 field renames)
    pub fn compute_healing_map(
        &self,
        expected: &ExpectedSchema,
        actual_fields: &[&str],
        actual_types: &HashMap<&str, JsonType>,
    ) -> Option<HealingMap> {
        let expected_embeddings = self.schema_embeddings.get(&expected.fingerprint)?;
        let lsh_index = self.lsh_indices.get(&expected.fingerprint);

        // Build set of expected field names for quick lookup
        let expected_names: std::collections::HashSet<&str> = expected
            .root_fields
            .iter()
            .map(|s| s.as_ref())
            .collect();

        // Find fields that need mapping
        let mut operations = Vec::new();
        let mut total_confidence = 0.0;
        let mut match_count = 0;

        for &actual_field in actual_fields {
            // Skip if field exists in expected schema
            if expected_names.contains(actual_field) {
                continue;
            }

            // Create embedding for actual field
            let actual_type = actual_types
                .get(actual_field)
                .copied()
                .unwrap_or(JsonType::String);
            let actual_embedding = FieldEmbedding::new(actual_field, actual_type);

            // Find best match among expected fields
            // Use LSH for O(1) lookup on large schemas, O(n) linear scan otherwise
            let best_match = if let Some(lsh) = lsh_index {
                self.find_best_match_lsh(
                    &actual_embedding,
                    lsh,
                    expected_embeddings,
                    actual_fields,
                )
            } else {
                self.find_best_match_linear(
                    &actual_embedding,
                    expected_embeddings,
                    actual_fields,
                )
            };

            if let Some((matched_field, confidence)) = best_match {
                // Log the healing operation
                info!(
                    "HEAL: {} -> {} | Confidence: {:.2}",
                    actual_field, matched_field.field_name, confidence.overall
                );

                operations.push(HealingOp::Rename {
                    from: Arc::from(actual_field),
                    to: Arc::clone(&matched_field.field_name),
                    confidence: confidence.overall,
                });

                // Check if type coercion is needed
                if actual_type != matched_field.json_type && matched_field.json_type != JsonType::Null
                {
                    operations.push(HealingOp::CoerceType {
                        field: Arc::clone(&matched_field.field_name),
                        from_type: actual_type,
                        to_type: matched_field.json_type,
                    });
                }

                total_confidence += confidence.overall;
                match_count += 1;
            } else {
                // No confident match found - mark as potential deletion
                warn!(
                    field = actual_field,
                    "No confident match found for field, may need deletion"
                );
            }
        }

        // Check for missing expected fields (might need defaults)
        for expected_emb in expected_embeddings {
            if !actual_fields.contains(&expected_emb.field_name.as_ref())
                && !operations
                    .iter()
                    .any(|op| matches!(op, HealingOp::Rename { to, .. } if to == &expected_emb.field_name))
            {
                warn!(
                    field = expected_emb.field_name.as_ref(),
                    "Expected field missing from response"
                );
            }
        }

        // Return healing map if we have operations
        if operations.is_empty() {
            None
        } else {
            let overall_confidence = if match_count > 0 {
                total_confidence / match_count as f32
            } else {
                0.0
            };

            Some(HealingMap {
                operations,
                confidence: overall_confidence,
                source_fingerprint: SchemaFingerprint(0), // Actual fingerprint of received JSON
                target_fingerprint: expected.fingerprint,
            })
        }
    }

    /// Find best match using LSH index (O(1) candidate lookup).
    fn find_best_match_lsh<'a>(
        &self,
        actual: &FieldEmbedding,
        lsh: &LshIndex,
        expected_embeddings: &'a [FieldEmbedding],
        actual_fields: &[&str],
    ) -> Option<(&'a FieldEmbedding, MatchConfidence)> {
        // Get candidates from LSH bucket
        let candidates = lsh.find_candidates(&actual.embedding);
        
        let mut best: Option<(&'a FieldEmbedding, MatchConfidence)> = None;
        
        for idx in candidates {
            if idx >= expected_embeddings.len() {
                continue;
            }
            let expected_emb = &expected_embeddings[idx];
            
            // Skip if expected field is already present in actual
            if actual_fields.contains(&expected_emb.field_name.as_ref()) {
                continue;
            }
            
            let confidence = MatchConfidence::compute(expected_emb, actual);
            
            if confidence.is_confident(self.confidence_threshold) {
                if best.is_none() || confidence.overall > best.as_ref().unwrap().1.overall {
                    best = Some((expected_emb, confidence));
                }
            }
        }
        
        best
    }
    
    /// Find best match using linear scan (O(n)).
    fn find_best_match_linear<'a>(
        &self,
        actual: &FieldEmbedding,
        expected_embeddings: &'a [FieldEmbedding],
        actual_fields: &[&str],
    ) -> Option<(&'a FieldEmbedding, MatchConfidence)> {
        let mut best: Option<(&'a FieldEmbedding, MatchConfidence)> = None;
        
        for expected_emb in expected_embeddings {
            // Skip if expected field is already present in actual
            if actual_fields.contains(&expected_emb.field_name.as_ref()) {
                continue;
            }
            
            let confidence = MatchConfidence::compute(expected_emb, actual);
            
            if confidence.is_confident(self.confidence_threshold) {
                if best.is_none() || confidence.overall > best.as_ref().unwrap().1.overall {
                    best = Some((expected_emb, confidence));
                }
            }
        }
        
        best
    }

    /// Apply healing operations to a JSON buffer.
    ///
    /// This is a simple implementation that works on the raw JSON text.
    /// For production, this would use the WASM healer for isolation.
    ///
    /// # Arguments
    /// - `input`: Raw JSON bytes
    /// - `healing_map`: Operations to apply
    /// - `arena`: Bump allocator for scratch space
    ///
    /// # Returns
    /// Healed JSON bytes (allocated from arena)
    pub fn apply_healing<'a>(
        &self,
        input: &'a [u8],
        healing_map: &HealingMap,
        arena: &'a bumpalo::Bump,
    ) -> &'a [u8] {
        // Simple string-based healing (production would use WASM)
        let mut json_str = match std::str::from_utf8(input) {
            Ok(s) => s.to_string(),
            Err(_) => return arena.alloc_slice_copy(input), // Pass through on decode error
        };

        for op in &healing_map.operations {
            match op {
                HealingOp::Rename { from, to, confidence: _ } => {
                    // Replace "from": with "to":
                    let from_key = format!("\"{}\":", from);
                    let to_key = format!("\"{}\":", to);
                    json_str = json_str.replace(&from_key, &to_key);

                    debug!(from = from.as_ref(), to = to.as_ref(), "Applied field rename");
                }
                HealingOp::CoerceType {
                    field,
                    from_type,
                    to_type,
                } => {
                    debug!(
                        field = field.as_ref(),
                        from = ?from_type,
                        to = ?to_type,
                        "Type coercion noted (implementation pending)"
                    );
                    // Type coercion would be implemented here
                }
                HealingOp::SetDefault { field, default_value } => {
                    debug!(
                        field = field.as_ref(),
                        default = default_value.as_ref(),
                        "Default value set (implementation pending)"
                    );
                }
                HealingOp::Delete { field } => {
                    debug!(field = field.as_ref(), "Field deletion (implementation pending)");
                }
            }
        }

        // Allocate output in arena
        let bytes = json_str.into_bytes();
        let output = arena.alloc_slice_copy(&bytes);
        output
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigram_embedding() {
        let emb1 = FieldEmbedding::new("user_id", JsonType::String);
        let emb2 = FieldEmbedding::new("userId", JsonType::String);
        let emb3 = FieldEmbedding::new("uuid", JsonType::String);
        let emb4 = FieldEmbedding::new("timestamp", JsonType::Number);

        // Similar names should have higher similarity
        let sim_12 = emb1.similarity(&emb2);
        let sim_13 = emb1.similarity(&emb3);
        let sim_14 = emb1.similarity(&emb4);

        println!("user_id vs userId: {:.3}", sim_12);
        println!("user_id vs uuid: {:.3}", sim_13);
        println!("user_id vs timestamp: {:.3}", sim_14);

        // user_id and userId should be more similar than user_id and timestamp
        assert!(sim_12 > sim_14, "user_id should match userId better than timestamp");
    }

    #[test]
    fn test_simd_dot_product() {
        let a: [i8; EMBEDDING_DIM] = [1; EMBEDDING_DIM];
        let b: [i8; EMBEDDING_DIM] = [2; EMBEDDING_DIM];

        let result = simd_dot_product_i8(&a, &b);
        assert_eq!(result, 128); // 64 * 1 * 2 = 128
    }

    #[test]
    fn test_confidence_scoring() {
        let old = FieldEmbedding::new("user_id", JsonType::String);
        let new_similar = FieldEmbedding::new("userId", JsonType::String);
        let new_different_type = FieldEmbedding::new("userId", JsonType::Number);

        let conf1 = MatchConfidence::compute(&old, &new_similar);
        let conf2 = MatchConfidence::compute(&old, &new_different_type);

        println!("Same type confidence: {:.3}", conf1.overall);
        println!("Different type confidence: {:.3}", conf2.overall);

        // Same types should have higher confidence
        assert!(conf1.overall >= conf2.overall);
    }

    #[test]
    fn test_type_compatibility() {
        assert_eq!(type_compatibility_score(JsonType::String, JsonType::String), 1.0);
        assert!(type_compatibility_score(JsonType::String, JsonType::Number) > 0.5);
        assert!(type_compatibility_score(JsonType::Array, JsonType::String) < 0.5);
    }

    #[test]
    fn test_semantic_healer() {
        let mut healer = SemanticHealer::new();

        // Create expected schema
        let expected = ExpectedSchema {
            fingerprint: SchemaFingerprint(12345),
            root_fields: vec![Arc::from("user_id"), Arc::from("full_name"), Arc::from("email")],
            field_types: [
                (Arc::from("user_id"), JsonType::String),
                (Arc::from("full_name"), JsonType::String),
                (Arc::from("email"), JsonType::String),
            ]
            .into_iter()
            .collect(),
            version: 1,
            last_seen_timestamp: 0,
        };

        healer.register_schema(&expected);

        // Simulate drift: API now returns uuid and name instead
        let actual_fields = ["uuid", "name", "email"];
        let actual_types: HashMap<&str, JsonType> = [
            ("uuid", JsonType::String),
            ("name", JsonType::String),
            ("email", JsonType::String),
        ]
        .into_iter()
        .collect();

        let healing_map = healer.compute_healing_map(&expected, &actual_fields, &actual_types);

        assert!(healing_map.is_some());
        let map = healing_map.unwrap();
        assert!(!map.operations.is_empty());

        // Print healing operations
        for op in &map.operations {
            println!("{}", op);
        }
    }

    #[test]
    fn test_healing_application() {
        let healer = SemanticHealer::new();
        let arena = bumpalo::Bump::new();

        let input = br#"{"uuid": "123", "name": "Alice"}"#;
        let healing_map = HealingMap {
            operations: vec![
                HealingOp::Rename {
                    from: Arc::from("uuid"),
                    to: Arc::from("user_id"),
                    confidence: 0.95,
                },
                HealingOp::Rename {
                    from: Arc::from("name"),
                    to: Arc::from("full_name"),
                    confidence: 0.92,
                },
            ],
            confidence: 0.935,
            source_fingerprint: SchemaFingerprint(0),
            target_fingerprint: SchemaFingerprint(12345),
        };

        let output = healer.apply_healing(input, &healing_map, &arena);
        let output_str = std::str::from_utf8(output).unwrap();

        assert!(output_str.contains("user_id"));
        assert!(output_str.contains("full_name"));
        assert!(!output_str.contains("uuid"));
        assert!(!output_str.contains("\"name\""));
    }
}
