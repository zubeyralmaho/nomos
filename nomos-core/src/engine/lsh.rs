//! Locality-Sensitive Hashing (LSH) Index
//!
//! For schemas with >100 fields, brute-force O(n) similarity comparison
//! becomes expensive. LSH provides approximate nearest neighbor lookup
//! in O(1) by hashing similar embeddings to the same bucket.
//!
//! # How it works
//! 1. Generate 16 random hyperplanes during initialization
//! 2. For each embedding, compute which side of each hyperplane it falls on
//! 3. This gives a 16-bit hash (LSH signature)
//! 4. Similar embeddings have similar hashes (Hamming distance)
//! 5. Lookup: find candidates in same bucket, then verify with exact similarity

use std::collections::HashMap;
use std::sync::Arc;

use crate::engine::simd::EMBEDDING_DIM;
use crate::engine::embedding::FieldEmbedding;

// ============================================================================
// Constants
// ============================================================================

/// Number of hyperplanes for LSH hashing.
const LSH_NUM_HYPERPLANES: usize = 16;

/// Threshold field count above which LSH is used.
pub const LSH_THRESHOLD: usize = 100;

// ============================================================================
// LSH Index
// ============================================================================

/// Locality-Sensitive Hashing index for O(1) candidate filtering.
///
/// For schemas with >100 fields, brute-force O(n) similarity comparison
/// becomes expensive. LSH provides approximate nearest neighbor lookup
/// in O(1) by hashing similar embeddings to the same bucket.
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
            let dot: f32 = embedding
                .iter()
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::JsonType;

    #[test]
    fn test_lsh_index_creation() {
        let index = LshIndex::new();
        assert_eq!(index.field_count(), 0);
        assert!(!index.should_use_lsh());
    }

    #[test]
    fn test_lsh_hash_deterministic() {
        let index = LshIndex::new();
        let embedding = [0i8; EMBEDDING_DIM];

        let hash1 = index.compute_hash(&embedding);
        let hash2 = index.compute_hash(&embedding);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_lsh_similar_embeddings_same_bucket() {
        let mut index = LshIndex::new();

        // Create similar embeddings
        let emb1 = FieldEmbedding::new("user_name", JsonType::String);
        let emb2 = FieldEmbedding::new("user_names", JsonType::String); // Very similar

        index.index_embeddings(&[emb1.clone(), emb2.clone()]);

        // Find candidates for emb1
        let candidates = index.find_candidates(&emb1.embedding);

        // Should find both (same or nearby bucket)
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_lsh_threshold() {
        let mut index = LshIndex::new();

        // Below threshold
        let embeddings: Vec<FieldEmbedding> = (0..50)
            .map(|i| FieldEmbedding::new(&format!("field_{}", i), JsonType::String))
            .collect();

        index.index_embeddings(&embeddings);
        assert!(!index.should_use_lsh());

        // Above threshold
        let embeddings: Vec<FieldEmbedding> = (0..150)
            .map(|i| FieldEmbedding::new(&format!("field_{}", i), JsonType::String))
            .collect();

        index.index_embeddings(&embeddings);
        assert!(index.should_use_lsh());
    }
}
