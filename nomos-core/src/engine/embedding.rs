//! Field Embedding - Trigram-based (No ML Runtime)
//!
//! Uses character trigrams hashed to a fixed-size vector.
//! This is deterministic, fast, and requires no ML runtime.
//!
//! # How it works
//! 1. Extract character trigrams from field name (e.g., "user_id" -> ["use", "ser", "er_", ...])
//! 2. Hash each trigram to a bucket in [0, EMBEDDING_DIM)
//! 3. Increment/decrement the bucket based on position parity
//! 4. Normalize to INT8 range

use std::sync::Arc;
use crate::schema::JsonType;
use super::simd::{EMBEDDING_DIM, simd_dot_product_i8};

/// Field embedding for semantic matching.
///
/// Uses character trigrams hashed to a fixed-size vector.
/// This is deterministic, fast, and requires no ML runtime.
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
/// # Algorithm
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigram_embedding() {
        let emb1 = FieldEmbedding::new("user_id", JsonType::String);
        let emb2 = FieldEmbedding::new("userId", JsonType::String);
        let emb3 = FieldEmbedding::new("timestamp", JsonType::Number);

        // Similar names should have higher similarity
        let sim_12 = emb1.similarity(&emb2);
        let sim_13 = emb1.similarity(&emb3);

        assert!(sim_12 > sim_13, "user_id should match userId better than timestamp");
    }

    #[test]
    fn test_embedding_magnitude() {
        let emb = FieldEmbedding::new("test", JsonType::String);
        assert!(emb.magnitude_inv > 0.0);
    }
}
