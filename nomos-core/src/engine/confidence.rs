//! Confidence Scoring Module
//!
//! Match confidence combining multiple signals:
//! - Vector similarity from trigram embeddings
//! - Type compatibility for JSON types
//! - Full NLP ensemble scoring

use crate::nlp::combined_nlp_similarity;
use crate::engine::embedding::FieldEmbedding;
use crate::schema::JsonType;

// ============================================================================
// Match Confidence
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
    ///
    /// Uses NLP ENSEMBLE:
    /// 1. Levenshtein Distance - edit distance for typo detection
    /// 2. Jaro-Winkler Similarity - prefix-aware string matching
    /// 3. N-gram TF-IDF - structural similarity via trigrams
    /// 4. Synonym Dictionary - semantic equivalence lookup
    /// 5. Type Compatibility - JSON type coercion feasibility
    pub fn compute(old_field: &FieldEmbedding, new_field: &FieldEmbedding) -> Self {
        // FULL NLP ENSEMBLE (Levenshtein + Jaro-Winkler + TF-IDF + Synonym)
        let nlp_score = combined_nlp_similarity(&old_field.field_name, &new_field.field_name);

        // Trigram Vector Similarity (fast byte-level, backup)
        let vector_similarity = old_field.similarity(new_field).max(0.0).min(1.0);

        // Type Compatibility
        let type_compatibility = type_compatibility_score(old_field.json_type, new_field.json_type);

        // Ensemble: NLP has higher weight, vector as backup
        let semantic_score = nlp_score * 0.7 + vector_similarity * 0.3;

        // Final: weighted combination of semantic + type compatibility
        // sqrt(semantic^0.7 * type^0.3)
        let overall = (semantic_score.powf(0.7) * type_compatibility.powf(0.3)).sqrt();

        Self {
            vector_similarity: semantic_score, // Report combined score
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

// ============================================================================
// Type Compatibility Scoring
// ============================================================================

/// Compute type compatibility score.
///
/// Higher score for compatible/coercible types.
/// 
/// Scores:
/// - 1.0: Exact type match
/// - 0.85: Null compatible with anything (optional fields)
/// - 0.75: String <-> Number (common ID representation change)
/// - 0.6: Bool <-> Number (0/1 encoding)
/// - 0.3: Array <-> Object (structural mismatch)
/// - 0.1: Incompatible primitives
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_compatibility_exact() {
        assert_eq!(type_compatibility_score(JsonType::String, JsonType::String), 1.0);
        assert_eq!(type_compatibility_score(JsonType::Number, JsonType::Number), 1.0);
        assert_eq!(type_compatibility_score(JsonType::Bool, JsonType::Bool), 1.0);
    }

    #[test]
    fn test_type_compatibility_coercible() {
        // String <-> Number
        assert_eq!(type_compatibility_score(JsonType::String, JsonType::Number), 0.75);
        assert_eq!(type_compatibility_score(JsonType::Number, JsonType::String), 0.75);

        // Null is optional
        assert_eq!(type_compatibility_score(JsonType::Null, JsonType::String), 0.85);
        assert_eq!(type_compatibility_score(JsonType::Number, JsonType::Null), 0.85);
    }

    #[test]
    fn test_type_compatibility_incompatible() {
        assert_eq!(type_compatibility_score(JsonType::Array, JsonType::Object), 0.3);
        assert_eq!(type_compatibility_score(JsonType::String, JsonType::Array), 0.1);
    }
}
