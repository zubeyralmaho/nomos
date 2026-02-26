//! Semantic Healer - Autonomous Schema Reconstruction Engine
//!
//! The core healing engine that:
//! - Detects schema drift (field renames, type changes)
//! - Computes healing maps using semantic similarity
//! - Applies transformations to normalize data
//!
//! # Design Principles
//! - No ML runtime - uses trigram-based embeddings
//! - SIMD-optimized similarity computation
//! - LSH indexing for O(1) lookups on large schemas (>100 fields)
//! - Sub-50µs healing path target

use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::schema::{ExpectedSchema, JsonType, SchemaFingerprint};
use crate::engine::embedding::FieldEmbedding;
use crate::engine::confidence::MatchConfidence;
use crate::engine::healing::{HealingMap, HealingOp};
use crate::engine::lsh::{LshIndex, LSH_THRESHOLD};

// ============================================================================
// Constants
// ============================================================================

/// Default confidence threshold for accepting matches.
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.70;

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

    /// Get current confidence threshold.
    pub fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
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

        self.schema_embeddings
            .insert(schema.fingerprint, embeddings);
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
                self.find_best_match_linear(&actual_embedding, expected_embeddings, actual_fields)
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
                if actual_type != matched_field.json_type
                    && matched_field.json_type != JsonType::Null
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
                && !operations.iter().any(
                    |op| matches!(op, HealingOp::Rename { to, .. } if to == &expected_emb.field_name),
                )
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
                HealingOp::Rename {
                    from,
                    to,
                    confidence: _,
                } => {
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
                HealingOp::SetDefault {
                    field,
                    default_value,
                } => {
                    debug!(
                        field = field.as_ref(),
                        default = default_value.as_ref(),
                        "Default value set (implementation pending)"
                    );
                }
                HealingOp::Delete { field } => {
                    debug!(
                        field = field.as_ref(),
                        "Field deletion (implementation pending)"
                    );
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
    fn test_semantic_healer_creation() {
        let healer = SemanticHealer::new();
        assert_eq!(
            healer.confidence_threshold(),
            DEFAULT_CONFIDENCE_THRESHOLD
        );
    }

    #[test]
    fn test_semantic_healer_with_threshold() {
        let healer = SemanticHealer::with_threshold(0.85);
        assert_eq!(healer.confidence_threshold(), 0.85);
    }

    #[test]
    fn test_register_schema() {
        let mut healer = SemanticHealer::new();

        let mut field_types = HashMap::new();
        field_types.insert(Arc::from("id"), JsonType::Number);
        field_types.insert(Arc::from("name"), JsonType::String);

        let schema = ExpectedSchema {
            fingerprint: SchemaFingerprint(12345),
            root_fields: vec![Arc::from("id"), Arc::from("name")],
            field_types,
            version: 1,
            last_seen_timestamp: 0,
        };

        healer.register_schema(&schema);

        // Schema should be registered
        assert!(healer
            .schema_embeddings
            .contains_key(&schema.fingerprint));
    }
}
