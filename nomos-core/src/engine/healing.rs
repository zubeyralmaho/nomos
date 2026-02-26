//! Healing Operations and Healing Map
//!
//! Types for representing schema transformations:
//! - HealingOp: Individual field transformation operations
//! - HealingMap: Complete transformation plan for schema drift

use std::sync::Arc;

use crate::schema::{JsonType, SchemaFingerprint};

// ============================================================================
// Healing Operation
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
            HealingOp::Rename {
                from,
                to,
                confidence,
            } => {
                write!(
                    f,
                    "RENAME: {} -> {} | Confidence: {:.2}",
                    from, to, confidence
                )
            }
            HealingOp::CoerceType {
                field,
                from_type,
                to_type,
            } => {
                write!(f, "COERCE: {} ({:?} -> {:?})", field, from_type, to_type)
            }
            HealingOp::SetDefault {
                field,
                default_value,
            } => {
                write!(f, "DEFAULT: {} = {}", field, default_value)
            }
            HealingOp::Delete { field } => {
                write!(f, "DELETE: {}", field)
            }
        }
    }
}

// ============================================================================
// Healing Map
// ============================================================================

/// Healing map containing all transformations for a schema drift.
///
/// This represents a complete plan for transforming data from a source
/// schema to a target schema, including field renames, type coercions,
/// default values, and deletions.
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

impl HealingMap {
    /// Create a new empty healing map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a healing operation.
    pub fn add_operation(&mut self, op: HealingOp) {
        self.operations.push(op);
    }

    /// Get number of operations.
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    /// Check if healing map is empty.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Get rename operations only.
    pub fn renames(&self) -> impl Iterator<Item = &HealingOp> {
        self.operations
            .iter()
            .filter(|op| matches!(op, HealingOp::Rename { .. }))
    }

    /// Get coercion operations only.
    pub fn coercions(&self) -> impl Iterator<Item = &HealingOp> {
        self.operations
            .iter()
            .filter(|op| matches!(op, HealingOp::CoerceType { .. }))
    }

    /// Get default value operations only.
    pub fn defaults(&self) -> impl Iterator<Item = &HealingOp> {
        self.operations
            .iter()
            .filter(|op| matches!(op, HealingOp::SetDefault { .. }))
    }

    /// Get delete operations only.
    pub fn deletions(&self) -> impl Iterator<Item = &HealingOp> {
        self.operations
            .iter()
            .filter(|op| matches!(op, HealingOp::Delete { .. }))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_healing_op_display() {
        let rename = HealingOp::Rename {
            from: "old_field".into(),
            to: "new_field".into(),
            confidence: 0.95,
        };
        assert!(format!("{}", rename).contains("RENAME"));
        assert!(format!("{}", rename).contains("old_field"));
        assert!(format!("{}", rename).contains("new_field"));

        let coerce = HealingOp::CoerceType {
            field: "id".into(),
            from_type: JsonType::String,
            to_type: JsonType::Number,
        };
        assert!(format!("{}", coerce).contains("COERCE"));

        let default = HealingOp::SetDefault {
            field: "status".into(),
            default_value: "active".into(),
        };
        assert!(format!("{}", default).contains("DEFAULT"));

        let delete = HealingOp::Delete {
            field: "deprecated".into(),
        };
        assert!(format!("{}", delete).contains("DELETE"));
    }

    #[test]
    fn test_healing_map_operations() {
        let mut map = HealingMap::new();
        assert!(map.is_empty());
        assert_eq!(map.operation_count(), 0);

        map.add_operation(HealingOp::Rename {
            from: "a".into(),
            to: "b".into(),
            confidence: 0.9,
        });

        map.add_operation(HealingOp::Delete {
            field: "c".into(),
        });

        assert!(!map.is_empty());
        assert_eq!(map.operation_count(), 2);
        assert_eq!(map.renames().count(), 1);
        assert_eq!(map.deletions().count(), 1);
        assert_eq!(map.coercions().count(), 0);
    }
}
