//! Semantic Healing Engine
//!
//! This module contains the core schema healing functionality:
//!
//! - `simd`: SIMD-optimized dot product operations
//! - `embedding`: Field embeddings using trigram hashing
//! - `confidence`: Match confidence scoring
//! - `lsh`: Locality-Sensitive Hashing for O(1) lookups
//! - `healing`: Healing operations and healing maps
//! - `healer`: The main SemanticHealer implementation
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    SemanticHealer                               │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
//! │  │ LSH Index    │  │ Embeddings   │  │ Healing Operations   │  │
//! │  │ O(1) lookup  │  │ Trigram-based│  │ Rename/Coerce/Delete │  │
//! │  └──────────────┘  └──────────────┘  └──────────────────────┘  │
//! │                           │                                     │
//! │                    ┌──────┴──────┐                              │
//! │                    │ Confidence  │                              │
//! │                    │ NLP Ensemble│                              │
//! │                    └─────────────┘                              │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance Targets
//! - Sub-50µs healing path for typical drift
//! - O(1) candidate lookup with LSH for >100 fields
//! - SIMD-accelerated similarity computation

pub mod simd;
pub mod embedding;
pub mod confidence;
pub mod lsh;
pub mod healing;
pub mod healer;

// Re-export main types
pub use simd::{EMBEDDING_DIM, simd_dot_product_i8};
pub use embedding::FieldEmbedding;
pub use confidence::{MatchConfidence, type_compatibility_score};
pub use lsh::{LshIndex, LSH_THRESHOLD};
pub use healing::{HealingOp, HealingMap};
pub use healer::{SemanticHealer, DEFAULT_CONFIDENCE_THRESHOLD};
