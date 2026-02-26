//! NLP Module - String Similarity Algorithms
//!
//! This module provides a collection of proven algorithms
//! for string similarity matching.
//!
//! # Algorithms
//!
//! | Algorithm | Complexity | Best For |
//! |-----------|------------|----------|
//! | [Levenshtein](levenshtein) | O(m×n) | Typo detection |
//! | [Jaro-Winkler](jaro) | O(m×n) | Name matching |
//! | [TF-IDF](tfidf) | O(n) | Structural sim |
//! | [Synonym](synonym) | O(1) | Semantic match |
//! | [Combined](combined) | O(m×n) | Robust matching |
//!
//! # Example
//!
//! ```
//! use nomos_core::nlp::combined::combined_nlp_similarity;
//! use nomos_core::nlp::levenshtein::levenshtein_distance;
//! use nomos_core::nlp::jaro::jaro_winkler_similarity;
//!
//! // Combined similarity for field name matching
//! let sim = combined_nlp_similarity("userId", "user_id");
//! assert!(sim > 0.9);
//!
//! // Individual algorithms available
//! let dist = levenshtein_distance("kitten", "sitting");
//! assert_eq!(dist, 3);
//! ```

pub mod synonym;
pub mod levenshtein;
pub mod jaro;
pub mod tfidf;
pub mod combined;

// Re-export main functions for convenience
pub use synonym::{synonym_match, normalize_field_name, SYNONYM_GROUPS, SYNONYM_MAP};
pub use levenshtein::{levenshtein_distance, levenshtein_similarity};
pub use jaro::{jaro_similarity, jaro_winkler_similarity};
pub use tfidf::{extract_ngrams, compute_tf, ngram_tfidf_similarity};
pub use combined::combined_nlp_similarity;
