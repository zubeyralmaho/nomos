//! Combined NLP Similarity Score (Ensemble)
//!
//! Combines multiple NLP metrics for robust matching.
//! Each algorithm has different strengths:
//! - Levenshtein: Good for typos and character-level errors
//! - Jaro-Winkler: Good for prefix-similar names
//! - N-gram TF-IDF: Good for structural similarity
//! - Synonym: Good for semantic equivalence

use super::synonym::{synonym_match, normalize_field_name};
use super::levenshtein::levenshtein_similarity;
use super::jaro::jaro_winkler_similarity;
use super::tfidf::ngram_tfidf_similarity;

/// Combined NLP similarity score using ensemble of algorithms.
/// 
/// # Weights
/// - Synonym match: 0.35 (highest priority - true semantic)
/// - Jaro-Winkler: 0.25 (good for names)
/// - Levenshtein: 0.20 (character errors)
/// - N-gram TF-IDF: 0.20 (structural)
/// 
/// # Returns
/// Combined score from 0.0 to 1.0
///
/// # Example
/// ```
/// use nomos_core::nlp::combined::combined_nlp_similarity;
/// 
/// // Exact match
/// assert!(combined_nlp_similarity("user", "user") == 1.0);
/// 
/// // Synonym match (high score)
/// assert!(combined_nlp_similarity("user", "person") > 0.6);
/// 
/// // Character-level similarity
/// assert!(combined_nlp_similarity("userId", "user_id") > 0.9);
/// ```
pub fn combined_nlp_similarity(a: &str, b: &str) -> f32 {
    // Short-circuit for exact match
    if a == b {
        return 1.0;
    }
    
    // Normalize for comparison
    let norm_a = normalize_field_name(a);
    let norm_b = normalize_field_name(b);
    
    if norm_a == norm_b {
        return 1.0;
    }
    
    // 1. Synonym lookup (true semantic)
    let synonym_score = synonym_match(a, b);
    
    // 2. Jaro-Winkler (prefix-aware)
    let jw_score = jaro_winkler_similarity(&norm_a, &norm_b);
    
    // 3. Levenshtein (edit distance)
    let lev_score = levenshtein_similarity(&norm_a, &norm_b);
    
    // 4. N-gram TF-IDF (trigrams)
    let tfidf_score = ngram_tfidf_similarity(&norm_a, &norm_b, 3);
    
    // Weighted ensemble
    const W_SYNONYM: f32 = 0.35;
    const W_JARO_WINKLER: f32 = 0.25;
    const W_LEVENSHTEIN: f32 = 0.20;
    const W_TFIDF: f32 = 0.20;
    
    // If synonym match found, boost it significantly
    if synonym_score > 0.5 {
        return synonym_score * 0.6 + (jw_score + lev_score + tfidf_score) / 3.0 * 0.4;
    }
    
    // Otherwise, use weighted average
    W_SYNONYM * synonym_score 
        + W_JARO_WINKLER * jw_score 
        + W_LEVENSHTEIN * lev_score 
        + W_TFIDF * tfidf_score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combined_similarity() {
        // Exact match → 1.0
        assert!((combined_nlp_similarity("user", "user") - 1.0).abs() < 0.001);
        
        // Normalized match → 1.0
        assert!((combined_nlp_similarity("userId", "user_id") - 1.0).abs() < 0.001);
        
        // Synonym match (semantic)
        assert!(combined_nlp_similarity("user", "person") > 0.65);
        
        // Unrelated fields
        assert!(combined_nlp_similarity("email", "status") < 0.4);
    }
}
