//! N-gram TF-IDF Similarity
//!
//! Reference: Salton, G. & McGill, M. J. (1983). "Introduction to Modern
//!            Information Retrieval"
//!
//! TF-IDF = Term Frequency × Inverse Document Frequency
//! 
//! For field names, we use character n-grams as terms and a modified
//! Dice coefficient since classical TF-IDF is designed for documents.

use std::collections::{HashMap, HashSet};

/// Extract character n-grams from a string.
/// 
/// Uses boundary markers ($) for start and end positions.
///
/// # Example
/// ```
/// use nomos_core::nlp::tfidf::extract_ngrams;
/// 
/// let trigrams = extract_ngrams("user", 3);
/// assert!(trigrams.contains(&"$$u".to_string()));
/// assert!(trigrams.contains(&"use".to_string()));
/// assert!(trigrams.contains(&"ser".to_string()));
/// ```
pub fn extract_ngrams(s: &str, n: usize) -> Vec<String> {
    if s.is_empty() || n == 0 {
        return vec![];
    }
    
    // Add boundary markers
    let padded = format!("{}{}{}", "$".repeat(n - 1), s, "$".repeat(n - 1));
    let chars: Vec<char> = padded.chars().collect();
    
    chars.windows(n)
        .map(|window| window.iter().collect())
        .collect()
}

/// Compute TF (Term Frequency) for n-grams.
/// 
/// Returns a map of n-gram → normalized frequency.
///
/// # Formula
/// `TF(t) = count(t) / total_count`
pub fn compute_tf(ngrams: &[String]) -> HashMap<String, f32> {
    let mut tf: HashMap<String, usize> = HashMap::new();
    
    for ngram in ngrams {
        *tf.entry(ngram.clone()).or_insert(0) += 1;
    }
    
    let total = ngrams.len() as f32;
    tf.into_iter()
        .map(|(k, v)| (k, v as f32 / total))
        .collect()
}

/// Compute n-gram TF-IDF similarity between two strings.
/// 
/// # Algorithm
/// For short strings (like field names), we use a modified approach:
/// 1. Extract n-grams from both strings
/// 2. Compute TF-weighted Dice coefficient: `2×|A∩B| / (|A| + |B|)`
/// 3. Weight by TF for repeated n-grams
/// 
/// # Parameters
/// - `n`: n-gram size (2 or 3 recommended for field names)
/// 
/// # Note
/// Classical TF-IDF is designed for documents, not short strings.
/// For field names, the Dice coefficient with TF weighting is more appropriate.
///
/// # Example
/// ```
/// use nomos_core::nlp::tfidf::ngram_tfidf_similarity;
/// 
/// let sim = ngram_tfidf_similarity("user_id", "userId", 3);
/// assert!(sim > 0.5);
/// ```
pub fn ngram_tfidf_similarity(s1: &str, s2: &str, n: usize) -> f32 {
    let ngrams1 = extract_ngrams(&s1.to_lowercase(), n);
    let ngrams2 = extract_ngrams(&s2.to_lowercase(), n);
    
    if ngrams1.is_empty() && ngrams2.is_empty() {
        return 1.0;
    }
    if ngrams1.is_empty() || ngrams2.is_empty() {
        return 0.0;
    }
    
    let tf1 = compute_tf(&ngrams1);
    let tf2 = compute_tf(&ngrams2);
    
    // Compute weighted Dice coefficient
    // Standard Dice: 2×|A∩B| / (|A| + |B|)
    // We weight the intersection by min TF
    
    let set1: HashSet<&String> = tf1.keys().collect();
    let set2: HashSet<&String> = tf2.keys().collect();
    
    let intersection: Vec<_> = set1.intersection(&set2).collect();
    
    if intersection.is_empty() {
        return 0.0;
    }
    
    // Weighted intersection (sum of min TF for shared n-grams)
    let weighted_intersection: f32 = intersection.iter()
        .map(|ng| {
            let t1 = tf1.get(**ng).copied().unwrap_or(0.0);
            let t2 = tf2.get(**ng).copied().unwrap_or(0.0);
            t1.min(t2) * 2.0 // *2 for Dice numerator
        })
        .sum();
    
    // Denominator: sum of all TF values
    let total_tf: f32 = tf1.values().sum::<f32>() + tf2.values().sum::<f32>();
    
    if total_tf == 0.0 {
        return 0.0;
    }
    
    weighted_intersection / total_tf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_extraction() {
        let trigrams = extract_ngrams("user", 3);
        assert!(trigrams.contains(&"$$u".to_string()));
        assert!(trigrams.contains(&"use".to_string()));
        assert!(trigrams.contains(&"ser".to_string()));
        assert!(trigrams.contains(&"er$".to_string()));
        
        let bigrams = extract_ngrams("ab", 2);
        assert!(bigrams.len() >= 2);
    }

    #[test]
    fn test_tfidf_similarity() {
        // Identical strings → 1.0
        let sim = ngram_tfidf_similarity("user", "user", 3);
        assert!((sim - 1.0).abs() < 0.001);
        
        // Similar strings should have moderate similarity
        let sim = ngram_tfidf_similarity("user_id", "userId", 3);
        assert!(sim > 0.5);
        
        // Different strings should have lower similarity
        let sim = ngram_tfidf_similarity("email", "phone", 3);
        assert!(sim < 0.3);
    }
}
