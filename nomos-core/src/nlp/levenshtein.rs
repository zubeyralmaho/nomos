//! Levenshtein Distance (Edit Distance)
//!
//! Reference: Levenshtein, V. I. (1966). "Binary codes capable of correcting
//!            deletions, insertions, and reversals"
//!
//! # Time Complexity
//! O(m × n) where m, n are string lengths
//!
//! # Space Complexity
//! O(min(m, n)) with space optimization
//!
//! # Operations
//! - Insert: cost = 1
//! - Delete: cost = 1
//! - Substitute: cost = 1

/// Compute Levenshtein edit distance between two strings.
/// 
/// # Algorithm
/// Uses dynamic programming with Wagner-Fischer algorithm.
/// Space-optimized to use only O(min(m,n)) memory.
/// 
/// # Returns
/// Number of single-character edits (insertions, deletions, substitutions)
/// needed to transform `a` into `b`.
///
/// # Example
/// ```
/// use nomos_core::nlp::levenshtein::levenshtein_distance;
/// 
/// assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
/// assert_eq!(levenshtein_distance("cat", "hat"), 1);
/// ```
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    
    let m = a_chars.len();
    let n = b_chars.len();
    
    // Optimization: ensure we iterate over the shorter string in inner loop
    if m > n {
        return levenshtein_distance(b, a);
    }
    
    // Space optimization: only keep two rows
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr: Vec<usize> = vec![0; m + 1];
    
    for j in 1..=n {
        curr[0] = j;
        
        for i in 1..=m {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            
            curr[i] = (prev[i] + 1)           // deletion
                .min(curr[i - 1] + 1)          // insertion
                .min(prev[i - 1] + cost);      // substitution
        }
        
        std::mem::swap(&mut prev, &mut curr);
    }
    
    prev[m]
}

/// Normalized Levenshtein similarity (0.0 to 1.0).
/// 
/// # Formula
/// `similarity = 1.0 - (distance / max_length)`
///
/// # Example
/// ```
/// use nomos_core::nlp::levenshtein::levenshtein_similarity;
/// 
/// assert!(levenshtein_similarity("hello", "hello") == 1.0);
/// assert!(levenshtein_similarity("cat", "hat") > 0.6);
/// ```
#[inline]
pub fn levenshtein_similarity(a: &str, b: &str) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    
    let distance = levenshtein_distance(a, b);
    let max_len = a.chars().count().max(b.chars().count());
    
    1.0 - (distance as f32 / max_len as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_distance() {
        // Empty strings
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "xyz"), 3);
        
        // Identical strings
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        
        // Single operations
        assert_eq!(levenshtein_distance("cat", "hat"), 1);      // substitution
        assert_eq!(levenshtein_distance("cat", "cats"), 1);     // insertion
        assert_eq!(levenshtein_distance("cats", "cat"), 1);     // deletion
        
        // Classic examples
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("saturday", "sunday"), 3);
    }

    #[test]
    fn test_levenshtein_similarity() {
        // Identical → 1.0
        assert!((levenshtein_similarity("hello", "hello") - 1.0).abs() < 0.001);
        
        // Similar strings
        assert!(levenshtein_similarity("user_id", "userId") > 0.6);
        
        // Completely different → low score
        assert!(levenshtein_similarity("abc", "xyz") < 0.1);
    }
}
