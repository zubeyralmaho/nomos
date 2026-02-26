//! Jaro and Jaro-Winkler Similarity
//!
//! Reference: Winkler, W. E. (1990). "String Comparator Metrics and Enhanced
//!            Decision Rules in the Fellegi-Sunter Model of Record Linkage"
//!
//! # Time Complexity
//! O(m × n) for matching window search
//!
//! # Space Complexity
//! O(m + n)
//!
//! Particularly effective for short strings like field names.

/// Compute Jaro similarity between two strings.
/// 
/// # Algorithm
/// 1. Find matching characters within a window of `floor(max(|s1|,|s2|)/2) - 1`
/// 2. Count transpositions (matched chars in different order)
/// 3. `Jaro = (m/|s1| + m/|s2| + (m-t)/m) / 3`
///    where m = matches, t = transpositions/2
///
/// # Example
/// ```
/// use nomos_core::nlp::jaro::jaro_similarity;
/// 
/// assert!(jaro_similarity("MARTHA", "MARHTA") > 0.94);
/// ```
pub fn jaro_similarity(s1: &str, s2: &str) -> f32 {
    if s1.is_empty() && s2.is_empty() {
        return 1.0;
    }
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }
    
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    
    let len1 = s1_chars.len();
    let len2 = s2_chars.len();
    
    // Match window: floor(max(len1, len2) / 2) - 1
    let match_window = (len1.max(len2) / 2).saturating_sub(1);
    
    let mut s1_matches = vec![false; len1];
    let mut s2_matches = vec![false; len2];
    
    let mut matches = 0;
    let mut transpositions = 0;
    
    // Find matching characters
    for i in 0..len1 {
        let start = i.saturating_sub(match_window);
        let end = (i + match_window + 1).min(len2);
        
        for j in start..end {
            if s2_matches[j] || s1_chars[i] != s2_chars[j] {
                continue;
            }
            s1_matches[i] = true;
            s2_matches[j] = true;
            matches += 1;
            break;
        }
    }
    
    if matches == 0 {
        return 0.0;
    }
    
    // Count transpositions
    let mut k = 0;
    for i in 0..len1 {
        if !s1_matches[i] {
            continue;
        }
        while !s2_matches[k] {
            k += 1;
        }
        if s1_chars[i] != s2_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }
    
    let m = matches as f32;
    let t = (transpositions / 2) as f32;
    
    (m / len1 as f32 + m / len2 as f32 + (m - t) / m) / 3.0
}

/// Jaro-Winkler similarity with prefix bonus.
/// 
/// # Algorithm
/// Adds a bonus for common prefix (up to 4 chars):
/// `JW = Jaro + (prefix_len × p × (1 - Jaro))`
/// 
/// Standard scaling factor `p = 0.1`
/// 
/// # Use Case
/// Particularly good for field names which often share prefixes
/// (`user_id`, `user_name`, `user_email`)
///
/// # Example
/// ```
/// use nomos_core::nlp::jaro::{jaro_similarity, jaro_winkler_similarity};
/// 
/// let jaro = jaro_similarity("user_id", "user_name");
/// let jw = jaro_winkler_similarity("user_id", "user_name");
/// assert!(jw >= jaro); // Winkler adds prefix bonus
/// ```
pub fn jaro_winkler_similarity(s1: &str, s2: &str) -> f32 {
    let jaro = jaro_similarity(s1, s2);
    
    // Standard Winkler scaling factor
    const SCALING_FACTOR: f32 = 0.1;
    const MAX_PREFIX_LENGTH: usize = 4;
    
    // Find common prefix length (up to 4 chars)
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    
    let prefix_len = s1_chars.iter()
        .zip(s2_chars.iter())
        .take(MAX_PREFIX_LENGTH)
        .take_while(|(c1, c2)| c1 == c2)
        .count();
    
    // Winkler modification
    jaro + (prefix_len as f32 * SCALING_FACTOR * (1.0 - jaro))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaro_similarity() {
        // Identical strings → 1.0
        assert!((jaro_similarity("hello", "hello") - 1.0).abs() < 0.001);
        
        // Empty strings
        assert!((jaro_similarity("", "") - 1.0).abs() < 0.001);
        assert!(jaro_similarity("abc", "").abs() < 0.001);
        
        // Classic examples from Jaro (1989)
        assert!(jaro_similarity("MARTHA", "MARHTA") > 0.94);
        assert!(jaro_similarity("DWAYNE", "DUANE") > 0.82);
    }

    #[test]
    fn test_jaro_winkler_similarity() {
        // Prefix bonus should increase score
        let jaro = jaro_similarity("user_id", "user_name");
        let jw = jaro_winkler_similarity("user_id", "user_name");
        assert!(jw >= jaro);
        
        // Names sharing prefix should score higher
        let jw1 = jaro_winkler_similarity("created_at", "created_date");
        let jw2 = jaro_winkler_similarity("created_at", "modified_at");
        assert!(jw1 > jw2);
    }
}
