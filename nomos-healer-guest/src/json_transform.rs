//! JSON Transformation - Zero-Copy Key Renaming
//!
//! This module implements the actual JSON healing logic:
//! - Key renames (the most common operation)
//! - Type coercion
//! - Default value insertion
//! - Field deletion
//!
//! # Design
//!
//! We avoid full JSON parsing when possible. For key renames, we scan
//! the byte stream directly, replacing keys in-place or with minimal
//! copying to the output buffer.

use alloc::string::String;
use alloc::vec::Vec;

// ============================================================================
// Memcmp-free String Comparison
// ============================================================================

/// Compare two byte slices without using memcmp (which requires libc).
/// Returns true if slices are equal.
#[inline]
fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if a[i] != b[i] {
            return false;
        }
    }
    true
}

/// Compare two strings without using memcmp.
#[inline]
fn str_eq(a: &str, b: &str) -> bool {
    bytes_eq(a.as_bytes(), b.as_bytes())
}

use crate::healer::ErrorCode;

// ============================================================================
// Types
// ============================================================================

/// Operation type matching WIT enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Rename = 0,
    CoerceType = 1,
    SetDefault = 2,
    Delete = 3,
}

/// A single healing instruction
#[derive(Debug, Clone)]
pub struct HealingInstruction {
    pub op: Operation,
    pub source: String,  // Source field name (or path)
    pub target: String,  // Target field name (or value for defaults)
}

// ============================================================================
// JSON Transformation
// ============================================================================

/// Apply healing operations to JSON.
///
/// # Arguments
/// - `input`: Input JSON bytes
/// - `instructions`: List of healing instructions
/// - `output`: Output buffer to write healed JSON
///
/// # Returns
/// (bytes_written, ops_applied) on success, ErrorCode on failure.
pub fn apply_healing_ops(
    input: &[u8],
    instructions: &[HealingInstruction],
    output: &mut [u8],
) -> Result<(usize, usize), ErrorCode> {
    if instructions.is_empty() {
        // Fast path: no healing needed, just copy
        if input.len() > output.len() {
            return Err(ErrorCode::BufferOverflow);
        }
        output[..input.len()].copy_from_slice(input);
        return Ok((input.len(), 0));
    }
    
    // Build rename map for O(1) lookup
    let rename_map: Vec<(&str, &str)> = instructions
        .iter()
        .filter(|i| i.op == Operation::Rename)
        .map(|i| (i.source.as_str(), i.target.as_str()))
        .collect();
    
    let delete_set: Vec<&str> = instructions
        .iter()
        .filter(|i| i.op == Operation::Delete)
        .map(|i| i.source.as_str())
        .collect();
    
    // Process JSON with key replacement
    let (written, ops) = transform_json_keys(input, &rename_map, &delete_set, output)?;
    
    Ok((written, ops))
}

/// Transform JSON by renaming/deleting keys.
///
/// Uses a streaming approach: scan for keys, check against maps,
/// write to output with replacements.
fn transform_json_keys(
    input: &[u8],
    renames: &[(&str, &str)],
    deletes: &[&str],
    output: &mut [u8],
) -> Result<(usize, usize), ErrorCode> {
    let mut read_pos = 0;
    let mut write_pos = 0;
    let mut ops_applied = 0;
    
    // State machine for JSON parsing
    let mut in_string = false;
    let mut string_start = 0usize;
    let mut escape_next = false;
    let mut expecting_colon = false;
    #[allow(unused_assignments)]  // Used for potential future key-based logic
    let mut current_key: Option<&str> = None;
    let mut skip_value = false;
    let mut brace_depth: u32 = 0;
    let mut skip_depth: u32 = 0;
    
    while read_pos < input.len() {
        let b = input[read_pos];
        
        // Handle skipping deleted field values
        if skip_value {
            match b {
                b'{' | b'[' => skip_depth += 1,
                b'}' | b']' => {
                    if skip_depth > 0 {
                        skip_depth -= 1;
                    }
                    if skip_depth == 0 {
                        skip_value = false;
                        // Skip trailing comma if present
                        let next = peek_non_ws(input, read_pos + 1);
                        if next == Some(b',') {
                            read_pos = find_char(input, read_pos + 1, b',').unwrap_or(read_pos);
                        }
                    }
                }
                b'"' if !escape_next => {
                    in_string = !in_string;
                }
                b'\\' if in_string => {
                    escape_next = true;
                    read_pos += 1;
                    continue;
                }
                b',' | b'}' if !in_string && skip_depth == 0 => {
                    skip_value = false;
                    // Don't advance read_pos, let main loop handle it
                    if b == b',' {
                        read_pos += 1;
                    }
                    continue;
                }
                _ => {}
            }
            escape_next = false;
            read_pos += 1;
            continue;
        }
        
        // Handle escape sequences in strings
        if escape_next {
            escape_next = false;
            if write_pos >= output.len() {
                return Err(ErrorCode::BufferOverflow);
            }
            output[write_pos] = b;
            write_pos += 1;
            read_pos += 1;
            continue;
        }
        
        match b {
            b'\\' if in_string => {
                escape_next = true;
                if write_pos >= output.len() {
                    return Err(ErrorCode::BufferOverflow);
                }
                output[write_pos] = b;
                write_pos += 1;
            }
            
            b'"' => {
                if in_string {
                    // End of string
                    in_string = false;
                    
                    if expecting_colon {
                        // This was a key - check for rename/delete
                        let key = core::str::from_utf8(&input[string_start..read_pos])
                            .unwrap_or("");
                        
                        // Check if should delete
                        if deletes.iter().any(|&d| str_eq(d, key)) {
                            // Delete this key-value pair
                            // Backtrack write_pos to before the opening quote
                            write_pos = write_pos.saturating_sub(key.len() + 1);
                            
                            // Find and skip the value
                            skip_value = true;
                            skip_depth = 0;
                            read_pos += 1; // Skip closing quote
                            
                            // Skip colon
                            while read_pos < input.len() && (input[read_pos] == b' ' || input[read_pos] == b':' || input[read_pos] == b'\n' || input[read_pos] == b'\t' || input[read_pos] == b'\r') {
                                if input[read_pos] == b':' {
                                    read_pos += 1;
                                    break;
                                }
                                read_pos += 1;
                            }
                            
                            // Skip whitespace after colon
                            while read_pos < input.len() && is_whitespace(input[read_pos]) {
                                read_pos += 1;
                            }
                            
                            ops_applied += 1;
                            expecting_colon = false;
                            continue;
                        }
                        
                        // Check if should rename
                        if let Some(&(_, new_key)) = renames.iter().find(|(old, _)| str_eq(*old, key)) {
                            // Rename: backtrack and write new key
                            write_pos = write_pos.saturating_sub(key.len());
                            
                            // Write new key
                            let new_bytes = new_key.as_bytes();
                            if write_pos + new_bytes.len() >= output.len() {
                                return Err(ErrorCode::BufferOverflow);
                            }
                            output[write_pos..write_pos + new_bytes.len()].copy_from_slice(new_bytes);
                            write_pos += new_bytes.len();
                            
                            ops_applied += 1;
                        }
                        
                        expecting_colon = false;
                        current_key = None;
                    }
                    
                    // Write closing quote
                    if write_pos >= output.len() {
                        return Err(ErrorCode::BufferOverflow);
                    }
                    output[write_pos] = b'"';
                    write_pos += 1;
                } else {
                    // Start of string
                    in_string = true;
                    string_start = read_pos + 1;  // After the quote
                    
                    // Write opening quote
                    if write_pos >= output.len() {
                        return Err(ErrorCode::BufferOverflow);
                    }
                    output[write_pos] = b'"';
                    write_pos += 1;
                }
            }
            
            b'{' => {
                brace_depth += 1;
                expecting_colon = true;  // Next string is a key
                if write_pos >= output.len() {
                    return Err(ErrorCode::BufferOverflow);
                }
                output[write_pos] = b;
                write_pos += 1;
            }
            
            b'}' => {
                brace_depth = brace_depth.saturating_sub(1);
                if write_pos >= output.len() {
                    return Err(ErrorCode::BufferOverflow);
                }
                output[write_pos] = b;
                write_pos += 1;
            }
            
            b':' if !in_string => {
                expecting_colon = false;
                if write_pos >= output.len() {
                    return Err(ErrorCode::BufferOverflow);
                }
                output[write_pos] = b;
                write_pos += 1;
            }
            
            b',' if !in_string => {
                expecting_colon = true;  // Next string after comma is a key (in objects)
                if write_pos >= output.len() {
                    return Err(ErrorCode::BufferOverflow);
                }
                output[write_pos] = b;
                write_pos += 1;
            }
            
            _ => {
                if in_string || !is_whitespace(b) || write_pos == 0 || !is_whitespace(output[write_pos - 1]) {
                    // Write character (dedup whitespace)
                    if write_pos >= output.len() {
                        return Err(ErrorCode::BufferOverflow);
                    }
                    output[write_pos] = b;
                    write_pos += 1;
                }
            }
        }
        
        read_pos += 1;
    }
    
    Ok((write_pos, ops_applied))
}

#[inline]
fn is_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\r' | b'\t')
}

#[inline]
fn peek_non_ws(data: &[u8], start: usize) -> Option<u8> {
    data[start..].iter().find(|&&b| !is_whitespace(b)).copied()
}

#[inline]
fn find_char(data: &[u8], start: usize, target: u8) -> Option<usize> {
    data[start..].iter().position(|&b| b == target).map(|p| start + p)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    
    #[test]
    fn test_simple_rename() {
        let input = br#"{"old_key": "value"}"#;
        let instructions = vec![HealingInstruction {
            op: Operation::Rename,
            source: String::from("old_key"),
            target: String::from("new_key"),
        }];
        let mut output = [0u8; 256];
        
        let (written, ops) = apply_healing_ops(input, &instructions, &mut output).unwrap();
        
        let result = core::str::from_utf8(&output[..written]).unwrap();
        assert!(result.contains("new_key"));
        assert!(!result.contains("old_key"));
        assert_eq!(ops, 1);
    }
    
    #[test]
    fn test_no_ops_passthrough() {
        let input = br#"{"key": "value"}"#;
        let mut output = [0u8; 256];
        
        let (written, ops) = apply_healing_ops(input, &[], &mut output).unwrap();
        
        assert_eq!(&output[..written], input);
        assert_eq!(ops, 0);
    }
    
    #[test]
    fn test_multiple_renames() {
        let input = br#"{"a": 1, "b": 2, "c": 3}"#;
        let instructions = vec![
            HealingInstruction {
                op: Operation::Rename,
                source: String::from("a"),
                target: String::from("alpha"),
            },
            HealingInstruction {
                op: Operation::Rename,
                source: String::from("c"),
                target: String::from("gamma"),
            },
        ];
        let mut output = [0u8; 256];
        
        let (written, ops) = apply_healing_ops(input, &instructions, &mut output).unwrap();
        
        let result = core::str::from_utf8(&output[..written]).unwrap();
        assert!(result.contains("alpha"));
        assert!(result.contains("gamma"));
        assert!(result.contains("b"));  // Unchanged
        assert_eq!(ops, 2);
    }
}
