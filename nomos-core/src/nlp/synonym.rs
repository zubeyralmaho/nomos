//! Synonym Dictionary for True Semantic Understanding
//!
//! Unlike trigram hashing which only captures character-level similarity,
//! the synonym dictionary enables TRUE semantic matching:
//!   - "user" ↔ "person" ↔ "account" ↔ "member"
//!   - "created_at" ↔ "timestamp" ↔ "date"
//!   - "description" ↔ "summary" ↔ "text"
//!
//! # Performance
//! - Sub-100µs latency (no ML runtime)
//! - Zero dependencies (no model files)
//! - O(1) lookup via pre-built HashMap

use std::collections::HashMap;
use std::sync::LazyLock;

/// Synonym groups for NLP-level semantic matching.
/// Each group contains semantically equivalent terms.
pub static SYNONYM_GROUPS: &[&[&str]] = &[
    // Identity
    &["id", "identifier", "key", "uuid", "uid", "guid", "pk", "primary_key"],
    &["user", "person", "account", "member", "customer", "client", "owner"],
    &["user_id", "userId", "uid", "user_key", "account_id", "member_id", "person_id"],
    
    // Names
    &["name", "title", "label", "display_name", "displayName"],
    &["full_name", "fullName", "full_name", "displayName", "display_name"],
    &["first_name", "firstName", "given_name", "givenName", "fname"],
    &["last_name", "lastName", "family_name", "familyName", "surname", "lname"],
    &["username", "user_name", "login", "handle", "screen_name", "nickname"],
    
    // Contact
    &["email", "mail", "e_mail", "email_address", "emailAddress"],
    &["phone", "telephone", "mobile", "cell", "phone_number", "phoneNumber", "tel"],
    &["address", "location", "place", "addr", "street_address"],
    
    // Time - merged group for interchangeable time fields
    &["created_at", "createdAt", "created", "creation_date", "create_date", "date_created", "timestamp", "time", "datetime", "date_time", "ts"],
    &["updated_at", "updatedAt", "updated", "modified", "modified_at", "modifiedAt", "last_modified"],
    &["date", "day", "calendar_date"],
    &["expires_at", "expiresAt", "expiry", "expiration", "valid_until"],
    
    // Status
    &["status", "state", "condition", "phase"],
    &["active", "enabled", "is_active", "isActive"],
    &["deleted", "removed", "is_deleted", "isDeleted", "soft_deleted"],
    
    // Content
    &["description", "desc", "summary", "details", "about"],
    &["message", "msg", "text", "body", "content", "payload"],
    &["comment", "note", "remark", "annotation"],
    &["url", "link", "href", "uri", "web_address"],
    &["image", "img", "picture", "photo", "avatar", "icon"],
    
    // Numeric
    &["count", "number", "qty", "quantity", "num", "total"],
    &["amount", "value", "sum", "total"],
    &["price", "cost", "rate", "fee", "charge"],
    &["size", "length", "dimension", "magnitude"],
    &["age", "years", "years_old"],
    
    // Classification
    &["type", "kind", "category", "class", "group", "classification"],
    &["tag", "label", "marker", "flag"],
    &["version", "ver", "revision", "rev"],
    &["code", "key", "identifier", "ref"],
    
    // Boolean
    &["is_verified", "isVerified", "verified", "confirmed"],
    &["is_admin", "isAdmin", "admin", "administrator"],
    &["is_premium", "isPremium", "premium", "pro", "paid"],
    &["public", "is_public", "isPublic", "visible"],
    &["private", "is_private", "isPrivate", "hidden"],
    
    // Collections
    &["items", "list", "elements", "entries", "records", "data"],
    &["users", "members", "people", "persons", "accounts"],
    &["results", "response", "data", "output"],
    &["metadata", "meta", "info", "properties", "attrs", "attributes"],
    &["settings", "config", "configuration", "preferences", "options"],
    
    // Relationships
    &["parent", "parent_id", "parentId", "owner", "owner_id"],
    &["children", "child_ids", "childIds", "items", "members"],
    &["reference", "ref", "ref_id", "refId", "foreign_key", "fk"],
];

/// Pre-built synonym lookup map for O(1) access.
/// Maps normalized field name -> canonical form.
pub static SYNONYM_MAP: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut map = HashMap::with_capacity(500);
    for group in SYNONYM_GROUPS {
        let canonical = group[0]; // First item is canonical form
        for &synonym in *group {
            map.insert(synonym, canonical);
        }
    }
    map
});

/// Check if two field names are semantically equivalent via synonym dictionary.
/// 
/// # Returns
/// - `1.0` for exact match after normalization
/// - `0.95` for synonym group match
/// - `0.0` for no semantic match
///
/// # Example
/// ```
/// use nomos_core::nlp::synonym::synonym_match;
/// 
/// assert!(synonym_match("user", "person") > 0.9);
/// assert!(synonym_match("email", "mail") > 0.9);
/// ```
#[inline]
pub fn synonym_match(field_a: &str, field_b: &str) -> f32 {
    let norm_a = normalize_field_name(field_a);
    let norm_b = normalize_field_name(field_b);
    
    // Direct match after normalization
    if norm_a == norm_b {
        return 1.0;
    }
    
    // Check synonym dictionary
    let canonical_a = SYNONYM_MAP.get(norm_a.as_str());
    let canonical_b = SYNONYM_MAP.get(norm_b.as_str());
    
    match (canonical_a, canonical_b) {
        (Some(a), Some(b)) if a == b => 0.95, // Same synonym group
        _ => 0.0, // No semantic match
    }
}

/// Normalize field name for synonym lookup.
/// 
/// Converts camelCase to snake_case and lowercases.
/// Also strips common prefixes like `get_`, `set_`, `has_`.
#[inline]
pub fn normalize_field_name(name: &str) -> String {
    // First: convert camelCase to snake_case (before lowercasing!)
    let mut snake = String::with_capacity(name.len() + 4);
    for (i, c) in name.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            snake.push('_');
        }
        snake.push(c.to_ascii_lowercase());
    }
    
    // Then: strip common prefixes
    let stripped = snake
        .strip_prefix("get_")
        .or_else(|| snake.strip_prefix("set_"))
        .or_else(|| snake.strip_prefix("has_"))
        .or_else(|| snake.strip_prefix("_"))
        .unwrap_or(&snake);
    
    stripped.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synonym_exact_match() {
        assert!(synonym_match("user", "person") > 0.9);
        assert!(synonym_match("user", "account") > 0.9);
        assert!(synonym_match("email", "mail") > 0.9);
    }

    #[test]
    fn test_synonym_normalization() {
        assert!(synonym_match("userId", "user_id") > 0.9);
        assert!(synonym_match("firstName", "first_name") > 0.9);
    }

    #[test]
    fn test_synonym_no_match() {
        assert!(synonym_match("email", "status") < 0.1);
        assert!(synonym_match("user", "timestamp") < 0.1);
    }
}
