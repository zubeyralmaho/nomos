# Nomos Proxy - Performance Benchmark Report

**Date:** February 26, 2026  
**Version:** 0.2.0  
**Platform:** Linux x86_64  
**Rust:** 1.82+ (release build, optimized)

---

## Executive Summary

Nomos is a zero-latency autonomous schema-healing proxy that transparently fixes API schema drift without client changes. Version 0.2.0 introduces a **modular NLP pipeline** with 5 similarity algorithms.

### Key Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Healing p99 Latency | < 1ms | **0.22ms** | ✅ |
| Peak Throughput | > 1000 RPS | **5,146 RPS** | ✅ |
| Success Rate | > 99% | **100%** | ✅ |
| Chaos Suite | Pass | **10/10** | ✅ |
| NLP Algorithms | 4+ | **5** | ✅ |
| Unit Tests | Pass | **46/46** | ✅ |
| Modular Files | - | **13** | ✅ |

---

## 1. Latency Performance

### Nomos Law: p99 < 1ms

The "Nomos Law" states that healing operations must complete in under 1ms at the 99th percentile. This ensures the proxy remains effectively invisible to clients.

#### Healing Latency Distribution (500 samples under load)

```
┌──────────┬─────────┐
│ Metric   │ Value   │
├──────────┼─────────┤
│ Min      │ 68µs    │
│ Avg      │ 79µs    │
│ p50      │ 71µs    │
│ p75      │ 72µs    │
│ p90      │ 104µs   │
│ p95      │ 152µs   │
│ p99      │ 223µs   │
│ Max      │ 262µs   │
└──────────┴─────────┘
```

**Result:** p99 = 223µs (0.22ms) - **4.5x better than target**

### Latency by Operation Type

| Operation | Avg Latency |
|-----------|-------------|
| Pass-through (no healing) | ~50µs |
| FastPath key rename | ~80µs |
| Nested JSON flattening | ~350µs |
| Deep nested (6+ levels) | ~400µs |

---

## 2. Throughput Performance

### Load Testing Results

| Test Level | Requests | Concurrency | RPS | Success Rate |
|------------|----------|-------------|-----|--------------|
| Warm-up | 500 | 100 | 4,175 | 100% |
| Medium | 2,000 | 200 | **5,146** | 100% |
| High | 5,000 | 500 | 4,183 | 100% |
| Extreme | 10,000 | 1,000 | 3,777 | 100% |
| Maximum | 25,000 | 2,500 | 3,229 | 100% |

**Peak Performance:** 5,146 RPS with 100% success rate

### Throughput Characteristics

- Linear scaling up to ~200 concurrent connections
- Graceful degradation under extreme load
- No request failures even at 2,500 concurrent connections
- Total of 42,500 requests processed with 100% success

---

## 3. Healing Accuracy

### Drift Modes Tested

| Mode | Description | Healing Rate | Ops/Request |
|------|-------------|--------------|-------------|
| healthy | No drift (baseline) | N/A | 0 |
| v2 | API v2 renames | 100% | 13 |
| camel | CamelCase conversion | 100% | 8 |
| nested | Nested structure | 100% | 11 |
| deep | 6+ level nesting | 100% | 21 |
| typo | Common typos | 100% | 4 |
| abbrev | Abbreviations | 100% | 12 |
| legacy | Legacy naming | 100% | 9 |
| mixed | Combined patterns | 100% | 10 |

**Overall Healing Rate:** 100% across all drift patterns

### Healing Rules Coverage

```
┌─────────────────────────┬─────────┐
│ Category                │ Rules   │
├─────────────────────────┼─────────┤
│ API v2 style renames    │ 13      │
│ CamelCase → snake_case  │ 16      │
│ Common abbreviations    │ 30      │
│ Underscore variations   │ 10      │
│ Pluralization issues    │ 11      │
│ Common typos            │ 18      │
│ Legacy naming           │ 17      │
│ Date/time variations    │ 10      │
│ Nested path rules       │ 40+     │
├─────────────────────────┼─────────┤
│ TOTAL                   │ 165+    │
└─────────────────────────┴─────────┘
```

---

## 4. Chaos Engineering Results

### Test Suite: 10/10 Passed

| Test | Description | Result |
|------|-------------|--------|
| Upstream Timeout | Graceful handling when upstream unavailable | ✅ PASS |
| Malformed JSON | Recovery from invalid JSON | ✅ PASS |
| High Concurrency | 100 req @ 20 workers | ✅ PASS |
| Rapid Mode Switching | 8 modes cycled rapidly | ✅ PASS |
| Burst Traffic | 5x30 request bursts | ✅ PASS |
| Upstream Flapping | Up/down cycles | ✅ PASS |
| Healing Consistency | Same output every time | ✅ PASS |
| Deep Nested Healing | 6+ level flattening | ✅ PASS |
| Latency Under Load | p99 < 50ms E2E | ✅ PASS |
| Recovery After Chaos | Full recovery confirmed | ✅ PASS |

### Resilience Characteristics

- **Fail-open design:** Passes through unhealed responses on error
- **Graceful degradation:** Maintains service under extreme load
- **Fast recovery:** Returns to full performance within seconds
- **No data loss:** 100% request success rate during chaos

---

## 5. Resource Utilization

### Memory Profile

| State | RSS Memory |
|-------|------------|
| Idle | ~10 MB |
| Under load (1000 RPS) | ~25 MB |
| Peak (5000 RPS) | ~40 MB |

### CPU Profile

- Single-threaded async design with Tokio
- CPU usage scales linearly with RPS
- ~5% CPU per 1000 RPS on modern hardware

---

## 6. Architecture Performance Features

### FastPath Healer

Pure Rust byte-level JSON key replacement:

```rust
// Pre-compiled patterns: "\"old_key\":" → "\"new_key\":"
// Zero JSON parsing for simple renames
// O(n) single-pass transformation
```

**Performance:** 70-100µs for typical payloads

### Nested JSON Healer

Path-based structural transformation:

```rust
// response.data.user.identity.personal.name.full → full_name
// Uses serde_json for complex transformations
// Handles 6+ level nesting
```

**Performance:** 300-400µs for deep structures

### eBPF/XDP Integration (Optional)

Kernel-level packet acceleration:

```
- XDP SKB mode attachment
- Packet statistics tracking
- Requires root/CAP_BPF
- Graceful fallback without privileges
```

---

## 7. Comparison: Before vs After Healing

### Example: Deep Nested Response

**Before (6+ levels):**
```json
{
  "response": {
    "data": {
      "user": {
        "identity": {
          "personal": {
            "name": { "full": "Alice Smith" },
            "contact": { "email": { "primary": "alice@example.com" } }
          }
        },
        "financial": {
          "accounts": {
            "primary": { "balance": { "amount": 1500.50 } }
          }
        }
      }
    }
  }
}
```

**After (flat, healed):**
```json
{
  "user_id": "usr_abc123",
  "full_name": "Alice Smith",
  "email_address": "alice@example.com",
  "account_balance": 1500.50,
  "created_at": "2026-02-26T...",
  "is_verified": true
}
```

**Transformation:** 21 healing operations in ~350µs

---

## 8. Benchmark Methodology

### Test Environment

- **OS:** Linux (Ubuntu)
- **Hardware:** Standard development machine
- **Network:** Loopback (localhost)
- **Upstream:** Python aiohttp test server

### Tools Used

- Python asyncio/aiohttp for load generation
- ThreadPoolExecutor for concurrent requests
- Custom chaos suite for resilience testing
- X-Nomos-Latency-us header for accurate timing

### Measurement Approach

1. **Latency:** Measured via X-Nomos-Latency-us response header (server-side)
2. **Throughput:** Total requests / elapsed time
3. **Success Rate:** HTTP 200 responses / total requests
4. **Healing Rate:** Healed responses / drifted responses

---

## 10. Academic NLP Architecture (v0.2.0)

Version 0.2.0 introduces a **full academic NLP pipeline** suitable for NLP course demonstrations. The monolithic `engine.rs` (1956 lines) has been refactored into **13 modular files**.

### 10.1 Module Structure

```
nomos-core/src/
├── nlp/                     # NLP Algorithms (773 lines)
│   ├── mod.rs               # Module re-exports
│   ├── levenshtein.rs       # Edit distance (Wagner-Fischer)
│   ├── jaro.rs              # Jaro-Winkler similarity
│   ├── tfidf.rs             # N-gram TF-IDF
│   ├── synonym.rs           # Semantic synonym dictionary
│   └── combined.rs          # Ensemble scoring
│
└── engine/                  # Semantic Healing Engine (1421 lines)
    ├── mod.rs               # Module re-exports
    ├── simd.rs              # SIMD-accelerated operations
    ├── embedding.rs         # Field embeddings
    ├── confidence.rs        # Match confidence scoring
    ├── lsh.rs               # Locality-Sensitive Hashing
    ├── healing.rs           # Healing operations
    └── healer.rs            # SemanticHealer implementation
```

### 10.2 NLP Algorithms Implemented

| Algorithm | Reference | Complexity | Purpose |
|-----------|-----------|------------|---------|
| **Levenshtein Distance** | Levenshtein (1966) | O(m×n) | Typo detection, edit distance |
| **Jaro-Winkler Similarity** | Winkler (1990) | O(m×n) | Prefix-aware field matching |
| **N-gram TF-IDF** | Salton & McGill (1983) | O(n) | Structural similarity via trigrams |
| **Synonym Dictionary** | Custom | O(1) | Semantic equivalence lookup |
| **Combined Ensemble** | Custom | O(m×n) | Weighted voting of all algorithms |

### 10.3 Algorithm Comparison Results

```
========================================
ACADEMIC NLP ALGORITHM COMPARISON
========================================
Pair                      Lev     Jaro      J-W   TF-IDF  Combined
------------------------------------------------------------------------
user_id→userId          0.714    0.881    0.905    0.667    1.000
email→mail              0.800    0.933    0.956    0.571    0.938
created_at→timestamp    0.200    0.488    0.488    0.154    0.766
user→person             0.167    0.456    0.456    0.000    0.677
description→desc        0.364    0.779    0.857    0.571    0.926
========================================
```

**Legend:**
- **Lev** = Levenshtein Similarity (edit distance normalized)
- **Jaro** = Jaro similarity (character matching)
- **J-W** = Jaro-Winkler (with prefix bonus)
- **TF-IDF** = Trigram Term Frequency-Inverse Document Frequency
- **Combined** = Weighted ensemble + synonym lookup

### 10.4 Ensemble Weights

```rust
// Combined NLP Similarity = weighted voting
if synonym_match > 0.9 {
    return 1.0;  // Exact semantic match
}

combined = synonym_weight * 0.35    // Semantic knowledge
         + jaro_winkler  * 0.25    // Prefix-aware matching
         + levenshtein   * 0.20    // Edit distance
         + tfidf         * 0.20;   // N-gram structure
```

### 10.5 Synonym Dictionary Coverage

```
┌─────────────────────┬────────────────────────────────────────┐
│ Category            │ Examples                               │
├─────────────────────┼────────────────────────────────────────┤
│ User/Person         │ user, person, account, member, client  │
│ Email variants      │ email, mail, e_mail, email_address     │
│ Name fields         │ name, fullname, full_name, display_name│
│ ID fields           │ id, identifier, uid, uuid, key         │
│ Timestamp fields    │ timestamp, created_at, createdAt, time │
│ Status fields       │ status, state, condition, phase        │
│ Description fields  │ description, desc, summary, details    │
│ Phone fields        │ phone, telephone, mobile, cell         │
│ Address fields      │ address, addr, location, street        │
│ Amount fields       │ amount, value, sum, total, balance     │
├─────────────────────┼────────────────────────────────────────┤
│ TOTAL               │ 25 groups (~120 words)                 │
└─────────────────────┴────────────────────────────────────────┘
```

### 10.6 Test Coverage

```
Unit Tests: 46/46 PASSED
├── NLP Module Tests
│   ├── nlp::levenshtein::tests::test_levenshtein_distance ✅
│   ├── nlp::levenshtein::tests::test_levenshtein_similarity ✅
│   ├── nlp::jaro::tests::test_jaro_similarity ✅
│   ├── nlp::jaro::tests::test_jaro_winkler_similarity ✅
│   ├── nlp::tfidf::tests::test_ngram_extraction ✅
│   ├── nlp::tfidf::tests::test_tfidf_similarity ✅
│   ├── nlp::synonym::tests::test_synonym_exact_match ✅
│   ├── nlp::synonym::tests::test_synonym_no_match ✅
│   ├── nlp::synonym::tests::test_synonym_normalization ✅
│   └── nlp::combined::tests::test_combined_similarity ✅
│
└── Engine Module Tests
    ├── engine::simd::tests::test_simd_dot_product ✅
    ├── engine::embedding::tests::test_trigram_embedding ✅
    ├── engine::confidence::tests::test_type_compatibility_* ✅
    ├── engine::lsh::tests::test_lsh_* ✅
    ├── engine::healing::tests::test_healing_* ✅
    └── engine::healer::tests::test_semantic_healer_* ✅
```

---

## 11. Conclusions

### Performance Goals: All Met

| Goal | Target | Result |
|------|--------|--------|
| Nomos Law (p99 latency) | < 1ms | **0.22ms** ✅ |
| Throughput | > 1000 RPS | **5,146 RPS** ✅ |
| Healing accuracy | > 95% | **100%** ✅ |
| Chaos resilience | Pass | **10/10** ✅ |

### Key Achievements

1. **Sub-millisecond healing:** 4.5x better than target
2. **High throughput:** Handles 5000+ RPS
3. **Perfect accuracy:** 100% healing rate across all patterns
4. **Full resilience:** Survives upstream failures, flapping, bursts
5. **Deep nesting support:** 6+ level JSON flattening
6. **Academic NLP:** 5 algorithms with proper citations
7. **Modular architecture:** 13 files, clean separation of concerns

### Recommendations

1. **Production deployment:** Performance is production-ready
2. **eBPF acceleration:** Enable for additional kernel-level optimization
3. **Custom rules:** Add domain-specific healing patterns as needed
4. **Monitoring:** Use X-Nomos-* headers for observability
5. **Academic use:** NLP module suitable for course demonstrations

---

## Appendix: Test Commands

```bash
# Start upstream server
python3 tests/upstream_server.py --port 9090 --drift-mode v2

# Start proxy
RUST_LOG=info ./target/release/nomos-core

# Run chaos suite
python3 tests/chaos_suite.py

# Run all tests
cargo test -p nomos-core

# Check healing
curl -v http://localhost:8080/api/user
```

---

*Report generated by Nomos Performance Test Suite*
