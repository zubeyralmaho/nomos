# Nomos API Documentation

Control plane API for monitoring, configuration, and schema management.

**Base URL:** `http://127.0.0.1:8081`

---

## Health & Status

### GET /health

Check proxy health and get basic metrics.

**Response:**
```json
{
    "status": "healthy",
    "uptime_seconds": 3600,
    "version": "1.0.0"
}
```

---

### GET /metrics

Get detailed performance metrics.

**Response:**
```json
{
    "latency_p50_ms": 0.12,
    "latency_p99_ms": 0.22,
    "throughput_rps": 5146,
    "success_rate": 100.0,
    "total_requests": 1000000,
    "healed_requests": 12847,
    "active_connections": 42
}
```

---

### GET /status

Get current proxy status and configuration.

**Response:**
```json
{
    "proxy_port": 8080,
    "control_port": 8081,
    "healing_enabled": true,
    "drift_mode": "strict",
    "schema_loaded": true,
    "nlp_algorithms": ["levenshtein", "jaro_winkler", "ngram", "soundex", "metaphone"]
}
```

---

## Schema Management

### GET /schema

Get current schema definition.

**Response:**
```json
{
    "schema": {
        "type": "object",
        "properties": {
            "id": { "type": "integer" },
            "username": { "type": "string" },
            "email": { "type": "string" }
        },
        "required": ["id", "username"]
    }
}
```

---

### POST /schema

Update schema definition.

**Request:**
```json
{
    "schema": {
        "type": "object",
        "properties": {
            "id": { "type": "integer" },
            "name": { "type": "string" }
        }
    }
}
```

**Response:**
```json
{
    "success": true,
    "message": "Schema updated"
}
```

---

### POST /schema/validate

Validate JSON against current schema.

**Request:**
```json
{
    "data": {
        "id": 123,
        "username": "john"
    }
}
```

**Response:**
```json
{
    "valid": true,
    "errors": []
}
```

---

## Healing Operations

### GET /healing/stats

Get healing statistics.

**Response:**
```json
{
    "total_healed": 12847,
    "operations": {
        "rename": 8234,
        "coerce": 3156,
        "default": 1457
    },
    "confidence_avg": 0.92,
    "patterns_applied": 165
}
```

---

### GET /healing/history

Get recent healing operations.

**Query Parameters:**
- `limit` (optional): Maximum number of entries (default: 100)
- `offset` (optional): Pagination offset

**Response:**
```json
{
    "entries": [
        {
            "timestamp": "2025-01-15T12:34:56Z",
            "original_field": "user_name",
            "healed_field": "username",
            "operation": "rename",
            "confidence": 0.95,
            "algorithm": "jaro_winkler"
        }
    ],
    "total": 12847
}
```

---

### POST /healing/test

Test healing on sample data.

**Request:**
```json
{
    "data": {
        "user_name": "john",
        "created_date": "2025-01-15"
    }
}
```

**Response:**
```json
{
    "original": {
        "user_name": "john",
        "created_date": "2025-01-15"
    },
    "healed": {
        "username": "john",
        "createdAt": "2025-01-15"
    },
    "operations": [
        {
            "field": "user_name",
            "healed_to": "username",
            "operation": "rename",
            "confidence": 0.95
        }
    ]
}
```

---

## NLP Engine

### POST /nlp/compare

Compare two field names using all NLP algorithms.

**Request:**
```json
{
    "source": "user_name",
    "target": "username"
}
```

**Response:**
```json
{
    "levenshtein": 0.875,
    "jaro_winkler": 0.933,
    "ngram": 0.714,
    "soundex": 0.750,
    "metaphone": 0.800,
    "ensemble": 0.851,
    "match": true
}
```

---

### GET /nlp/algorithms

Get information about available NLP algorithms.

**Response:**
```json
{
    "algorithms": [
        {
            "name": "levenshtein",
            "description": "Edit distance between strings",
            "complexity": "O(mn)",
            "weight": 0.25
        },
        {
            "name": "jaro_winkler",
            "description": "Character matching with prefix bonus",
            "complexity": "O(mn)",
            "weight": 0.30
        },
        {
            "name": "ngram",
            "description": "Substring overlap comparison",
            "complexity": "O(n)",
            "weight": 0.20
        },
        {
            "name": "soundex",
            "description": "Phonetic encoding similarity",
            "complexity": "O(n)",
            "weight": 0.10
        },
        {
            "name": "metaphone",
            "description": "Advanced phonetic matching",
            "complexity": "O(n)",
            "weight": 0.15
        }
    ]
}
```

---

## Configuration

### GET /config

Get current configuration.

**Response:**
```json
{
    "healing": {
        "enabled": true,
        "confidence_threshold": 0.80,
        "max_candidates": 5,
        "drift_mode": "strict"
    },
    "performance": {
        "buffer_size": 65536,
        "max_connections": 1000,
        "timeout_ms": 30000
    },
    "nlp": {
        "weights": {
            "levenshtein": 0.25,
            "jaro_winkler": 0.30,
            "ngram": 0.20,
            "soundex": 0.10,
            "metaphone": 0.15
        }
    }
}
```

---

### PATCH /config

Update configuration values.

**Request:**
```json
{
    "healing": {
        "confidence_threshold": 0.85
    }
}
```

**Response:**
```json
{
    "success": true,
    "updated": ["healing.confidence_threshold"]
}
```

---

## Patterns

### GET /patterns

Get registered healing patterns.

**Response:**
```json
{
    "patterns": [
        {
            "source": "user_name",
            "target": "username",
            "operation": "rename",
            "hits": 1234
        },
        {
            "source": "created_date",
            "target": "createdAt",
            "operation": "rename",
            "hits": 567
        }
    ],
    "total": 165
}
```

---

### POST /patterns

Register a new healing pattern.

**Request:**
```json
{
    "source": "usr_id",
    "target": "userId",
    "operation": "rename"
}
```

**Response:**
```json
{
    "success": true,
    "pattern_id": "p_abc123"
}
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
    "error": true,
    "code": "INVALID_SCHEMA",
    "message": "Schema validation failed",
    "details": {
        "field": "properties.id",
        "reason": "Invalid type specification"
    }
}
```

**Common Error Codes:**
- `INVALID_REQUEST` - Malformed request body
- `INVALID_SCHEMA` - Schema validation failed
- `NOT_FOUND` - Resource not found
- `INTERNAL_ERROR` - Server error
- `RATE_LIMITED` - Too many requests

---

## Rate Limiting

Control plane endpoints are rate limited:
- `/health`, `/metrics`: No limit
- Other endpoints: 100 requests/minute

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705320000
```

---

## WebSocket: /ws

Real-time metrics and healing events.

**Connect:**
```javascript
const ws = new WebSocket('ws://127.0.0.1:8081/ws');
```

**Message Types:**

Metrics update:
```json
{
    "type": "metrics",
    "data": {
        "latency_p99_ms": 0.22,
        "throughput_rps": 5146
    }
}
```

Healing event:
```json
{
    "type": "healing",
    "data": {
        "original_field": "user_name",
        "healed_field": "username",
        "confidence": 0.95
    }
}
```

---

## Examples

### cURL

```bash
# Health check
curl http://127.0.0.1:8081/health

# Get metrics
curl http://127.0.0.1:8081/metrics

# Test healing
curl -X POST http://127.0.0.1:8081/healing/test \
  -H "Content-Type: application/json" \
  -d '{"data": {"user_name": "john"}}'

# Compare field names
curl -X POST http://127.0.0.1:8081/nlp/compare \
  -H "Content-Type: application/json" \
  -d '{"source": "user_name", "target": "username"}'

# Update config
curl -X PATCH http://127.0.0.1:8081/config \
  -H "Content-Type: application/json" \
  -d '{"healing": {"confidence_threshold": 0.85}}'
```

### JavaScript

```javascript
// Fetch metrics
const response = await fetch('http://127.0.0.1:8081/metrics');
const metrics = await response.json();
console.log(`RPS: ${metrics.throughput_rps}`);

// Test healing
const healResponse = await fetch('http://127.0.0.1:8081/healing/test', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        data: { user_name: 'john', created_date: '2025-01-15' }
    })
});
const result = await healResponse.json();
console.log('Healed:', result.healed);
```

### Python

```python
import requests

# Health check
health = requests.get('http://127.0.0.1:8081/health').json()
print(f"Status: {health['status']}")

# NLP comparison
compare = requests.post('http://127.0.0.1:8081/nlp/compare', json={
    'source': 'user_name',
    'target': 'username'
}).json()
print(f"Similarity: {compare['ensemble']:.2%}")
```
