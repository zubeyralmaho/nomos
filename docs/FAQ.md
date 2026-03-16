# Frequently Asked Questions (FAQ)

## General Questions

### What is Nomos?

Nomos is a **transparent proxy** that automatically heals API schema drift. When external APIs change their JSON response format, Nomos transforms responses back to what your client expects — without any code changes.

### Why is it called "Nomos"?

"Nomos" (νόμος) is Greek for "law" or "custom". The name reflects the project's core philosophy: enforcing a "law" that responses must conform to the expected schema, and the "custom" of how your client expects data to look.

### Who should use Nomos?

Nomos is ideal for teams that:
- Depend on third-party APIs that change without warning
- Maintain legacy clients that can't be easily updated
- Need a buffer between API versions during migrations
- Want to reduce downtime caused by API drift

### Is Nomos production-ready?

Yes! Nomos is designed for production use with:
- Sub-millisecond latency (p99 < 0.3ms)
- 100% success rate under load
- Zero-downtime hot-swap of healing logic
- Comprehensive testing (46 unit tests + chaos suite)

---

## Technical Questions

### How does Nomos detect schema drift?

Nomos uses **schema fingerprinting**:

1. You define the expected schema (field names, types, structure)
2. Nomos compares incoming responses against this schema
3. If fields don't match, it applies NLP algorithms to find the best matches
4. It transforms the response to match the expected format

### What types of schema changes can Nomos heal?

| Change Type | Example | Support |
|-------------|---------|---------|
| Field renaming | `user_id` → `userId` | ✅ Full |
| CamelCase/snake_case | `userName` ↔ `user_name` | ✅ Full |
| Nested structures | `name` → `user.profile.name` | ✅ Full |
| Type coercion | `"123"` → `123` | ✅ Full |
| Missing fields | Add default values | ✅ Full |
| Typos | `emial` → `email` | ✅ Full |
| Abbreviations | `desc` → `description` | ✅ Full |
| Semantic synonyms | `person` → `user` | ✅ Full |
| Array structure changes | | ⚠️ Partial |
| Complete restructuring | | ⚠️ Limited |

### What NLP algorithms does Nomos use?

Nomos uses a **weighted ensemble** of 4 algorithms:

| Algorithm | Weight | Best For |
|-----------|--------|----------|
| Synonym Dictionary | 35% | Semantic equivalence |
| Jaro-Winkler | 25% | Prefix matching |
| Levenshtein | 20% | Typo detection |
| TF-IDF N-grams | 20% | Structural similarity |

### What's the confidence threshold?

The confidence threshold (default: 0.70) determines when a field match is accepted:

- **0.60** - Aggressive: catches more drift, higher false positive risk
- **0.70** - Balanced: good accuracy with low false positives
- **0.85** - Conservative: very accurate, may miss some valid matches

Adjust via environment variable:
```bash
NOMOS_CONFIDENCE_THRESHOLD=0.75 ./nomos-core
```

### How fast is Nomos?

Nomos is engineered for sub-millisecond latency:

| Metric | Target | Achieved |
|--------|--------|----------|
| p50 Latency | < 500µs | 71µs |
| p99 Latency | < 1ms | 223µs |
| Peak Throughput | > 1000 RPS | 5,146 RPS |

### Does Nomos support HTTPS?

Nomos proxies HTTP traffic. For HTTPS:

1. **Terminate TLS upstream**: Use HTTPS when connecting to the upstream API
2. **Terminate TLS downstream**: Put Nomos behind nginx/HAProxy with TLS
3. **mTLS**: Configure in your reverse proxy layer

### Can I customize the healing logic?

Yes! Nomos supports **WASM healer modules** for custom logic:

1. Write your healer in Rust (or any WASM-compatible language)
2. Compile to WASM: `./build-wasm.sh`
3. Deploy with hot-swap (no downtime)

### What's the eBPF feature?

eBPF (Extended Berkeley Packet Filter) enables kernel-level packet classification:

- **Fast-path bypass**: Healthy routes skip userspace entirely
- **Circuit breaker**: Drop blocked traffic at kernel level
- **Lower CPU usage**: Kernel handles classification

Requires: Linux 5.4+, root access.

---

## Operational Questions

### How do I monitor Nomos?

**Health endpoint:**
```bash
curl http://localhost:8081/health
```

**Metrics endpoint:**
```bash
curl http://localhost:8081/metrics
```

**Real-time TUI:**
```bash
python3 tests/nomos_tui.py
```

### How do I configure the expected schema?

Via the control plane API:

```bash
curl -X POST http://localhost:8081/schema \
  -H "Content-Type: application/json" \
  -d '{
    "schema": {
      "type": "object",
      "properties": {
        "user_id": {"type": "integer"},
        "user_name": {"type": "string"}
      }
    }
  }'
```

### How do I update configuration at runtime?

```bash
curl -X PATCH http://localhost:8081/config \
  -H "Content-Type: application/json" \
  -d '{
    "healing": {
      "confidence_threshold": 0.80
    }
  }'
```

### How do I test healing before production?

Use the test endpoint:

```bash
curl -X POST http://localhost:8081/healing/test \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "userId": "123",
      "userName": "Alice"
    }
  }'
```

### What headers does Nomos add?

| Header | Description |
|--------|-------------|
| `X-Nomos-Healed` | `true` if healing was applied |
| `X-Nomos-Latency-us` | Healing time in microseconds |
| `X-Nomos-Healing-Ops` | Number of operations applied |
| `X-Nomos-Version` | Nomos version |

### How do I scale Nomos?

**Horizontal scaling:**
- Deploy multiple instances behind a load balancer
- Each instance is stateless (except for cached patterns)

**Vertical scaling:**
```bash
TOKIO_WORKER_THREADS=8 ./nomos-core
```

### How do I handle high load?

1. Build with optimizations:
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo build --release
   ```

2. Enable eBPF (if on Linux):
   ```bash
   sudo ./nomos-core --enable-ebpf
   ```

3. Tune kernel parameters (see [Deployment Guide](DEPLOYMENT.md))

---

## Troubleshooting

### Nomos returns the original (unhealed) response

**Possible causes:**
1. Confidence threshold too high
2. Schema not loaded
3. Healing disabled

**Solutions:**
1. Lower threshold: `NOMOS_CONFIDENCE_THRESHOLD=0.60`
2. Check schema: `curl http://localhost:8081/schema`
3. Enable healing: `curl -X PATCH http://localhost:8081/config -d '{"healing":{"enabled":true}}'`

### High latency (> 1ms)

**Possible causes:**
1. Running debug build
2. Cold cache
3. Complex nested structures

**Solutions:**
1. Rebuild: `cargo build --release`
2. Warm up with traffic
3. Pre-register patterns

### "Connection refused" errors

**Possible causes:**
1. Nomos not running
2. Wrong port
3. Firewall blocking

**Solutions:**
1. Start Nomos: `./target/release/nomos-core`
2. Check port: `ss -tlnp | grep 8080`
3. Check firewall: `ufw status`

### Memory usage keeps growing

**Possible causes:**
1. Memory leak (report as bug)
2. Cached patterns accumulating

**Solutions:**
1. Monitor with: `python3 tests/memory_monitor.py`
2. Restart periodically (if leak confirmed)
3. Report issue on GitHub

### NLP matching wrong fields

**Possible causes:**
1. Ambiguous field names
2. Threshold too low
3. Missing synonyms

**Solutions:**
1. Register explicit patterns
2. Increase threshold: `NOMOS_CONFIDENCE_THRESHOLD=0.85`
3. Add to synonym dictionary (requires recompile)

---

## Integration Questions

### Can I use Nomos with GraphQL?

Nomos currently focuses on REST/JSON APIs. GraphQL support is on the roadmap.

### Can I use Nomos with gRPC?

Not directly. Nomos handles JSON over HTTP. For gRPC, consider a transcoding proxy first.

### Does Nomos work with websockets?

Nomos proxies HTTP request/response traffic. For websocket support, use a dedicated websocket proxy.

### Can I chain multiple Nomos instances?

Yes, but it's not recommended. Use a single Nomos instance with comprehensive healing rules instead.

### How do I integrate with service mesh (Istio, Linkerd)?

Deploy Nomos as a sidecar or dedicated service:

```yaml
# Kubernetes sidecar example
containers:
- name: nomos
  image: nomos:latest
  ports:
  - containerPort: 8080
- name: app
  image: myapp:latest
  env:
  - name: API_URL
    value: "http://localhost:8080"  # Through Nomos
```

---

## Contributing Questions

### How can I contribute?

1. Read [CONTRIBUTING.md](../CONTRIBUTING.md)
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

### How do I add a new NLP algorithm?

1. Create `nomos-core/src/nlp/your_algorithm.rs`
2. Implement similarity function
3. Add to `nlp/mod.rs`
4. Include in ensemble (`engine/confidence.rs`)
5. Write tests

### How do I run tests?

```bash
# Unit tests
cargo test -p nomos-core

# Lint check
cargo clippy -p nomos-core

# Stress test
python3 tests/stress_test.py

# Full chaos suite
python3 tests/run_chaos_suite.py
```

---

## More Questions?

- **GitHub Issues**: [github.com/zubeyralmaho/nomos/issues](https://github.com/zubeyralmaho/nomos/issues)
- **Documentation**: See the [docs folder](.)
- **Source Code**: Inline documentation throughout
