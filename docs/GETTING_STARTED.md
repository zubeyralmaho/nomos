# Getting Started with Nomos

This guide will help you get Nomos up and running in minutes.

## What is Nomos?

Nomos is a **transparent proxy** that automatically fixes API schema drift. When external APIs change their JSON response structure, Nomos "heals" the response back to the format your client expects — without code changes or redeployments.

## Prerequisites

Before you begin, ensure you have the following installed:

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Rust | 1.82+ | `rustc --version` |
| Cargo | Latest | `cargo --version` |
| Python | 3.8+ | `python3 --version` |
| Git | Latest | `git --version` |

### Optional (for eBPF features)

- Linux kernel 5.4+ with BTF support
- `bpf-linker` for building eBPF programs
- Root access for loading XDP programs

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/zubeyralmaho/nomos.git
cd nomos
```

### 2. Build the Core Components

```bash
# Build in release mode (recommended for production)
cargo build --release -p nomos-core

# Or build in debug mode for development
cargo build -p nomos-core
```

### 3. Verify the Build

```bash
# Run unit tests to verify everything works
cargo test -p nomos-core
```

You should see all 46 tests pass:
```
running 46 tests
...
test result: ok. 46 passed; 0 failed
```

## Quick Start

### Step 1: Start a Test Upstream Server

The test suite includes a mock server that simulates various API drift scenarios:

```bash
# Start the upstream server with API v2 drift (simulating schema changes)
python3 tests/upstream_server.py --port 9090 --drift-mode v2 &
```

### Step 2: Start Nomos Proxy

```bash
# Start Nomos with info-level logging
RUST_LOG=info ./target/release/nomos-core
```

The proxy will start and listen on:
- **Port 8080**: Proxy port (send your requests here)
- **Port 8081**: Control plane API (for monitoring and configuration)

### Step 3: Test the Healing

**Without Nomos (direct to drifted API):**
```bash
curl http://localhost:9090/api/user
# Returns drifted response:
# {"userId": "123", "userName": "Alice", ...}
```

**Through Nomos (healed response):**
```bash
curl http://localhost:8080/api/user
# Returns healed response:
# {"user_id": "123", "user_name": "Alice", ...}
```

### Step 4: Check Healing Status

Nomos adds diagnostic headers to every response:

```bash
curl -v http://localhost:8080/api/user 2>&1 | grep "X-Nomos"
# X-Nomos-Healed: true
# X-Nomos-Latency-us: 89
# X-Nomos-Healing-Ops: 5
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NOMOS_UPSTREAM_HOST` | Target API hostname | `localhost` |
| `NOMOS_UPSTREAM_PORT` | Target API port | `9090` |
| `NOMOS_LISTEN_PORT` | Proxy listen port | `8080` |
| `NOMOS_CONFIDENCE_THRESHOLD` | Minimum match confidence | `0.70` |
| `RUST_LOG` | Log level (trace/debug/info/warn/error) | `info` |

### Example: Connect to a Real API

```bash
NOMOS_UPSTREAM_HOST=api.example.com \
NOMOS_UPSTREAM_PORT=443 \
NOMOS_LISTEN_PORT=8080 \
./target/release/nomos-core
```

## Testing Different Drift Modes

Nomos handles various types of schema drift. Test them with the mock server:

```bash
# CamelCase conversion (user_name → userName)
python3 tests/upstream_server.py --drift-mode camel &

# Deep nesting (name → data.response.user.profile.name)
python3 tests/upstream_server.py --drift-mode deep &

# Typos (email → emial)
python3 tests/upstream_server.py --drift-mode typo &

# All patterns combined
python3 tests/upstream_server.py --drift-mode mixed &
```

## Monitoring

### Control Plane API

Check the proxy status:
```bash
curl http://localhost:8081/health
# {"status": "healthy", "uptime_seconds": 3600, "version": "1.0.0"}

curl http://localhost:8081/metrics
# {"latency_p50_ms": 0.12, "latency_p99_ms": 0.22, ...}
```

### Real-Time TUI

For live monitoring, use the Terminal User Interface:

```bash
cd tests
pip install -r requirements.txt
python3 nomos_tui.py
```

## Running Tests

### Unit Tests

```bash
# All tests
cargo test -p nomos-core

# Specific module
cargo test -p nomos-core nlp
cargo test -p nomos-core engine
```

### Stress Test

```bash
cd tests
python3 stress_test.py --rps 5000 --duration 30
```

### Chaos Suite

```bash
python3 tests/chaos_suite.py
```

## Troubleshooting

### "Connection refused" on port 8080

The proxy isn't running. Start it with:
```bash
./target/release/nomos-core
```

### "No upstream server" errors

The upstream server isn't running. Start the test server:
```bash
python3 tests/upstream_server.py &
```

### High latency (> 1ms)

Check if you're running in release mode:
```bash
cargo build --release -p nomos-core
```

Enable SIMD acceleration:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Tests fail

Ensure all dependencies are installed:
```bash
cd tests
pip install -r requirements.txt
```

## Next Steps

- **[Architecture Guide](ARCHITECTURE.md)** - Understand how Nomos works internally
- **[API Reference](API.md)** - Full control plane API documentation
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment instructions
- **[Contributing](../CONTRIBUTING.md)** - Contribute to Nomos

---

*Need help? Open an issue on [GitHub](https://github.com/zubeyralmaho/nomos/issues).*
