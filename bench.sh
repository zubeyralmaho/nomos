#!/usr/bin/env bash
#
# Nomos Proxy Baseline Benchmark
#
# This script measures the pass-through latency of the Nomos proxy
# to verify we meet the sub-1ms overhead requirement.
#
# Prerequisites:
#   - Rust toolchain installed
#   - `wrk` or `hey` for HTTP benchmarking
#   - Python 3 for mock server (or your own target server)
#
# Usage:
#   ./bench.sh [requests] [concurrency]
#

set -euo pipefail

# Configuration
PROXY_PORT="${PROXY_PORT:-8080}"
TARGET_PORT="${TARGET_PORT:-9090}"
REQUESTS="${1:-10000}"
CONCURRENCY="${2:-50}"
WARMUP_REQUESTS=1000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for required tools
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing=()
    
    if ! command -v cargo &> /dev/null; then
        missing+=("cargo (Rust toolchain)")
    fi
    
    # Check for at least one HTTP benchmark tool
    if command -v wrk &> /dev/null; then
        BENCH_TOOL="wrk"
    elif command -v hey &> /dev/null; then
        BENCH_TOOL="hey"
    elif command -v ab &> /dev/null; then
        BENCH_TOOL="ab"
    else
        missing+=("wrk, hey, or ab (HTTP benchmark tool)")
    fi
    
    if ! command -v python3 &> /dev/null; then
        missing+=("python3 (for mock server)")
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing dependencies:"
        for dep in "${missing[@]}"; do
            echo "  - $dep"
        done
        exit 1
    fi
    
    log_success "Using $BENCH_TOOL for benchmarking"
}

# Start a simple mock HTTP server
start_mock_server() {
    log_info "Starting mock server on port $TARGET_PORT..."
    
    python3 -c "
import http.server
import socketserver
import json

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logging
    
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response = {
            'status': 'ok',
            'timestamp': 1234567890,
            'data': {
                'user_id': 'usr_12345',
                'name': 'Test User',
                'email': 'test@example.com'
            }
        }
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        self.do_GET()

with socketserver.TCPServer(('', $TARGET_PORT), Handler) as httpd:
    httpd.serve_forever()
" &
    MOCK_PID=$!
    sleep 1
    
    if ! kill -0 $MOCK_PID 2>/dev/null; then
        log_error "Failed to start mock server"
        exit 1
    fi
    
    log_success "Mock server started (PID: $MOCK_PID)"
}

# Build and start the proxy
start_proxy() {
    log_info "Building Nomos proxy (release mode)..."
    cargo build --release -p nomos-core 2>&1 | tail -5
    
    log_info "Starting Nomos proxy on port $PROXY_PORT..."
    
    TARGET_URL="http://127.0.0.1:$TARGET_PORT" \
    LISTEN_ADDR="127.0.0.1:$PROXY_PORT" \
    RUST_LOG=warn \
        ./target/release/nomos-core &
    PROXY_PID=$!
    sleep 2
    
    if ! kill -0 $PROXY_PID 2>/dev/null; then
        log_error "Failed to start proxy"
        cleanup
        exit 1
    fi
    
    log_success "Proxy started (PID: $PROXY_PID)"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    if [ -n "${PROXY_PID:-}" ]; then
        kill $PROXY_PID 2>/dev/null || true
    fi
    
    if [ -n "${MOCK_PID:-}" ]; then
        kill $MOCK_PID 2>/dev/null || true
    fi
    
    log_success "Cleanup complete"
}

# Run warmup requests
warmup() {
    log_info "Warming up with $WARMUP_REQUESTS requests..."
    
    case $BENCH_TOOL in
        wrk)
            wrk -t2 -c10 -d5s "http://127.0.0.1:$PROXY_PORT/" > /dev/null 2>&1
            ;;
        hey)
            hey -n $WARMUP_REQUESTS -c 10 -q 1000 "http://127.0.0.1:$PROXY_PORT/" > /dev/null 2>&1
            ;;
        ab)
            ab -n $WARMUP_REQUESTS -c 10 "http://127.0.0.1:$PROXY_PORT/" > /dev/null 2>&1
            ;;
    esac
    
    log_success "Warmup complete"
}

# Run direct benchmark (without proxy)
bench_direct() {
    log_info "Benchmarking DIRECT to mock server (baseline)..."
    
    case $BENCH_TOOL in
        wrk)
            DIRECT_RESULT=$(wrk -t4 -c$CONCURRENCY -d10s "http://127.0.0.1:$TARGET_PORT/" 2>&1)
            echo "$DIRECT_RESULT"
            ;;
        hey)
            hey -n $REQUESTS -c $CONCURRENCY "http://127.0.0.1:$TARGET_PORT/"
            ;;
        ab)
            ab -n $REQUESTS -c $CONCURRENCY "http://127.0.0.1:$TARGET_PORT/"
            ;;
    esac
    
    echo
}

# Run proxy benchmark
bench_proxy() {
    log_info "Benchmarking THROUGH Nomos proxy..."
    
    case $BENCH_TOOL in
        wrk)
            PROXY_RESULT=$(wrk -t4 -c$CONCURRENCY -d10s "http://127.0.0.1:$PROXY_PORT/" 2>&1)
            echo "$PROXY_RESULT"
            ;;
        hey)
            hey -n $REQUESTS -c $CONCURRENCY "http://127.0.0.1:$PROXY_PORT/"
            ;;
        ab)
            ab -n $REQUESTS -c $CONCURRENCY "http://127.0.0.1:$PROXY_PORT/"
            ;;
    esac
    
    echo
}

# Verify Nomos headers
verify_headers() {
    log_info "Verifying Nomos response headers..."
    
    RESPONSE=$(curl -s -I "http://127.0.0.1:$PROXY_PORT/" 2>&1)
    
    if echo "$RESPONSE" | grep -q "X-Nomos-Healed"; then
        log_success "X-Nomos-Healed header present"
    else
        log_warn "X-Nomos-Healed header missing"
    fi
    
    if echo "$RESPONSE" | grep -q "X-Nomos-Latency-Us"; then
        LATENCY=$(echo "$RESPONSE" | grep "X-Nomos-Latency-Us" | awk '{print $2}' | tr -d '\r')
        log_success "X-Nomos-Latency-Us: ${LATENCY}µs"
        
        # Check if overhead is under 1ms (1000µs)
        if [ "${LATENCY:-0}" -lt 1000 ]; then
            log_success "Nomos overhead is under 1ms target!"
        else
            log_warn "Nomos overhead exceeds 1ms target"
        fi
    fi
    
    echo
}

# Main execution
main() {
    echo
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║              NOMOS PROXY BASELINE BENCHMARK                   ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  Requests: $REQUESTS  |  Concurrency: $CONCURRENCY                          ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo
    
    trap cleanup EXIT
    
    check_dependencies
    start_mock_server
    start_proxy
    warmup
    
    echo
    echo "═══════════════════════════════════════════════════════════════"
    echo "BASELINE (Direct to mock server)"
    echo "═══════════════════════════════════════════════════════════════"
    bench_direct
    
    echo "═══════════════════════════════════════════════════════════════"
    echo "THROUGH NOMOS PROXY"
    echo "═══════════════════════════════════════════════════════════════"
    bench_proxy
    
    verify_headers
    
    echo "═══════════════════════════════════════════════════════════════"
    echo "Benchmark complete."
    echo "Compare the latency numbers above to verify sub-1ms overhead."
    echo "═══════════════════════════════════════════════════════════════"
}

main "$@"
