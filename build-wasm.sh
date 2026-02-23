#!/bin/bash
# Build script for nomos-healer-guest WASM module
#
# Prerequisites:
#   rustup target add wasm32-wasip1
#
# Output:
#   target/wasm32-wasip1/release/nomos_healer_guest.wasm

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building WASM healer guest..."

# Build the WASM module
cargo build \
    --package nomos-healer-guest \
    --target wasm32-wasip1 \
    --release

# Get the output path
WASM_PATH="target/wasm32-wasip1/release/nomos_healer_guest.wasm"

if [ -f "$WASM_PATH" ]; then
    SIZE=$(stat --printf="%s" "$WASM_PATH" 2>/dev/null || stat -f %z "$WASM_PATH")
    SIZE_KB=$(echo "scale=2; $SIZE / 1024" | bc)
    
    echo ""
    echo "✓ Built successfully: $WASM_PATH"
    echo "  Size: ${SIZE_KB} KB"
    
    # Check against 500KB target
    if [ "$SIZE" -gt 512000 ]; then
        echo "  ⚠  WARNING: Exceeds 500KB target!"
    else
        echo "  ✓ Under 500KB target"
    fi
    
    # Optional: Run wasm-opt for further size reduction
    if command -v wasm-opt &> /dev/null; then
        echo ""
        echo "Running wasm-opt for size optimization..."
        wasm-opt -Oz "$WASM_PATH" -o "${WASM_PATH%.wasm}.opt.wasm"
        OPT_SIZE=$(stat --printf="%s" "${WASM_PATH%.wasm}.opt.wasm" 2>/dev/null || stat -f %z "${WASM_PATH%.wasm}.opt.wasm")
        OPT_SIZE_KB=$(echo "scale=2; $OPT_SIZE / 1024" | bc)
        echo "  Optimized size: ${OPT_SIZE_KB} KB"
    fi
else
    echo "✗ Build failed!"
    exit 1
fi
