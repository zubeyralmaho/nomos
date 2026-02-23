#!/bin/bash
# Nomos eBPF Build Script
# Builds the XDP program for the BPF target
#
# Usage:
#   ./build-ebpf.sh          # Build in release mode
#   ./build-ebpf.sh debug    # Build in debug mode
#
# Prerequisites:
#   - Rust nightly toolchain with rust-src component
#   - bpf-linker (cargo install bpf-linker)
#   - Linux kernel headers (for BTF info)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-release}"
EBPF_DIR="nomos-ebpf"
TARGET="bpfel-unknown-none"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Nomos eBPF Build ===${NC}"
echo "Mode: $MODE"
echo "Target: $TARGET"
echo ""

# Check prerequisites
check_prereqs() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check for nightly Rust (installed, not necessarily active)
    if ! rustup toolchain list | grep -q "nightly"; then
        echo -e "${RED}Error: Nightly Rust toolchain required${NC}"
        echo "Install with: rustup install nightly"
        exit 1
    fi
    
    # Check for rust-src component
    if ! rustup component list --toolchain nightly | grep -q "rust-src (installed)"; then
        echo -e "${YELLOW}Installing rust-src component...${NC}"
        rustup component add rust-src --toolchain nightly
    fi
    
    # Check for bpf-linker
    if ! command -v bpf-linker &> /dev/null; then
        echo -e "${YELLOW}Installing bpf-linker...${NC}"
        cargo install bpf-linker
    fi
    
    echo -e "${GREEN}Prerequisites satisfied${NC}"
    echo ""
}

# Build the eBPF program
build_ebpf() {
    echo -e "${YELLOW}Building eBPF program...${NC}"
    
    cd "$EBPF_DIR"
    
    if [ "$MODE" = "release" ]; then
        cargo +nightly build \
            --target "$TARGET" \
            --release \
            -Z build-std=core \
            --verbose
        OUTPUT_PATH="target/$TARGET/release/nomos-ebpf"
    else
        cargo +nightly build \
            --target "$TARGET" \
            -Z build-std=core \
            --verbose
        OUTPUT_PATH="target/$TARGET/debug/nomos-ebpf"
    fi
    
    cd "$SCRIPT_DIR"
    
    # Copy to a standard location
    mkdir -p target/ebpf
    cp "$EBPF_DIR/$OUTPUT_PATH" target/ebpf/nomos-xdp.o 2>/dev/null || \
        cp "$EBPF_DIR/$OUTPUT_PATH" target/ebpf/nomos-xdp

    echo ""
    echo -e "${GREEN}Build successful!${NC}"
    echo "Output: target/ebpf/nomos-xdp.o"
    echo ""
    
    # Show BPF program info if llvm-objdump is available
    if command -v llvm-objdump &> /dev/null; then
        echo -e "${YELLOW}BPF Program Sections:${NC}"
        llvm-objdump --section-headers target/ebpf/nomos-xdp* 2>/dev/null || true
    fi
}

# Build userspace components
build_userspace() {
    echo -e "${YELLOW}Building userspace components...${NC}"
    
    if [ "$MODE" = "release" ]; then
        cargo build --release -p nomos-core -p nomos-ebpf-common
    else
        cargo build -p nomos-core -p nomos-ebpf-common
    fi
    
    echo -e "${GREEN}Userspace build successful!${NC}"
    echo ""
}

# Main
main() {
    check_prereqs
    build_ebpf
    build_userspace
    
    echo -e "${GREEN}=== Build Complete ===${NC}"
    echo ""
    echo "To load the XDP program:"
    echo "  sudo ./target/release/nomos-core"
    echo ""
    echo "Or manually with bpftool:"
    echo "  sudo bpftool prog load target/ebpf/nomos-xdp.o /sys/fs/bpf/nomos_xdp"
    echo "  sudo bpftool net attach xdp pinned /sys/fs/bpf/nomos_xdp dev eth0"
}

main "$@"
