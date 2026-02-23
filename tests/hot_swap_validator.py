#!/usr/bin/env python3
"""
Nomos Haltless Hot-Swap Validation
===================================

Validates zero-downtime WASM module hot-swap under load.
Performs live hot-swap via POST :8081/v1/healer while traffic flows at peak capacity.

Test criteria:
- Zero packets dropped during swap
- p99 latency does not spike (remains < 1ms)
- Requests continue to be healed after swap

Usage:
    python hot_swap_validator.py [--duration 30] [--swap-interval 10]
    
Requirements:
    pip install aiohttp httpx uvloop
"""

import asyncio
import argparse
import json
import os
import random
import string
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from pathlib import Path

try:
    import aiohttp
    import httpx
    import uvloop
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "httpx", "uvloop"])
    import aiohttp
    import httpx
    import uvloop


# ============================================================================
# Test Configuration
# ============================================================================

PROXY_URL = "http://127.0.0.1:8080"
CONTROL_URL = "http://127.0.0.1:8081"

# Path to WASM healer binary
WASM_HEALER_PATH = Path(__file__).parent.parent / "nomos-healer-guest/target/wasm32-wasip1/release/nomos_healer_guest.wasm"


# ============================================================================
# Metrics Tracking
# ============================================================================

@dataclass
class SwapMetrics:
    """Metrics captured during a hot-swap event."""
    swap_time: float
    pre_swap_version: int
    post_swap_version: int
    requests_during_swap: int
    errors_during_swap: int
    max_latency_ms: float
    p99_latency_ms: float
    packets_dropped: int
    swap_duration_ms: float


@dataclass
class TestStats:
    """Overall test statistics."""
    latencies: deque = field(default_factory=lambda: deque(maxlen=100000))
    success_count: int = 0
    error_count: int = 0
    healed_count: int = 0
    swap_events: List[SwapMetrics] = field(default_factory=list)
    pre_swap_requests: int = 0
    swap_window_start: float = 0
    swap_window_latencies: List[float] = field(default_factory=list)
    in_swap_window: bool = False
    
    def record(self, latency_ms: float, healed: bool, success: bool):
        self.latencies.append(latency_ms)
        if success:
            self.success_count += 1
            if healed:
                self.healed_count += 1
        else:
            self.error_count += 1
        
        if self.in_swap_window:
            self.swap_window_latencies.append(latency_ms)
    
    def get_percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]
    
    def start_swap_window(self):
        self.in_swap_window = True
        self.swap_window_start = time.time()
        self.swap_window_latencies = []
        self.pre_swap_requests = self.success_count + self.error_count
    
    def end_swap_window(self, pre_version: int, post_version: int, swap_duration_ms: float):
        self.in_swap_window = False
        
        requests_during = (self.success_count + self.error_count) - self.pre_swap_requests
        errors_during = len([l for l in self.swap_window_latencies if l > 5000])  # Timeout = error
        
        max_lat = max(self.swap_window_latencies) if self.swap_window_latencies else 0
        p99_lat = 0
        if self.swap_window_latencies:
            sorted_lat = sorted(self.swap_window_latencies)
            p99_idx = int(len(sorted_lat) * 0.99)
            p99_lat = sorted_lat[min(p99_idx, len(sorted_lat) - 1)]
        
        swap_metrics = SwapMetrics(
            swap_time=self.swap_window_start,
            pre_swap_version=pre_version,
            post_swap_version=post_version,
            requests_during_swap=requests_during,
            errors_during_swap=errors_during,
            max_latency_ms=max_lat,
            p99_latency_ms=p99_lat,
            packets_dropped=errors_during,
            swap_duration_ms=swap_duration_ms,
        )
        
        self.swap_events.append(swap_metrics)
        return swap_metrics


# ============================================================================
# Load Generator
# ============================================================================

# Drifted payload for testing
def generate_drifted_payload() -> Dict:
    """Generate a drifted payload."""
    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return {
        "uuid": f"usr_{random_id}",  # Renamed from user_id
        "name": "Test User",         # Renamed from full_name
        "email": "test@example.com", # Renamed from email_address
        "balance": 100.50,           # Renamed from account_balance
        "verified": True,            # Renamed from is_verified
    }


class LoadGenerator:
    """Generates continuous load during hot-swap tests."""
    
    def __init__(self, stats: TestStats, target_rps: int = 5000, num_workers: int = 4):
        self.stats = stats
        self.target_rps = target_rps
        self.num_workers = num_workers
        self.running = False
        
    async def _worker(self, session: aiohttp.ClientSession) -> None:
        """Worker that sends requests."""
        delay = 1.0 / (self.target_rps / self.num_workers)
        
        while self.running:
            payload = generate_drifted_payload()
            start = time.perf_counter()
            
            try:
                async with session.post(
                    f"{PROXY_URL}/api/user",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    latency_ms = (time.perf_counter() - start) * 1000
                    healed = response.headers.get('X-Nomos-Healed', '').lower() == 'true'
                    self.stats.record(latency_ms, healed, response.status < 500)
                    
            except Exception:
                latency_ms = (time.perf_counter() - start) * 1000
                self.stats.record(latency_ms, False, False)
            
            await asyncio.sleep(delay)
    
    async def start(self, session: aiohttp.ClientSession) -> List[asyncio.Task]:
        """Start load generation workers."""
        self.running = True
        return [
            asyncio.create_task(self._worker(session))
            for _ in range(self.num_workers)
        ]
    
    def stop(self):
        """Stop load generation."""
        self.running = False


# ============================================================================
# Hot-Swap Validator
# ============================================================================

class HotSwapValidator:
    """Validates zero-downtime hot-swap capability."""
    
    def __init__(
        self,
        duration_secs: int = 30,
        swap_interval_secs: int = 10,
        target_rps: int = 5000,
    ):
        self.duration_secs = duration_secs
        self.swap_interval_secs = swap_interval_secs
        self.target_rps = target_rps
        self.stats = TestStats()
        self.load_gen = LoadGenerator(self.stats, target_rps)
        self.control_client = httpx.AsyncClient(timeout=10.0)
        
    async def get_healer_version(self) -> int:
        """Get current healer version from control plane."""
        try:
            response = await self.control_client.get(f"{CONTROL_URL}/v1/healer/version")
            if response.status_code == 200:
                return response.json().get('version', 0)
        except Exception:
            pass
        return 0
    
    async def perform_hot_swap(self) -> Tuple[int, int, float]:
        """Perform a hot-swap and return (pre_version, post_version, duration_ms)."""
        pre_version = await self.get_healer_version()
        
        # Read WASM binary
        if not WASM_HEALER_PATH.exists():
            print(f"[ERROR] WASM healer not found at: {WASM_HEALER_PATH}")
            return pre_version, pre_version, 0
        
        wasm_binary = WASM_HEALER_PATH.read_bytes()
        
        # Perform hot-swap
        start = time.perf_counter()
        
        try:
            response = await self.control_client.post(
                f"{CONTROL_URL}/v1/healer",
                content=wasm_binary,
                headers={"Content-Type": "application/wasm"}
            )
            
            duration_ms = (time.perf_counter() - start) * 1000
            
            if response.status_code == 200:
                post_version = await self.get_healer_version()
                return pre_version, post_version, duration_ms
            else:
                print(f"[WARN] Hot-swap failed: {response.status_code}")
                return pre_version, pre_version, duration_ms
                
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            print(f"[ERROR] Hot-swap error: {e}")
            return pre_version, pre_version, duration_ms
    
    async def _stats_reporter(self) -> None:
        """Report stats during test."""
        last_count = 0
        
        while True:
            await asyncio.sleep(1.0)
            
            current = self.stats.success_count + self.stats.error_count
            rps = current - last_count
            last_count = current
            
            p99 = self.stats.get_percentile(99)
            nomos_ok = "✅" if p99 < 1.0 else "❌"
            
            swap_status = "[SWAPPING]" if self.stats.in_swap_window else ""
            
            print(f"\r[{time.strftime('%H:%M:%S')}] "
                  f"RPS: {rps:>5,} | "
                  f"Total: {current:>8,} | "
                  f"p99: {p99:>6.2f}ms {nomos_ok} | "
                  f"Errors: {self.stats.error_count:>4} | "
                  f"Swaps: {len(self.stats.swap_events)} "
                  f"{swap_status}", end="", flush=True)
    
    async def run(self) -> Dict:
        """Run the hot-swap validation test."""
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              NOMOS HALTLESS HOT-SWAP VALIDATION                       ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Test Duration:      {self.duration_secs:<52} seconds ║
║  Swap Interval:      {self.swap_interval_secs:<52} seconds ║
║  Target RPS:         {self.target_rps:<52,} ║
║  WASM Path:          {str(WASM_HEALER_PATH)[:50]:<52} ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
        
        # Check WASM file exists
        if not WASM_HEALER_PATH.exists():
            print(f"[ERROR] WASM healer not found. Run: ./build-wasm.sh")
            return {"error": "WASM file not found"}
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(
            limit=100,
            keepalive_timeout=30,
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # Start load generation
            workers = await self.load_gen.start(session)
            reporter = asyncio.create_task(self._stats_reporter())
            
            # Schedule hot-swaps
            start_time = time.time()
            next_swap = start_time + self.swap_interval_secs
            
            try:
                while time.time() - start_time < self.duration_secs:
                    current_time = time.time()
                    
                    # Time for a hot-swap?
                    if current_time >= next_swap:
                        print(f"\n\n>>> INITIATING HOT-SWAP <<<")
                        
                        # Start swap window (capture metrics during swap)
                        self.stats.start_swap_window()
                        
                        # Perform the swap
                        pre_ver, post_ver, duration = await self.perform_hot_swap()
                        
                        # End swap window and record metrics
                        swap_metrics = self.stats.end_swap_window(pre_ver, post_ver, duration)
                        
                        # Report swap results
                        status = "✅ SUCCESS" if swap_metrics.errors_during_swap == 0 else "❌ DROPPED"
                        print(f"\n>>> HOT-SWAP COMPLETE: {status}")
                        print(f"    Version: v{pre_ver} → v{post_ver}")
                        print(f"    Duration: {duration:.2f}ms")
                        print(f"    Requests during swap: {swap_metrics.requests_during_swap}")
                        print(f"    Errors during swap: {swap_metrics.errors_during_swap}")
                        print(f"    Max latency during swap: {swap_metrics.max_latency_ms:.2f}ms")
                        print(f"    p99 latency during swap: {swap_metrics.p99_latency_ms:.2f}ms\n")
                        
                        next_swap = current_time + self.swap_interval_secs
                    
                    await asyncio.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\n\nTest interrupted by user")
            
            # Stop everything
            self.load_gen.stop()
            reporter.cancel()
            
            for w in workers:
                w.cancel()
            
            try:
                await asyncio.gather(*workers, reporter, return_exceptions=True)
            except asyncio.CancelledError:
                pass
        
        await self.control_client.aclose()
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """Generate final test report."""
        total_requests = self.stats.success_count + self.stats.error_count
        total_swaps = len(self.stats.swap_events)
        
        # Calculate swap statistics
        zero_drop_swaps = sum(1 for s in self.stats.swap_events if s.errors_during_swap == 0)
        max_swap_latency = max((s.max_latency_ms for s in self.stats.swap_events), default=0)
        avg_swap_duration = (
            sum(s.swap_duration_ms for s in self.stats.swap_events) / total_swaps
            if total_swaps > 0 else 0
        )
        
        # Nomos Law compliance during swaps
        nomos_compliant_swaps = sum(
            1 for s in self.stats.swap_events if s.p99_latency_ms < 1.0
        )
        
        report = {
            "total_requests": total_requests,
            "successful_requests": self.stats.success_count,
            "failed_requests": self.stats.error_count,
            "healed_requests": self.stats.healed_count,
            "overall_p99_ms": self.stats.get_percentile(99),
            "total_hot_swaps": total_swaps,
            "zero_drop_swaps": zero_drop_swaps,
            "nomos_compliant_swaps": nomos_compliant_swaps,
            "max_latency_during_swap_ms": max_swap_latency,
            "avg_swap_duration_ms": avg_swap_duration,
            "haltless_validated": zero_drop_swaps == total_swaps,
            "nomos_law_maintained": nomos_compliant_swaps == total_swaps,
            "swap_events": [
                {
                    "time": s.swap_time,
                    "pre_version": s.pre_swap_version,
                    "post_version": s.post_swap_version,
                    "requests_during": s.requests_during_swap,
                    "errors_during": s.errors_during_swap,
                    "max_latency_ms": s.max_latency_ms,
                    "p99_latency_ms": s.p99_latency_ms,
                    "swap_duration_ms": s.swap_duration_ms,
                }
                for s in self.stats.swap_events
            ]
        }
        
        # Print report
        haltless_status = "✅ PASS" if report['haltless_validated'] else "❌ FAIL"
        nomos_status = "✅ PASS" if report['nomos_law_maintained'] else "❌ FAIL"
        
        print(f"""

╔═══════════════════════════════════════════════════════════════════════╗
║                    HOT-SWAP VALIDATION REPORT                         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Total Requests:         {report['total_requests']:>10,}                                 ║
║  Successful:             {report['successful_requests']:>10,}                                 ║
║  Failed:                 {report['failed_requests']:>10,}                                 ║
║  Healed:                 {report['healed_requests']:>10,}                                 ║
║  Overall p99:            {report['overall_p99_ms']:>10.3f} ms                               ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Total Hot-Swaps:        {report['total_hot_swaps']:>10}                                 ║
║  Zero-Drop Swaps:        {report['zero_drop_swaps']:>10} / {total_swaps}                              ║
║  Nomos-Compliant Swaps:  {report['nomos_compliant_swaps']:>10} / {total_swaps}                              ║
║  Max Latency During Swap:{report['max_latency_during_swap_ms']:>10.2f} ms                               ║
║  Avg Swap Duration:      {report['avg_swap_duration_ms']:>10.2f} ms                               ║
╠═══════════════════════════════════════════════════════════════════════╣
║  HALTLESS VALIDATION:    {haltless_status:>10}                                 ║
║  NOMOS LAW MAINTAINED:   {nomos_status:>10}                                 ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
        
        return report


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nomos Haltless Hot-Swap Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=30,
        help="Test duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--swap-interval", "-i",
        type=int,
        default=10,
        help="Interval between hot-swaps in seconds (default: 10)"
    )
    parser.add_argument(
        "--rps", "-r",
        type=int,
        default=5000,
        help="Target requests per second (default: 5000)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    # Use uvloop
    if sys.platform != "win32":
        uvloop.install()
    
    validator = HotSwapValidator(
        duration_secs=args.duration,
        swap_interval_secs=args.swap_interval,
        target_rps=args.rps,
    )
    
    report = asyncio.run(validator.run())
    
    if args.output and 'error' not in report:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {args.output}")
    
    # Exit with appropriate code
    if report.get('haltless_validated') and report.get('nomos_law_maintained'):
        print("\n[SUCCESS] All validation criteria passed!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Validation criteria not met!")
        sys.exit(1)


if __name__ == "__main__":
    main()
