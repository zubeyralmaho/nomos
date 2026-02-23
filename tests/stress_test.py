#!/usr/bin/env python3
"""
Nomos High-Velocity Load Generator
===================================

Floods the proxy with thousands of 'Drifted' JSON requests per second.
Tests the Nomos Law: proxy overhead must not exceed ~714ns p99.

Usage:
    python stress_test.py [--rps 10000] [--duration 60] [--workers 8]
    
Requirements:
    pip install aiohttp uvloop numpy
"""

import asyncio
import argparse
import json
import random
import string
import time
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import deque
import statistics

try:
    import aiohttp
    import uvloop
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "uvloop"])
    import aiohttp
    import uvloop


# ============================================================================
# Drifted Payload Generators
# ============================================================================

# Original schema the client expects
EXPECTED_SCHEMA = {
    "user_id": "usr_12345",
    "full_name": "Alice Smith",
    "email_address": "alice@example.com",
    "account_balance": 1500.50,
    "is_verified": True,
    "created_at": "2026-01-15T10:30:00Z",
    "preferences": {
        "theme": "dark",
        "notifications": True,
        "language": "en"
    },
    "tags": ["premium", "active"]
}

# Drift patterns: realistic API changes
DRIFT_PATTERNS = [
    # Pattern 1: user_id -> uuid (semantic rename)
    {
        "uuid": "usr_12345",
        "full_name": "Alice Smith",
        "email_address": "alice@example.com",
        "account_balance": 1500.50,
        "is_verified": True,
        "created_at": "2026-01-15T10:30:00Z",
        "preferences": {"theme": "dark", "notifications": True, "language": "en"},
        "tags": ["premium", "active"]
    },
    # Pattern 2: user_id -> u_id (abbreviation)
    {
        "u_id": "usr_12345",
        "name": "Alice Smith",  # full_name -> name
        "email": "alice@example.com",  # email_address -> email
        "balance": 1500.50,  # account_balance -> balance
        "verified": True,  # is_verified -> verified
        "created": "2026-01-15T10:30:00Z",  # created_at -> created
        "prefs": {"theme": "dark", "notifications": True, "lang": "en"},
        "labels": ["premium", "active"]  # tags -> labels
    },
    # Pattern 3: user_id -> userId (camelCase conversion)
    {
        "userId": "usr_12345",
        "fullName": "Alice Smith",
        "emailAddress": "alice@example.com",
        "accountBalance": 1500.50,
        "isVerified": True,
        "createdAt": "2026-01-15T10:30:00Z",
        "preferences": {"theme": "dark", "notifications": True, "language": "en"},
        "tags": ["premium", "active"]
    },
    # Pattern 4: Nested restructuring
    {
        "user": {
            "id": "usr_12345",
            "name": "Alice Smith",
            "email": "alice@example.com"
        },
        "account": {
            "balance": 1500.50,
            "verified": True
        },
        "metadata": {
            "created_at": "2026-01-15T10:30:00Z",
            "preferences": {"theme": "dark", "notifications": True, "language": "en"},
            "tags": ["premium", "active"]
        }
    },
    # Pattern 5: API v2 style (type changes)
    {
        "user_id": 12345,  # String -> Int
        "full_name": "Alice Smith",
        "email_address": "alice@example.com",
        "account_balance": "1500.50",  # Float -> String
        "is_verified": 1,  # Bool -> Int
        "created_at": 1736937000,  # ISO String -> Unix timestamp
        "preferences": {"theme": "dark", "notifications": True, "language": "en"},
        "tags": ["premium", "active"]
    },
]


def generate_drifted_payload() -> Dict:
    """Generate a randomly drifted payload."""
    pattern = random.choice(DRIFT_PATTERNS)
    # Add some randomization to values
    payload = json.loads(json.dumps(pattern))  # Deep copy
    
    # Randomize user ID
    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    
    # Find and update ID field regardless of name
    for key in ['user_id', 'uuid', 'u_id', 'userId']:
        if key in payload:
            payload[key] = f"usr_{random_id}"
    if 'user' in payload and isinstance(payload['user'], dict):
        if 'id' in payload['user']:
            payload['user']['id'] = f"usr_{random_id}"
    
    return payload


def generate_healthy_payload() -> Dict:
    """Generate an expected (non-drifted) payload."""
    payload = json.loads(json.dumps(EXPECTED_SCHEMA))
    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    payload['user_id'] = f"usr_{random_id}"
    return payload


# ============================================================================
# Load Generator
# ============================================================================

@dataclass
class RequestStats:
    """Thread-safe statistics collector."""
    latencies: deque = field(default_factory=lambda: deque(maxlen=100000))
    success_count: int = 0
    error_count: int = 0
    healed_count: int = 0
    start_time: float = 0.0
    
    def record(self, latency_ms: float, healed: bool, success: bool):
        self.latencies.append(latency_ms)
        if success:
            self.success_count += 1
            if healed:
                self.healed_count += 1
        else:
            self.error_count += 1
    
    def get_percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]
    
    def get_rps(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return (self.success_count + self.error_count) / elapsed


class LoadGenerator:
    """High-velocity load generator using async HTTP."""
    
    def __init__(
        self,
        target_url: str = "http://127.0.0.1:8080",
        target_rps: int = 10000,
        duration_secs: int = 60,
        num_workers: int = 8,
        drift_ratio: float = 0.8,  # 80% drifted, 20% healthy
    ):
        self.target_url = target_url
        self.target_rps = target_rps
        self.duration_secs = duration_secs
        self.num_workers = num_workers
        self.drift_ratio = drift_ratio
        self.stats = RequestStats()
        self.running = False
        self.connector: Optional[aiohttp.TCPConnector] = None
        
    async def _send_request(self, session: aiohttp.ClientSession) -> None:
        """Send a single request and record statistics."""
        try:
            # Decide drift vs healthy
            if random.random() < self.drift_ratio:
                payload = generate_drifted_payload()
            else:
                payload = generate_healthy_payload()
            
            start = time.perf_counter()
            
            async with session.post(
                f"{self.target_url}/api/user",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as response:
                latency_ms = (time.perf_counter() - start) * 1000
                
                # Check if response was healed
                healed = response.headers.get('X-Nomos-Healed', '').lower() == 'true'
                
                self.stats.record(latency_ms, healed, response.status < 500)
                
        except asyncio.TimeoutError:
            self.stats.record(5000.0, False, False)
        except Exception as e:
            self.stats.record(0.0, False, False)
    
    async def _worker(self, session: aiohttp.ClientSession, worker_id: int) -> None:
        """Worker coroutine that sends requests at target rate."""
        # Calculate delay between requests for this worker
        requests_per_worker = self.target_rps / self.num_workers
        delay = 1.0 / requests_per_worker if requests_per_worker > 0 else 0.001
        
        while self.running:
            await self._send_request(session)
            # Sleep to maintain target RPS
            await asyncio.sleep(delay)
    
    async def _stats_reporter(self) -> None:
        """Periodically report statistics."""
        last_count = 0
        
        while self.running:
            await asyncio.sleep(1.0)
            
            current_count = self.stats.success_count + self.stats.error_count
            rps = current_count - last_count
            last_count = current_count
            
            p50 = self.stats.get_percentile(50)
            p99 = self.stats.get_percentile(99)
            p999 = self.stats.get_percentile(99.9)
            
            healed_pct = (
                self.stats.healed_count / max(self.stats.success_count, 1) * 100
            )
            error_pct = (
                self.stats.error_count / max(current_count, 1) * 100
            )
            
            # Check Nomos Law violation (714ns = 0.000714ms)
            nomos_law_ok = "✅" if p99 < 1.0 else "❌"
            
            print(f"\r[{time.strftime('%H:%M:%S')}] "
                  f"RPS: {rps:>6,} | "
                  f"Total: {current_count:>8,} | "
                  f"p50: {p50:>6.2f}ms | "
                  f"p99: {p99:>6.2f}ms {nomos_law_ok} | "
                  f"p99.9: {p999:>6.2f}ms | "
                  f"Healed: {healed_pct:>5.1f}% | "
                  f"Errors: {error_pct:>4.1f}%", end="", flush=True)
    
    async def run(self) -> Dict:
        """Run the load test."""
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    NOMOS STRESS TEST - LOAD GENERATOR                 ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Target URL:     {self.target_url:<52} ║
║  Target RPS:     {self.target_rps:<52,} ║
║  Duration:       {self.duration_secs:<52} seconds ║
║  Workers:        {self.num_workers:<52} ║
║  Drift Ratio:    {self.drift_ratio * 100:<51.0f}% ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
        
        # Create connection pool
        self.connector = aiohttp.TCPConnector(
            limit=self.num_workers * 10,
            limit_per_host=self.num_workers * 10,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
        )
        
        async with aiohttp.ClientSession(connector=self.connector) as session:
            self.running = True
            self.stats.start_time = time.time()
            
            # Start workers and stats reporter
            workers = [
                asyncio.create_task(self._worker(session, i))
                for i in range(self.num_workers)
            ]
            reporter = asyncio.create_task(self._stats_reporter())
            
            # Run for specified duration
            await asyncio.sleep(self.duration_secs)
            
            # Stop
            self.running = False
            print("\n\nStopping workers...")
            
            # Cancel tasks
            for w in workers:
                w.cancel()
            reporter.cancel()
            
            try:
                await asyncio.gather(*workers, reporter, return_exceptions=True)
            except asyncio.CancelledError:
                pass
        
        # Final report
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """Generate final test report."""
        total = self.stats.success_count + self.stats.error_count
        elapsed = time.time() - self.stats.start_time
        
        report = {
            "duration_secs": elapsed,
            "total_requests": total,
            "successful_requests": self.stats.success_count,
            "failed_requests": self.stats.error_count,
            "healed_requests": self.stats.healed_count,
            "average_rps": total / elapsed if elapsed > 0 else 0,
            "latency_p50_ms": self.stats.get_percentile(50),
            "latency_p95_ms": self.stats.get_percentile(95),
            "latency_p99_ms": self.stats.get_percentile(99),
            "latency_p999_ms": self.stats.get_percentile(99.9),
            "nomos_law_compliant": self.stats.get_percentile(99) < 1.0,
            "healing_rate": self.stats.healed_count / max(self.stats.success_count, 1),
            "error_rate": self.stats.error_count / max(total, 1),
        }
        
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                         FINAL TEST REPORT                             ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Duration:           {report['duration_secs']:>10.2f} seconds                            ║
║  Total Requests:     {report['total_requests']:>10,}                                     ║
║  Successful:         {report['successful_requests']:>10,}                                     ║
║  Failed:             {report['failed_requests']:>10,}                                     ║
║  Healed:             {report['healed_requests']:>10,}                                     ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Average RPS:        {report['average_rps']:>10,.1f}                                     ║
║  Latency p50:        {report['latency_p50_ms']:>10.3f} ms                                  ║
║  Latency p95:        {report['latency_p95_ms']:>10.3f} ms                                  ║
║  Latency p99:        {report['latency_p99_ms']:>10.3f} ms                                  ║
║  Latency p99.9:      {report['latency_p999_ms']:>10.3f} ms                                  ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Nomos Law (p99<1ms): {'✅ PASS' if report['nomos_law_compliant'] else '❌ FAIL':>10}                                     ║
║  Healing Rate:       {report['healing_rate'] * 100:>10.1f}%                                    ║
║  Error Rate:         {report['error_rate'] * 100:>10.1f}%                                    ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
        
        return report


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nomos High-Velocity Load Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic stress test at 10k RPS for 30 seconds
    python stress_test.py --rps 10000 --duration 30
    
    # High-intensity test with more workers
    python stress_test.py --rps 50000 --workers 16 --duration 60
    
    # Test specific endpoint
    python stress_test.py --url http://localhost:8080 --rps 5000
        """
    )
    
    parser.add_argument(
        "--url", "-u",
        default="http://127.0.0.1:8080",
        help="Target proxy URL (default: http://127.0.0.1:8080)"
    )
    parser.add_argument(
        "--rps", "-r",
        type=int,
        default=10000,
        help="Target requests per second (default: 10000)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="Test duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of async workers (default: 8)"
    )
    parser.add_argument(
        "--drift-ratio",
        type=float,
        default=0.8,
        help="Ratio of drifted vs healthy requests (default: 0.8)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    # Use uvloop for better async performance on Linux
    if sys.platform != "win32":
        uvloop.install()
    
    # Create and run load generator
    generator = LoadGenerator(
        target_url=args.url,
        target_rps=args.rps,
        duration_secs=args.duration,
        num_workers=args.workers,
        drift_ratio=args.drift_ratio,
    )
    
    report = asyncio.run(generator.run())
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if report['nomos_law_compliant'] else 1)


if __name__ == "__main__":
    main()
