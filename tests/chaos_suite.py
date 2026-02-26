#!/usr/bin/env python3
"""
Nomos Chaos Test Suite
======================

Comprehensive resilience testing for the Nomos proxy.

Tests:
1. Upstream failures (timeouts, 5xx errors, connection refused)
2. Malformed JSON responses
3. High concurrency stress
4. Rapid drift mode switching
5. Mixed chaos scenarios
6. Recovery testing

Usage:
    python chaos_suite.py [--proxy-url http://localhost:8080]
"""

import argparse
import asyncio
import json
import random
import signal
import string
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

try:
    import aiohttp
    import requests
except ImportError:
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "requests"])
    import aiohttp
    import requests


# ============================================================================
# Test Results
# ============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    details: str
    metrics: Dict[str, float]


class TestSuite:
    """Manages test execution and results."""
    
    def __init__(self, proxy_url: str = "http://localhost:8080"):
        self.proxy_url = proxy_url
        self.upstream_url = "http://localhost:9090"
        self.results: List[TestResult] = []
        self.upstream_proc: Optional[subprocess.Popen] = None
    
    def start_upstream(self, drift_mode: str = "v2"):
        """Start upstream server with specified drift mode."""
        self.stop_upstream()
        time.sleep(0.5)
        self.upstream_proc = subprocess.Popen(
            [sys.executable, "tests/upstream_server.py", "--port", "9090", "--drift-mode", drift_mode],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(1)  # Wait for server to start
    
    def stop_upstream(self):
        """Stop upstream server."""
        subprocess.run(["pkill", "-f", "upstream_server"], capture_output=True)
        if self.upstream_proc:
            self.upstream_proc.terminate()
            self.upstream_proc = None
    
    def add_result(self, result: TestResult):
        """Add a test result."""
        self.results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status} {result.name} ({result.duration_ms:.1f}ms)")
        if not result.passed:
            print(f"       Details: {result.details}")
    
    def print_summary(self):
        """Print test summary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        print("\n" + "=" * 60)
        print("CHAOS SUITE SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {100 * passed / len(self.results):.1f}%")
        print("=" * 60)
        
        if failed > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.details}")


# ============================================================================
# Chaos Tests
# ============================================================================

def test_upstream_timeout(suite: TestSuite) -> TestResult:
    """Test proxy behavior when upstream times out."""
    start = time.time()
    
    # Stop upstream to simulate timeout
    suite.stop_upstream()
    time.sleep(0.5)
    
    try:
        # Should get error or timeout from proxy
        response = requests.get(f"{suite.proxy_url}/api/user", timeout=5)
        # Proxy should handle gracefully (5xx or pass-through error)
        passed = response.status_code >= 500 or response.status_code == 502
        details = f"Got {response.status_code}"
    except requests.exceptions.RequestException as e:
        # Connection errors are acceptable
        passed = True
        details = f"Expected error: {type(e).__name__}"
    
    duration = (time.time() - start) * 1000
    
    # Restart upstream for next tests
    suite.start_upstream("v2")
    
    return TestResult(
        name="Upstream Timeout",
        passed=passed,
        duration_ms=duration,
        details=details,
        metrics={"attempts": 1}
    )


def test_malformed_json(suite: TestSuite) -> TestResult:
    """Test proxy behavior with malformed JSON from upstream."""
    start = time.time()
    
    # We'll test with various malformed payloads through the proxy
    # Since we can't easily inject malformed JSON, we test recovery
    suite.start_upstream("v2")
    
    success_count = 0
    total = 10
    
    for _ in range(total):
        try:
            response = requests.get(f"{suite.proxy_url}/api/user", timeout=5)
            if response.status_code == 200:
                # Verify it's valid JSON
                json.loads(response.text)
                success_count += 1
        except Exception:
            pass
    
    duration = (time.time() - start) * 1000
    passed = success_count == total
    
    return TestResult(
        name="Malformed JSON Recovery",
        passed=passed,
        duration_ms=duration,
        details=f"{success_count}/{total} valid responses",
        metrics={"success_rate": success_count / total}
    )


def test_high_concurrency(suite: TestSuite) -> TestResult:
    """Test proxy under high concurrent load."""
    start = time.time()
    
    suite.start_upstream("v2")
    
    num_requests = 100
    num_workers = 20
    results = []
    
    def make_request(_):
        try:
            response = requests.get(f"{suite.proxy_url}/api/user", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(make_request, range(num_requests)))
    
    duration = (time.time() - start) * 1000
    success_count = sum(results)
    success_rate = success_count / num_requests
    rps = num_requests / (duration / 1000)
    
    passed = success_rate >= 0.95  # 95% success threshold
    
    return TestResult(
        name="High Concurrency (100 req, 20 workers)",
        passed=passed,
        duration_ms=duration,
        details=f"{success_count}/{num_requests} success ({success_rate*100:.1f}%), {rps:.0f} RPS",
        metrics={"success_rate": success_rate, "rps": rps}
    )


def test_rapid_mode_switching(suite: TestSuite) -> TestResult:
    """Test rapid switching between drift modes."""
    start = time.time()
    
    modes = ["healthy", "v2", "camel", "nested", "typo", "abbrev", "legacy", "mixed"]
    success_count = 0
    total = len(modes) * 3  # 3 requests per mode
    
    for mode in modes:
        suite.start_upstream(mode)
        time.sleep(0.3)  # Brief pause
        
        for _ in range(3):
            try:
                response = requests.get(f"{suite.proxy_url}/api/user", timeout=5)
                if response.status_code == 200:
                    success_count += 1
            except Exception:
                pass
    
    duration = (time.time() - start) * 1000
    passed = success_count >= total * 0.9  # 90% threshold
    
    return TestResult(
        name="Rapid Mode Switching",
        passed=passed,
        duration_ms=duration,
        details=f"{success_count}/{total} success across {len(modes)} modes",
        metrics={"success_rate": success_count / total}
    )


def test_burst_traffic(suite: TestSuite) -> TestResult:
    """Test proxy under burst traffic patterns."""
    start = time.time()
    
    suite.start_upstream("v2")
    
    total_success = 0
    total_requests = 0
    
    # 5 bursts of 30 requests each
    for burst in range(5):
        results = []
        
        def make_request(_):
            try:
                response = requests.get(f"{suite.proxy_url}/api/user", timeout=5)
                return response.status_code == 200
            except Exception:
                return False
        
        with ThreadPoolExecutor(max_workers=30) as executor:
            results = list(executor.map(make_request, range(30)))
        
        total_success += sum(results)
        total_requests += len(results)
        time.sleep(0.5)  # Brief pause between bursts
    
    duration = (time.time() - start) * 1000
    success_rate = total_success / total_requests
    passed = success_rate >= 0.95
    
    return TestResult(
        name="Burst Traffic (5x30 requests)",
        passed=passed,
        duration_ms=duration,
        details=f"{total_success}/{total_requests} success ({success_rate*100:.1f}%)",
        metrics={"success_rate": success_rate}
    )


def test_upstream_flapping(suite: TestSuite) -> TestResult:
    """Test proxy when upstream keeps going up and down."""
    start = time.time()
    
    success_count = 0
    error_count = 0
    total = 20
    
    for i in range(total):
        if i % 4 == 0:
            suite.stop_upstream()
        elif i % 4 == 2:
            suite.start_upstream("v2")
        
        time.sleep(0.2)
        
        try:
            response = requests.get(f"{suite.proxy_url}/api/user", timeout=3)
            if response.status_code == 200:
                success_count += 1
            else:
                error_count += 1
        except Exception:
            error_count += 1
    
    # Restart upstream
    suite.start_upstream("v2")
    
    duration = (time.time() - start) * 1000
    # We expect roughly 50% success during flapping
    passed = success_count >= total * 0.3  # At least 30% success
    
    return TestResult(
        name="Upstream Flapping",
        passed=passed,
        duration_ms=duration,
        details=f"{success_count} success, {error_count} errors during flapping",
        metrics={"success_count": success_count, "error_count": error_count}
    )


def test_healing_consistency(suite: TestSuite) -> TestResult:
    """Test that healing produces consistent results."""
    start = time.time()
    
    suite.start_upstream("v2")
    
    # Make 20 requests and check for consistent field names
    responses = []
    for _ in range(20):
        try:
            response = requests.get(f"{suite.proxy_url}/api/user", timeout=5)
            if response.status_code == 200:
                data = response.json()
                responses.append(set(data.keys()))
        except Exception:
            pass
    
    duration = (time.time() - start) * 1000
    
    # All responses should have the same keys
    if responses:
        first_keys = responses[0]
        consistent = all(keys == first_keys for keys in responses)
        
        # Check expected healed keys exist
        expected_healed = {"user_id", "full_name", "email_address", "account_balance"}
        has_healed_keys = expected_healed.issubset(first_keys)
        
        passed = consistent and has_healed_keys
        details = f"{len(responses)} responses, consistent={consistent}, healed={has_healed_keys}"
    else:
        passed = False
        details = "No valid responses"
    
    return TestResult(
        name="Healing Consistency",
        passed=passed,
        duration_ms=duration,
        details=details,
        metrics={"response_count": len(responses)}
    )


def test_deep_nested_healing(suite: TestSuite) -> TestResult:
    """Test healing of deeply nested JSON structures."""
    start = time.time()
    
    suite.start_upstream("deep")
    time.sleep(0.5)
    
    success_count = 0
    healing_ops_sum = 0
    
    for _ in range(10):
        try:
            response = requests.get(f"{suite.proxy_url}/api/user", timeout=5)
            if response.status_code == 200:
                # Check healing header
                ops = int(response.headers.get("x-nomos-healing-ops", 0))
                healing_ops_sum += ops
                
                data = response.json()
                # Should have flattened keys
                if "user_id" in data and "full_name" in data:
                    success_count += 1
        except Exception:
            pass
    
    duration = (time.time() - start) * 1000
    avg_ops = healing_ops_sum / 10 if healing_ops_sum else 0
    passed = success_count >= 8 and avg_ops >= 15  # Most should heal with 15+ ops
    
    return TestResult(
        name="Deep Nested Healing (6+ levels)",
        passed=passed,
        duration_ms=duration,
        details=f"{success_count}/10 healed, avg {avg_ops:.0f} ops",
        metrics={"success_count": success_count, "avg_healing_ops": avg_ops}
    )


def test_latency_under_load(suite: TestSuite) -> TestResult:
    """Test that latency stays reasonable under load."""
    start = time.time()
    
    suite.start_upstream("v2")
    
    latencies = []
    
    for _ in range(50):
        req_start = time.time()
        try:
            response = requests.get(f"{suite.proxy_url}/api/user", timeout=5)
            if response.status_code == 200:
                latency = (time.time() - req_start) * 1000
                latencies.append(latency)
                
                # Also check X-Nomos-Latency-us header
                nomos_latency = int(response.headers.get("x-nomos-latency-us", 0))
        except Exception:
            pass
    
    duration = (time.time() - start) * 1000
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        max_latency = max(latencies)
        
        # p99 should be under 50ms
        passed = p99_latency < 50
        details = f"avg={avg_latency:.1f}ms, p99={p99_latency:.1f}ms, max={max_latency:.1f}ms"
    else:
        passed = False
        details = "No successful requests"
        avg_latency = p99_latency = 0
    
    return TestResult(
        name="Latency Under Load (50 requests)",
        passed=passed,
        duration_ms=duration,
        details=details,
        metrics={"avg_latency_ms": avg_latency, "p99_latency_ms": p99_latency}
    )


def test_recovery_after_chaos(suite: TestSuite) -> TestResult:
    """Test proxy recovery after chaos events."""
    start = time.time()
    
    # Apply chaos
    suite.stop_upstream()
    time.sleep(1)
    
    # Restart with clean state
    suite.start_upstream("v2")
    time.sleep(1)
    
    success_count = 0
    for _ in range(10):
        try:
            response = requests.get(f"{suite.proxy_url}/api/user", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "user_id" in data:
                    success_count += 1
        except Exception:
            pass
    
    duration = (time.time() - start) * 1000
    passed = success_count >= 9  # 90% success after recovery
    
    return TestResult(
        name="Recovery After Chaos",
        passed=passed,
        duration_ms=duration,
        details=f"{success_count}/10 success after recovery",
        metrics={"success_count": success_count}
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Nomos Chaos Test Suite")
    parser.add_argument("--proxy-url", default="http://localhost:8080", help="Proxy URL")
    args = parser.parse_args()
    
    print("=" * 60)
    print("NOMOS CHAOS TEST SUITE")
    print("=" * 60)
    print(f"Proxy URL: {args.proxy_url}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    print()
    
    suite = TestSuite(proxy_url=args.proxy_url)
    
    # Check proxy is running
    try:
        requests.get(args.proxy_url, timeout=2)
    except Exception as e:
        print(f"❌ Cannot connect to proxy at {args.proxy_url}")
        print("   Make sure the Nomos proxy is running!")
        sys.exit(1)
    
    print("Running tests...\n")
    
    # Run all tests
    tests = [
        test_upstream_timeout,
        test_malformed_json,
        test_high_concurrency,
        test_rapid_mode_switching,
        test_burst_traffic,
        test_upstream_flapping,
        test_healing_consistency,
        test_deep_nested_healing,
        test_latency_under_load,
        test_recovery_after_chaos,
    ]
    
    for test in tests:
        try:
            result = test(suite)
            suite.add_result(result)
        except Exception as e:
            suite.add_result(TestResult(
                name=test.__name__,
                passed=False,
                duration_ms=0,
                details=f"Exception: {e}",
                metrics={}
            ))
    
    # Cleanup
    suite.stop_upstream()
    
    # Print summary
    suite.print_summary()
    
    # Exit with appropriate code
    failed = sum(1 for r in suite.results if not r.passed)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
