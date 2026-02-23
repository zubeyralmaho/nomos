#!/usr/bin/env python3
"""
Nomos Chaos Test Suite Runner
==============================

Master script to orchestrate all chaos and stress tests.

Usage:
    python run_chaos_suite.py [--full | --quick]
    
Components:
    1. stress_test.py       - High-velocity load generator
    2. hot_swap_validator.py - Haltless hot-swap validation
    3. boundary_test.py     - Critical drift scenarios
    4. memory_monitor.py    - Memory leak detection
    5. nomos_tui.py         - Real-time observability TUI
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_secs: float
    details: Dict
    error: Optional[str] = None


class UpstreamServerManager:
    """Manages the upstream test server lifecycle."""
    
    def __init__(self, port: int = 9090, drift_mode: str = "random"):
        self.port = port
        self.drift_mode = drift_mode
        self.process: Optional[subprocess.Popen] = None
    
    def start(self, timeout: float = 10.0) -> bool:
        """Start the upstream server and wait for it to be healthy."""
        script_path = Path(__file__).parent / "upstream_server.py"
        
        if not script_path.exists():
            print(f"ERROR: upstream_server.py not found at {script_path}")
            return False
        
        print(f"\n🚀 Starting upstream server on port {self.port} (drift={self.drift_mode})...")
        
        self.process = subprocess.Popen(
            [sys.executable, str(script_path), 
             "--port", str(self.port),
             "--drift-mode", self.drift_mode],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if sys.platform != 'win32' else None,
        )
        
        # Wait for server to become healthy
        health_url = f"http://127.0.0.1:{self.port}/health"
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                resp = requests.get(health_url, timeout=1.0)
                if resp.status_code == 200:
                    print(f"✅ Upstream server healthy at :{self.port}")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(0.2)
        
        print(f"❌ Upstream server failed to start within {timeout}s")
        self.stop()
        return False
    
    def stop(self):
        """Stop the upstream server."""
        if self.process:
            print("\n🛑 Stopping upstream server...")
            if sys.platform != 'win32':
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None


class ChaosSuiteRunner:
    """Orchestrates the full chaos test suite."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[TestResult] = []
        
    def run_test(self, name: str, command: List[str], output_file: str) -> TestResult:
        """Run a single test and capture results."""
        print(f"\n{'='*70}")
        print(f"RUNNING: {name}")
        print(f"Command: {' '.join(command)}")
        print('='*70 + "\n")
        
        start = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=Path(__file__).parent,
                capture_output=False,
                timeout=600,  # 10 minute timeout
            )
            
            duration = time.time() - start
            passed = result.returncode == 0
            
            # Load output file if exists
            details = {}
            output_path = self.output_dir / output_file
            if output_path.exists():
                with open(output_path) as f:
                    details = json.load(f)
            
            return TestResult(
                name=name,
                passed=passed,
                duration_secs=duration,
                details=details,
            )
            
        except subprocess.TimeoutExpired:
            return TestResult(
                name=name,
                passed=False,
                duration_secs=600,
                details={},
                error="Test timed out after 10 minutes"
            )
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration_secs=time.time() - start,
                details={},
                error=str(e)
            )
    
    def run_quick_suite(self) -> List[TestResult]:
        """Run quick validation tests."""
        print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                 NOMOS CHAOS TEST SUITE - QUICK MODE                   ║
║                                                                       ║
║  Running abbreviated tests for fast feedback.                         ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
        
        tests = [
            # Quick stress test (15 seconds, 5k RPS)
            {
                "name": "Stress Test (Quick)",
                "command": [
                    sys.executable, "stress_test.py",
                    "--duration", "15",
                    "--rps", "5000",
                    "--output", str(self.output_dir / "stress_quick.json")
                ],
                "output": "stress_quick.json"
            },
            # Quick hot-swap test (20 seconds, 1 swap)
            {
                "name": "Hot-Swap Validation (Quick)",
                "command": [
                    sys.executable, "hot_swap_validator.py",
                    "--duration", "20",
                    "--swap-interval", "10",
                    "--rps", "3000",
                    "--output", str(self.output_dir / "hotswap_quick.json")
                ],
                "output": "hotswap_quick.json"
            },
            # Quick boundary test (fewer requests)
            {
                "name": "Boundary Test (Quick)", 
                "command": [
                    sys.executable, "boundary_test.py",
                    "--requests", "20",
                    "--output", str(self.output_dir / "boundary_quick.json")
                ],
                "output": "boundary_quick.json"
            },
            # Quick memory check (30 seconds)
            {
                "name": "Memory Monitor (Quick)",
                "command": [
                    sys.executable, "memory_monitor.py",
                    "--duration", "30",
                    "--output", str(self.output_dir / "memory_quick.json")
                ],
                "output": "memory_quick.json"
            },
        ]
        
        for test in tests:
            result = self.run_test(test["name"], test["command"], test["output"])
            self.results.append(result)
        
        return self.results
    
    def run_full_suite(self) -> List[TestResult]:
        """Run comprehensive test suite."""
        print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                 NOMOS CHAOS TEST SUITE - FULL MODE                    ║
║                                                                       ║
║  Running comprehensive tests. This will take several minutes.         ║
║                                                                       ║
║  Tests:                                                               ║
║    - High-velocity stress test (60s @ 10k RPS)                        ║
║    - Hot-swap validation (60s, 3 swaps)                               ║
║    - Full boundary test suite (8 scenarios)                           ║
║    - Memory leak detection (5 minutes)                                ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
        
        tests = [
            # Full stress test
            {
                "name": "High-Velocity Stress Test",
                "command": [
                    sys.executable, "stress_test.py",
                    "--duration", "60",
                    "--rps", "10000",
                    "--workers", "8",
                    "--output", str(self.output_dir / "stress_full.json")
                ],
                "output": "stress_full.json"
            },
            # Full hot-swap validation
            {
                "name": "Haltless Hot-Swap Validation",
                "command": [
                    sys.executable, "hot_swap_validator.py",
                    "--duration", "60",
                    "--swap-interval", "20",
                    "--rps", "5000",
                    "--output", str(self.output_dir / "hotswap_full.json")
                ],
                "output": "hotswap_full.json"
            },
            # Full boundary test
            {
                "name": "Critical Drift Boundary Test",
                "command": [
                    sys.executable, "boundary_test.py",
                    "--requests", "100",
                    "--output", str(self.output_dir / "boundary_full.json")
                ],
                "output": "boundary_full.json"
            },
            # Full memory monitoring
            {
                "name": "Memory Leak Detection",
                "command": [
                    sys.executable, "memory_monitor.py",
                    "--duration", "300",
                    "--output", str(self.output_dir / "memory_full.json"),
                    "--plot", str(self.output_dir / "memory_plot.png")
                ],
                "output": "memory_full.json"
            },
        ]
        
        for test in tests:
            result = self.run_test(test["name"], test["command"], test["output"])
            self.results.append(result)
        
        return self.results
    
    def generate_report(self) -> Dict:
        """Generate final suite report."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        total_duration = sum(r.duration_secs for r in self.results)
        
        all_passed = passed == total
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "total_duration_secs": total_duration,
            "all_passed": all_passed,
            "tests": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration_secs": r.duration_secs,
                    "error": r.error,
                }
                for r in self.results
            ]
        }
        
        # Print summary
        status = "✅ ALL PASSED" if all_passed else "❌ SOME FAILED"
        
        print(f"""

╔═══════════════════════════════════════════════════════════════════════╗
║                    CHAOS TEST SUITE SUMMARY                           ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Timestamp:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<52} ║
║  Total Tests:        {total:>10}                                 ║
║  Passed:             {passed:>10}                                 ║
║  Failed:             {failed:>10}                                 ║
║  Duration:           {total_duration:>10.1f} seconds                            ║
╠═══════════════════════════════════════════════════════════════════════╣
║  OVERALL STATUS:     {status:>15}                            ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
        
        for r in self.results:
            emoji = "✅" if r.passed else "❌"
            print(f"  {emoji} {r.name:<40} ({r.duration_secs:.1f}s)")
            if r.error:
                print(f"     └─ Error: {r.error}")
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Nomos Chaos Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--full", "-f",
        action="store_true",
        help="Run full test suite (takes ~10 minutes)"
    )
    group.add_argument(
        "--quick", "-q",
        action="store_true",
        default=True,
        help="Run quick validation (default, ~2 minutes)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="chaos_results",
        help="Output directory for test results"
    )
    
    parser.add_argument(
        "--upstream-port",
        type=int,
        default=9090,
        help="Port for upstream test server (default: 9090)"
    )
    
    parser.add_argument(
        "--drift-mode",
        type=str,
        default="random",
        choices=["healthy", "v2", "camel", "nested", "random"],
        help="Drift mode for upstream server (default: random)"
    )
    
    parser.add_argument(
        "--skip-upstream",
        action="store_true",
        help="Skip starting upstream server (use if already running)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / args.output_dir
    runner = ChaosSuiteRunner(output_dir)
    
    # Manage upstream server lifecycle
    upstream_mgr = None
    if not args.skip_upstream:
        upstream_mgr = UpstreamServerManager(
            port=args.upstream_port,
            drift_mode=args.drift_mode
        )
        if not upstream_mgr.start():
            print("\n❌ ABORT: Could not start upstream server")
            print("   Ensure port 9090 is free or use --skip-upstream if server is running")
            sys.exit(1)
    
    try:
        if args.full:
            results = runner.run_full_suite()
        else:
            results = runner.run_quick_suite()
        
        report = runner.generate_report()
        
        # Save report
        report_path = output_dir / "suite_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to: {report_path}")
        
        # Exit code
        sys.exit(0 if report['all_passed'] else 1)
        
    finally:
        if upstream_mgr:
            upstream_mgr.stop()


if __name__ == "__main__":
    main()
