#!/usr/bin/env python3
"""
Nomos Memory Leak Monitor
==========================

Monitors RSS (Resident Set Size) memory to detect leaks in:
- WASM instance pool
- eBPF maps
- Schema store
- Connection pools

Engineering Standard: Zero memory leaks under sustained load.

Usage:
    python memory_monitor.py [--duration 300] [--interval 1]
    
Requirements:
    pip install psutil httpx matplotlib
"""

import asyncio
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import psutil
    import httpx
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil", "httpx"])
    import psutil
    import httpx


# Optional matplotlib for graphs
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

CONTROL_URL = "http://127.0.0.1:8081"
PROCESS_NAME = "nomos-core"

# Memory thresholds
LEAK_THRESHOLD_MB_PER_HOUR = 10.0  # Max acceptable growth rate
ABSOLUTE_MAX_MB = 500.0            # Absolute memory limit


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MemorySample:
    timestamp: float
    rss_mb: float
    vms_mb: float        # Virtual memory size
    shared_mb: float     # Shared memory
    private_mb: float    # Private memory (USS-like on Linux)
    heap_mb: float       # Approx heap (RSS - shared)
    requests_total: int
    wasm_calls: int
    open_fds: int
    threads: int


@dataclass
class LeakDetectionResult:
    has_leak: bool
    leak_rate_mb_per_hour: float
    total_growth_mb: float
    duration_hours: float
    confidence: float
    details: str


@dataclass 
class MemoryReport:
    duration_secs: float
    samples_collected: int
    initial_rss_mb: float
    final_rss_mb: float
    peak_rss_mb: float
    min_rss_mb: float
    avg_rss_mb: float
    growth_mb: float
    growth_rate_mb_per_hour: float
    leak_detected: bool
    leak_details: str
    samples: List[Dict]


# ============================================================================
# Memory Monitor
# ============================================================================

class MemoryMonitor:
    """Monitors Nomos memory usage over time."""
    
    def __init__(
        self,
        duration_secs: int = 300,
        sample_interval_secs: float = 1.0,
        process_name: str = PROCESS_NAME,
        leak_threshold_mb_per_hour: float = LEAK_THRESHOLD_MB_PER_HOUR,
    ):
        self.duration_secs = duration_secs
        self.sample_interval = sample_interval_secs
        self.process_name = process_name
        self.leak_threshold = leak_threshold_mb_per_hour
        self.samples: List[MemorySample] = []
        self.running = False
        self.control_client = httpx.AsyncClient(timeout=2.0)
        
    def find_nomos_process(self) -> Optional[psutil.Process]:
        """Find the Nomos process by name."""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if self.process_name in proc.info['name']:
                    return psutil.Process(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    async def get_metrics(self) -> Tuple[int, int]:
        """Get requests_total and wasm_calls from control plane."""
        try:
            response = await self.control_client.get(f"{CONTROL_URL}/v1/metrics")
            if response.status_code == 200:
                data = response.json()
                return data.get('requests_total', 0), data.get('wasm_calls', 0)
        except Exception:
            pass
        return 0, 0
    
    def sample_memory(self, proc: psutil.Process, requests: int, wasm_calls: int) -> MemorySample:
        """Take a memory sample."""
        try:
            mem = proc.memory_info()
            mem_full = proc.memory_full_info() if hasattr(proc, 'memory_full_info') else None
            
            rss_mb = mem.rss / (1024 * 1024)
            vms_mb = mem.vms / (1024 * 1024)
            shared_mb = getattr(mem, 'shared', 0) / (1024 * 1024)
            
            # Private memory (approximation)
            if mem_full and hasattr(mem_full, 'uss'):
                private_mb = mem_full.uss / (1024 * 1024)
            else:
                private_mb = rss_mb - shared_mb
            
            heap_mb = rss_mb - shared_mb
            
            return MemorySample(
                timestamp=time.time(),
                rss_mb=rss_mb,
                vms_mb=vms_mb,
                shared_mb=shared_mb,
                private_mb=private_mb,
                heap_mb=heap_mb,
                requests_total=requests,
                wasm_calls=wasm_calls,
                open_fds=proc.num_fds() if hasattr(proc, 'num_fds') else 0,
                threads=proc.num_threads(),
            )
        except psutil.NoSuchProcess:
            raise RuntimeError("Nomos process terminated")
    
    def detect_leak(self) -> LeakDetectionResult:
        """Analyze samples for memory leaks."""
        if len(self.samples) < 10:
            return LeakDetectionResult(
                has_leak=False,
                leak_rate_mb_per_hour=0,
                total_growth_mb=0,
                duration_hours=0,
                confidence=0,
                details="Insufficient samples for leak detection"
            )
        
        # Calculate time span
        first = self.samples[0]
        last = self.samples[-1]
        duration_hours = (last.timestamp - first.timestamp) / 3600
        
        if duration_hours < 0.01:  # Less than 36 seconds
            return LeakDetectionResult(
                has_leak=False,
                leak_rate_mb_per_hour=0,
                total_growth_mb=0,
                duration_hours=duration_hours,
                confidence=0,
                details="Duration too short for reliable leak detection"
            )
        
        # Simple linear regression for trend
        n = len(self.samples)
        sum_x = sum(s.timestamp - first.timestamp for s in self.samples)
        sum_y = sum(s.rss_mb for s in self.samples)
        sum_xy = sum((s.timestamp - first.timestamp) * s.rss_mb for s in self.samples)
        sum_xx = sum((s.timestamp - first.timestamp) ** 2 for s in self.samples)
        
        # Slope (growth rate in MB per second)
        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Convert to MB per hour
        growth_rate_mb_per_hour = slope * 3600
        total_growth_mb = last.rss_mb - first.rss_mb
        
        # Calculate R-squared for confidence
        mean_y = sum_y / n
        ss_tot = sum((s.rss_mb - mean_y) ** 2 for s in self.samples)
        
        if ss_tot > 0:
            intercept = (sum_y - slope * sum_x) / n
            ss_res = sum(
                (s.rss_mb - (intercept + slope * (s.timestamp - first.timestamp))) ** 2 
                for s in self.samples
            )
            r_squared = 1 - (ss_res / ss_tot)
            confidence = max(0, min(1, r_squared))
        else:
            confidence = 0
        
        # Leak detection criteria:
        # 1. Positive growth rate above threshold
        # 2. High confidence (memory is consistently growing, not fluctuating)
        has_leak = (
            growth_rate_mb_per_hour > self.leak_threshold and
            confidence > 0.7 and
            total_growth_mb > 5  # At least 5MB absolute growth
        )
        
        if has_leak:
            details = (
                f"LEAK DETECTED: Memory growing at {growth_rate_mb_per_hour:.2f} MB/hour "
                f"(threshold: {self.leak_threshold} MB/hour). "
                f"Total growth: {total_growth_mb:.2f} MB. Confidence: {confidence*100:.1f}%"
            )
        else:
            details = (
                f"No leak detected. Growth rate: {growth_rate_mb_per_hour:.2f} MB/hour. "
                f"Total growth: {total_growth_mb:.2f} MB. Confidence: {confidence*100:.1f}%"
            )
        
        return LeakDetectionResult(
            has_leak=has_leak,
            leak_rate_mb_per_hour=growth_rate_mb_per_hour,
            total_growth_mb=total_growth_mb,
            duration_hours=duration_hours,
            confidence=confidence,
            details=details,
        )
    
    async def _display_progress(self, proc: psutil.Process) -> None:
        """Display progress during monitoring."""
        start_time = time.time()
        
        while self.running:
            elapsed = time.time() - start_time
            remaining = self.duration_secs - elapsed
            
            if self.samples:
                current = self.samples[-1]
                initial = self.samples[0]
                growth = current.rss_mb - initial.rss_mb
                
                # Check for concerning growth
                growth_indicator = ""
                if growth > 50:
                    growth_indicator = "[yellow]⚠[/]"
                elif growth > 100:
                    growth_indicator = "[red]⚠⚠[/]"
                
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"RSS: {current.rss_mb:>7.2f} MB | "
                      f"Growth: {growth:>+7.2f} MB {growth_indicator} | "
                      f"Threads: {current.threads:>3} | "
                      f"FDs: {current.open_fds:>4} | "
                      f"Requests: {current.requests_total:>10,} | "
                      f"Remaining: {int(remaining):>4}s", end="", flush=True)
            
            await asyncio.sleep(1.0)
    
    async def run(self) -> MemoryReport:
        """Run the memory monitoring session."""
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    NOMOS MEMORY LEAK MONITOR                          ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Duration:           {self.duration_secs:<52} seconds ║
║  Sample Interval:    {self.sample_interval:<52} seconds ║
║  Leak Threshold:     {self.leak_threshold:<52} MB/hour ║
║  Absolute Max:       {ABSOLUTE_MAX_MB:<52} MB ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
        
        # Find process
        proc = self.find_nomos_process()
        if not proc:
            print(f"[ERROR] Process '{self.process_name}' not found. Start Nomos first.")
            return None
        
        print(f"Found Nomos process: PID {proc.pid}")
        print(f"Starting memory monitoring...\n")
        
        self.running = True
        start_time = time.time()
        
        # Start progress display
        display_task = asyncio.create_task(self._display_progress(proc))
        
        try:
            while time.time() - start_time < self.duration_secs:
                # Get metrics from control plane
                requests, wasm_calls = await self.get_metrics()
                
                # Sample memory
                try:
                    sample = self.sample_memory(proc, requests, wasm_calls)
                    self.samples.append(sample)
                    
                    # Check absolute limit
                    if sample.rss_mb > ABSOLUTE_MAX_MB:
                        print(f"\n\n[CRITICAL] RSS ({sample.rss_mb:.2f} MB) exceeded absolute limit ({ABSOLUTE_MAX_MB} MB)!")
                        break
                        
                except RuntimeError as e:
                    print(f"\n\n[ERROR] {e}")
                    break
                
                await asyncio.sleep(self.sample_interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring interrupted by user")
        finally:
            self.running = False
            display_task.cancel()
            try:
                await display_task
            except asyncio.CancelledError:
                pass
        
        await self.control_client.aclose()
        
        return self._generate_report()
    
    def _generate_report(self) -> MemoryReport:
        """Generate the memory analysis report."""
        if not self.samples:
            return None
        
        first = self.samples[0]
        last = self.samples[-1]
        duration = last.timestamp - first.timestamp
        
        rss_values = [s.rss_mb for s in self.samples]
        
        # Leak detection
        leak_result = self.detect_leak()
        
        report = MemoryReport(
            duration_secs=duration,
            samples_collected=len(self.samples),
            initial_rss_mb=first.rss_mb,
            final_rss_mb=last.rss_mb,
            peak_rss_mb=max(rss_values),
            min_rss_mb=min(rss_values),
            avg_rss_mb=sum(rss_values) / len(rss_values),
            growth_mb=last.rss_mb - first.rss_mb,
            growth_rate_mb_per_hour=leak_result.leak_rate_mb_per_hour,
            leak_detected=leak_result.has_leak,
            leak_details=leak_result.details,
            samples=[
                {
                    "timestamp": s.timestamp,
                    "rss_mb": s.rss_mb,
                    "heap_mb": s.heap_mb,
                    "requests": s.requests_total,
                    "wasm_calls": s.wasm_calls,
                    "threads": s.threads,
                    "fds": s.open_fds,
                }
                for s in self.samples
            ]
        )
        
        # Print report
        leak_status = "❌ LEAK DETECTED" if report.leak_detected else "✅ NO LEAK"
        
        print(f"""

╔═══════════════════════════════════════════════════════════════════════╗
║                    MEMORY ANALYSIS REPORT                             ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Duration:           {report.duration_secs:>10.1f} seconds                            ║
║  Samples Collected:  {report.samples_collected:>10}                                 ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Initial RSS:        {report.initial_rss_mb:>10.2f} MB                                ║
║  Final RSS:          {report.final_rss_mb:>10.2f} MB                                ║
║  Peak RSS:           {report.peak_rss_mb:>10.2f} MB                                ║
║  Min RSS:            {report.min_rss_mb:>10.2f} MB                                ║
║  Average RSS:        {report.avg_rss_mb:>10.2f} MB                                ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Total Growth:       {report.growth_mb:>+10.2f} MB                                ║
║  Growth Rate:        {report.growth_rate_mb_per_hour:>10.2f} MB/hour                          ║
╠═══════════════════════════════════════════════════════════════════════╣
║  LEAK STATUS:        {leak_status:>15}                            ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
        
        print(f"\n{report.leak_details}")
        
        # Memory per request analysis
        if last.requests_total > first.requests_total:
            requests_processed = last.requests_total - first.requests_total
            bytes_per_request = (report.growth_mb * 1024 * 1024) / requests_processed
            print(f"\nMemory per request: {bytes_per_request:.2f} bytes")
            print(f"Requests processed: {requests_processed:,}")
        
        return report
    
    def plot_memory(self, output_path: str = "memory_plot.png") -> None:
        """Generate memory plot (requires matplotlib)."""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available. Install with: pip install matplotlib")
            return
        
        if not self.samples:
            print("No samples to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        start_time = self.samples[0].timestamp
        times = [(s.timestamp - start_time) / 60 for s in self.samples]  # Minutes
        rss = [s.rss_mb for s in self.samples]
        heap = [s.heap_mb for s in self.samples]
        requests = [s.requests_total for s in self.samples]
        
        # Memory plot
        ax1.fill_between(times, rss, alpha=0.3, label='RSS')
        ax1.plot(times, rss, 'b-', linewidth=1, label='RSS (MB)')
        ax1.plot(times, heap, 'g--', linewidth=1, alpha=0.7, label='Heap (MB)')
        ax1.axhline(y=ABSOLUTE_MAX_MB, color='r', linestyle='--', label=f'Limit ({ABSOLUTE_MAX_MB} MB)')
        ax1.set_ylabel('Memory (MB)')
        ax1.set_title('Nomos Memory Usage Over Time')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Requests plot
        ax2.plot(times, requests, 'purple', linewidth=1)
        ax2.set_ylabel('Total Requests')
        ax2.set_xlabel('Time (minutes)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"\nMemory plot saved to: {output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nomos Memory Leak Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=300,
        help="Monitoring duration in seconds (default: 300)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=1.0,
        help="Sample interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--plot", "-p",
        type=str,
        help="Output PNG file for memory plot"
    )
    parser.add_argument(
        "--leak-threshold",
        type=float,
        default=10.0,
        help="Leak threshold in MB/hour (default: 10.0)"
    )
    
    args = parser.parse_args()
    
    # Update threshold if specified
    leak_threshold = args.leak_threshold
    
    monitor = MemoryMonitor(
        duration_secs=args.duration,
        sample_interval_secs=args.interval,
        leak_threshold_mb_per_hour=leak_threshold,
    )
    
    report = asyncio.run(monitor.run())
    
    if report is None:
        sys.exit(1)
    
    # Save report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "duration_secs": report.duration_secs,
                "samples_collected": report.samples_collected,
                "initial_rss_mb": report.initial_rss_mb,
                "final_rss_mb": report.final_rss_mb,
                "peak_rss_mb": report.peak_rss_mb,
                "min_rss_mb": report.min_rss_mb,
                "avg_rss_mb": report.avg_rss_mb,
                "growth_mb": report.growth_mb,
                "growth_rate_mb_per_hour": report.growth_rate_mb_per_hour,
                "leak_detected": report.leak_detected,
                "leak_details": report.leak_details,
                "samples": report.samples,
            }, f, indent=2)
        print(f"\nReport saved to: {args.output}")
    
    # Generate plot
    if args.plot:
        monitor.plot_memory(args.plot)
    
    # Exit code
    if report.leak_detected:
        print("\n[FAILURE] Memory leak detected!")
        sys.exit(1)
    else:
        print("\n[SUCCESS] No memory leaks detected!")
        sys.exit(0)


if __name__ == "__main__":
    main()
