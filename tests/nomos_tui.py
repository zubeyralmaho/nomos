#!/usr/bin/env python3
"""
Nomos Real-Time Observability TUI
==================================

Terminal User Interface for monitoring Nomos proxy metrics in real-time.
Polls :8081/v1/metrics and displays live counters.

Usage:
    python nomos_tui.py [--control-url http://127.0.0.1:8081]
    
Requirements:
    pip install rich httpx psutil
"""

import asyncio
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime

try:
    import httpx
    import psutil
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.style import Style
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "httpx", "psutil"])
    import httpx
    import psutil
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.style import Style


# ============================================================================
# Metrics Data Structures
# ============================================================================

@dataclass
class CoreMetricsSnapshot:
    core_id: int
    requests: int
    healed: int
    avg_healing_ns: int


@dataclass
class MetricsSnapshot:
    timestamp: float
    requests_total: int
    requests_healed: int
    healing_rate: float
    avg_healing_us: int
    p99_healing_us: int
    wasm_calls: int
    wasm_errors: int
    uptime_secs: int
    num_cores: int
    per_core: List[CoreMetricsSnapshot]
    healer_version: int
    ebpf_fast_path: int = 0
    ebpf_slow_path: int = 0
    ebpf_dropped: int = 0
    ebpf_bytes: int = 0


@dataclass
class SystemMetrics:
    nomos_pid: Optional[int]
    rss_mb: float
    cpu_percent: float
    open_fds: int
    threads: int


# ============================================================================
# Metrics Collector
# ============================================================================

class MetricsCollector:
    """Collects metrics from Nomos control plane and system."""
    
    def __init__(self, control_url: str, nomos_process_name: str = "nomos-core"):
        self.control_url = control_url.rstrip('/')
        self.nomos_process_name = nomos_process_name
        self.client = httpx.AsyncClient(timeout=2.0)
        self.history: List[MetricsSnapshot] = []
        self.max_history = 300  # 5 minutes at 1 sample/sec
        
    async def fetch_metrics(self) -> Optional[MetricsSnapshot]:
        """Fetch metrics from control plane API."""
        try:
            response = await self.client.get(f"{self.control_url}/v1/metrics")
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            per_core = [
                CoreMetricsSnapshot(
                    core_id=c['core_id'],
                    requests=c['requests'],
                    healed=c['healed'],
                    avg_healing_ns=c['avg_healing_ns']
                )
                for c in data.get('per_core', [])
            ]
            
            ebpf = data.get('ebpf', {})
            
            snapshot = MetricsSnapshot(
                timestamp=time.time(),
                requests_total=data['requests_total'],
                requests_healed=data['requests_healed'],
                healing_rate=data['healing_rate'],
                avg_healing_us=data['avg_healing_us'],
                p99_healing_us=data['p99_healing_us'],
                wasm_calls=data['wasm_calls'],
                wasm_errors=data['wasm_errors'],
                uptime_secs=data['uptime_secs'],
                num_cores=data['num_cores'],
                per_core=per_core,
                healer_version=data.get('healer_version', 0),
                ebpf_fast_path=ebpf.get('fast_path_packets', 0),
                ebpf_slow_path=ebpf.get('slow_path_packets', 0),
                ebpf_dropped=ebpf.get('dropped_packets', 0),
                ebpf_bytes=ebpf.get('bytes_processed', 0),
            )
            
            self.history.append(snapshot)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            return snapshot
            
        except Exception as e:
            return None
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get system metrics for Nomos process."""
        nomos_pid = None
        rss_mb = 0.0
        cpu_percent = 0.0
        open_fds = 0
        threads = 0
        
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent', 'num_fds', 'num_threads']):
            try:
                if self.nomos_process_name in proc.info['name']:
                    nomos_pid = proc.info['pid']
                    mem = proc.info['memory_info']
                    rss_mb = mem.rss / (1024 * 1024) if mem else 0.0
                    cpu_percent = proc.info['cpu_percent'] or 0.0
                    open_fds = proc.info.get('num_fds', 0) or 0
                    threads = proc.info.get('num_threads', 0) or 0
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return SystemMetrics(
            nomos_pid=nomos_pid,
            rss_mb=rss_mb,
            cpu_percent=cpu_percent,
            open_fds=open_fds,
            threads=threads
        )
    
    def get_rps(self) -> float:
        """Calculate current requests per second from history."""
        if len(self.history) < 2:
            return 0.0
        
        latest = self.history[-1]
        prev = self.history[-2]
        
        delta_time = latest.timestamp - prev.timestamp
        delta_requests = latest.requests_total - prev.requests_total
        
        if delta_time <= 0:
            return 0.0
        
        return delta_requests / delta_time
    
    def get_healing_rate_trend(self) -> str:
        """Get healing rate trend indicator."""
        if len(self.history) < 10:
            return "─"
        
        recent = [s.healing_rate for s in self.history[-5:]]
        older = [s.healing_rate for s in self.history[-10:-5]]
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        if recent_avg > older_avg + 0.05:
            return "↑"
        elif recent_avg < older_avg - 0.05:
            return "↓"
        return "─"
    
    async def close(self):
        await self.client.aclose()


# ============================================================================
# TUI Dashboard
# ============================================================================

class NomosTUI:
    """Terminal User Interface for Nomos monitoring."""
    
    def __init__(self, control_url: str):
        self.control_url = control_url
        self.collector = MetricsCollector(control_url)
        self.console = Console()
        self.running = False
        self.last_metrics: Optional[MetricsSnapshot] = None
        self.last_system: Optional[SystemMetrics] = None
        self.alerts: List[str] = []
        
    def _create_header(self) -> Panel:
        """Create header panel."""
        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(justify="center")
        grid.add_column(justify="right")
        
        status = "[green]● CONNECTED[/]" if self.last_metrics else "[red]● DISCONNECTED[/]"
        
        grid.add_row(
            "[bold cyan]NOMOS[/] [dim]Real-Time Monitor[/]",
            status,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return Panel(grid, style="bold white on blue")
    
    def _create_summary_panel(self) -> Panel:
        """Create summary metrics panel."""
        if not self.last_metrics:
            return Panel("[dim]Waiting for data...[/]", title="Summary")
        
        m = self.last_metrics
        rps = self.collector.get_rps()
        trend = self.collector.get_healing_rate_trend()
        
        # Nomos Law check
        nomos_status = "[green]✅ PASS[/]" if m.p99_healing_us < 1000 else "[red]❌ FAIL[/]"
        
        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(justify="right", style="dim")
        table.add_column(justify="left", style="bold")
        table.add_column(justify="right", style="dim")
        table.add_column(justify="left", style="bold")
        
        table.add_row(
            "Uptime:", f"{m.uptime_secs // 3600}h {(m.uptime_secs % 3600) // 60}m {m.uptime_secs % 60}s",
            "Healer Version:", f"v{m.healer_version}"
        )
        table.add_row(
            "Total Requests:", f"{m.requests_total:,}",
            "Current RPS:", f"{rps:,.0f}"
        )
        table.add_row(
            "Healed Requests:", f"{m.requests_healed:,}",
            f"Healing Rate {trend}:", f"{m.healing_rate * 100:.1f}%"
        )
        table.add_row(
            "WASM Calls:", f"{m.wasm_calls:,}",
            "WASM Errors:", f"{m.wasm_errors:,}"
        )
        table.add_row(
            "Avg Healing:", f"{m.avg_healing_us}µs",
            f"p99 Healing:", f"{m.p99_healing_us}µs {nomos_status}"
        )
        
        return Panel(table, title="[bold]Summary[/]", border_style="green")
    
    def _create_ebpf_panel(self) -> Panel:
        """Create eBPF statistics panel."""
        if not self.last_metrics:
            return Panel("[dim]No eBPF data[/]", title="eBPF Statistics")
        
        m = self.last_metrics
        total_packets = m.ebpf_fast_path + m.ebpf_slow_path
        fast_ratio = m.ebpf_fast_path / max(total_packets, 1) * 100
        
        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(justify="right", style="dim")
        table.add_column(justify="left")
        
        # Fast path bar
        fast_bar = "█" * int(fast_ratio / 5) + "░" * (20 - int(fast_ratio / 5))
        
        table.add_row("Fast Path Packets:", f"[green]{m.ebpf_fast_path:,}[/]")
        table.add_row("Slow Path (Userspace):", f"[yellow]{m.ebpf_slow_path:,}[/]")
        table.add_row("Dropped Packets:", f"[red]{m.ebpf_dropped:,}[/]")
        table.add_row("Bytes Processed:", f"{m.ebpf_bytes / (1024*1024):.2f} MB")
        table.add_row("Fast Path Ratio:", f"[cyan]{fast_bar}[/] {fast_ratio:.1f}%")
        
        return Panel(table, title="[bold]eBPF Statistics[/]", border_style="cyan")
    
    def _create_per_core_panel(self) -> Panel:
        """Create per-core metrics panel."""
        if not self.last_metrics or not self.last_metrics.per_core:
            return Panel("[dim]No per-core data[/]", title="Per-Core Metrics")
        
        table = Table(expand=True, box=None)
        table.add_column("Core", justify="center", style="dim")
        table.add_column("Requests", justify="right")
        table.add_column("Healed", justify="right")
        table.add_column("Avg Healing", justify="right")
        table.add_column("Load", justify="left")
        
        max_requests = max(c.requests for c in self.last_metrics.per_core) or 1
        
        for core in self.last_metrics.per_core:
            load_pct = core.requests / max_requests * 100
            load_bar = "█" * int(load_pct / 10) + "░" * (10 - int(load_pct / 10))
            
            heal_color = "green" if core.avg_healing_ns < 1000 else "yellow" if core.avg_healing_ns < 5000 else "red"
            
            table.add_row(
                f"#{core.core_id}",
                f"{core.requests:,}",
                f"{core.healed:,}",
                f"[{heal_color}]{core.avg_healing_ns}ns[/]",
                f"[blue]{load_bar}[/]"
            )
        
        return Panel(table, title="[bold]Per-Core Utilization[/]", border_style="blue")
    
    def _create_system_panel(self) -> Panel:
        """Create system metrics panel."""
        if not self.last_system:
            return Panel("[dim]No system data[/]", title="System")
        
        s = self.last_system
        
        # Memory leak detection (simple heuristic)
        memory_status = "[green]OK[/]"
        if len(self.collector.history) > 60:
            first_rss = self._get_initial_rss()
            if first_rss and s.rss_mb > first_rss * 1.5:
                memory_status = "[red]⚠ LEAK?[/]"
        
        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(justify="right", style="dim")
        table.add_column(justify="left")
        
        table.add_row("PID:", f"{s.nomos_pid or 'N/A'}")
        table.add_row("RSS Memory:", f"{s.rss_mb:.1f} MB {memory_status}")
        table.add_row("CPU Usage:", f"{s.cpu_percent:.1f}%")
        table.add_row("Threads:", f"{s.threads}")
        table.add_row("Open FDs:", f"{s.open_fds}")
        
        return Panel(table, title="[bold]System Resources[/]", border_style="magenta")
    
    def _get_initial_rss(self) -> Optional[float]:
        """Get initial RSS from first system metric."""
        # We don't track system metrics in history, so return None
        return None
    
    def _create_alerts_panel(self) -> Panel:
        """Create alerts panel."""
        if not self.alerts:
            return Panel("[green]No active alerts[/]", title="Alerts", border_style="green")
        
        alert_text = "\n".join(f"[red]⚠[/] {alert}" for alert in self.alerts[-5:])
        return Panel(alert_text, title="[bold red]Alerts[/]", border_style="red")
    
    def _check_alerts(self):
        """Check for alert conditions."""
        self.alerts = []
        
        if self.last_metrics:
            m = self.last_metrics
            
            # Nomos Law violation
            if m.p99_healing_us >= 1000:
                self.alerts.append(f"Nomos Law VIOLATION: p99 = {m.p99_healing_us}µs (target < 1000µs)")
            
            # High error rate
            if m.wasm_calls > 0:
                error_rate = m.wasm_errors / m.wasm_calls
                if error_rate > 0.01:
                    self.alerts.append(f"High WASM error rate: {error_rate * 100:.2f}%")
            
            # Low healing rate with traffic
            if m.requests_total > 1000 and m.healing_rate < 0.1:
                self.alerts.append(f"Low healing rate: {m.healing_rate * 100:.1f}% (expected ~80%)")
            
            # Dropped packets
            if m.ebpf_dropped > 0:
                self.alerts.append(f"eBPF dropped {m.ebpf_dropped:,} packets!")
        
        if self.last_system:
            s = self.last_system
            
            # High memory usage
            if s.rss_mb > 500:
                self.alerts.append(f"High memory usage: {s.rss_mb:.1f} MB")
            
            # High CPU
            if s.cpu_percent > 90:
                self.alerts.append(f"High CPU usage: {s.cpu_percent:.1f}%")
    
    def _create_layout(self) -> Layout:
        """Create the TUI layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=6)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="summary"),
            Layout(name="ebpf"),
        )
        
        layout["right"].split_column(
            Layout(name="per_core"),
            Layout(name="system"),
        )
        
        layout["header"].update(self._create_header())
        layout["summary"].update(self._create_summary_panel())
        layout["ebpf"].update(self._create_ebpf_panel())
        layout["per_core"].update(self._create_per_core_panel())
        layout["system"].update(self._create_system_panel())
        layout["footer"].update(self._create_alerts_panel())
        
        return layout
    
    async def run(self):
        """Run the TUI dashboard."""
        self.running = True
        
        self.console.print("""
[bold cyan]
╔═══════════════════════════════════════════════════════════════╗
║           NOMOS REAL-TIME OBSERVABILITY TUI                   ║
║                    "It never stops."                          ║
╚═══════════════════════════════════════════════════════════════╝
[/]
""")
        self.console.print(f"[dim]Connecting to {self.control_url}...[/]")
        
        try:
            with Live(self._create_layout(), console=self.console, refresh_per_second=2) as live:
                while self.running:
                    # Fetch metrics
                    self.last_metrics = await self.collector.fetch_metrics()
                    self.last_system = self.collector.get_system_metrics()
                    
                    # Check alerts
                    self._check_alerts()
                    
                    # Update display
                    live.update(self._create_layout())
                    
                    await asyncio.sleep(0.5)
                    
        except KeyboardInterrupt:
            self.running = False
        finally:
            await self.collector.close()
    
    def stop(self):
        self.running = False


# ============================================================================
# Simple Text Dashboard (fallback)
# ============================================================================

async def simple_dashboard(control_url: str):
    """Simple text-based dashboard for terminals without Rich support."""
    collector = MetricsCollector(control_url)
    
    print(f"\n{'='*60}")
    print("NOMOS METRICS MONITOR (Simple Mode)")
    print(f"Control URL: {control_url}")
    print(f"{'='*60}\n")
    
    try:
        while True:
            metrics = await collector.fetch_metrics()
            system = collector.get_system_metrics()
            rps = collector.get_rps()
            
            if metrics:
                os.system('clear' if os.name == 'posix' else 'cls')
                print(f"[{datetime.now().strftime('%H:%M:%S')}] NOMOS METRICS")
                print("-" * 50)
                print(f"Requests:     {metrics.requests_total:>12,}")
                print(f"Healed:       {metrics.requests_healed:>12,}")
                print(f"Healing Rate: {metrics.healing_rate * 100:>11.1f}%")
                print(f"Current RPS:  {rps:>12,.0f}")
                print(f"Avg Healing:  {metrics.avg_healing_us:>11}µs")
                print(f"p99 Healing:  {metrics.p99_healing_us:>11}µs")
                print(f"WASM Calls:   {metrics.wasm_calls:>12,}")
                print(f"WASM Errors:  {metrics.wasm_errors:>12,}")
                print("-" * 50)
                print(f"eBPF Fast:    {metrics.ebpf_fast_path:>12,}")
                print(f"eBPF Slow:    {metrics.ebpf_slow_path:>12,}")
                print(f"Dropped:      {metrics.ebpf_dropped:>12,}")
                print("-" * 50)
                if system.nomos_pid:
                    print(f"PID:          {system.nomos_pid:>12}")
                    print(f"RSS Memory:   {system.rss_mb:>11.1f}MB")
                    print(f"CPU:          {system.cpu_percent:>11.1f}%")
                print("-" * 50)
                
                # Nomos Law check
                nomos_ok = "✅ PASS" if metrics.p99_healing_us < 1000 else "❌ FAIL"
                print(f"Nomos Law:    {nomos_ok:>12}")
            else:
                print(".", end="", flush=True)
            
            await asyncio.sleep(1.0)
            
    except KeyboardInterrupt:
        pass
    finally:
        await collector.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nomos Real-Time Observability TUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--control-url", "-c",
        default="http://127.0.0.1:8081",
        help="Control plane URL (default: http://127.0.0.1:8081)"
    )
    parser.add_argument(
        "--simple", "-s",
        action="store_true",
        help="Use simple text mode instead of TUI"
    )
    
    args = parser.parse_args()
    
    try:
        if args.simple:
            asyncio.run(simple_dashboard(args.control_url))
        else:
            tui = NomosTUI(args.control_url)
            asyncio.run(tui.run())
    except KeyboardInterrupt:
        print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()
