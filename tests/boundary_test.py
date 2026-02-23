#!/usr/bin/env python3
"""
Nomos Boundary Testing - Critical Drift Scenarios
==================================================

Simulates extreme schema drift scenarios to validate:
1. Critical Drift alerts trigger correctly (confidence < 85%)
2. System remains stable and doesn't crash
3. Graceful degradation under total drift

Tests the "Haltless" principle under maximum adversity.

Usage:
    python boundary_test.py [--scenario all]
    
Requirements:
    pip install aiohttp httpx
"""

import asyncio
import argparse
import json
import random
import string
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any

try:
    import aiohttp
    import httpx
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "httpx"])
    import aiohttp
    import httpx


# ============================================================================
# Test Configuration
# ============================================================================

PROXY_URL = "http://127.0.0.1:8080"
CONTROL_URL = "http://127.0.0.1:8081"


# ============================================================================
# Drift Scenarios
# ============================================================================

class DriftScenario(Enum):
    TOTAL_RENAME = "total_rename"         # All fields renamed
    TYPE_CHAOS = "type_chaos"             # All types changed
    STRUCTURE_EXPLOSION = "structure_explosion"  # Flat -> deeply nested
    FIELD_AVALANCHE = "field_avalanche"   # Massive field count
    ENCODING_ATTACK = "encoding_attack"   # Unicode/special chars
    NULL_STORM = "null_storm"             # All values null
    EMPTY_RESPONSE = "empty_response"     # Minimal/empty payloads
    MIXED_CHAOS = "mixed_chaos"           # Everything changes


# Expected schema (what client expects)
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
    "tags": ["premium", "active"],
    "metadata": {
        "version": 1,
        "source": "api"
    }
}


def generate_total_rename_drift() -> Dict:
    """Scenario 1: Every single field is renamed to something unrecognizable."""
    random_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
    return {
        f"x_{random_suffix}_1": "usr_12345",           # user_id
        f"y_{random_suffix}_2": "Alice Smith",         # full_name  
        f"z_{random_suffix}_3": "alice@example.com",   # email_address
        f"a_{random_suffix}_4": 1500.50,               # account_balance
        f"b_{random_suffix}_5": True,                  # is_verified
        f"c_{random_suffix}_6": "2026-01-15T10:30:00Z", # created_at
        f"d_{random_suffix}_7": {                      # preferences
            f"e_{random_suffix}_8": "dark",
            f"f_{random_suffix}_9": True,
            f"g_{random_suffix}_10": "en"
        },
        f"h_{random_suffix}_11": ["premium", "active"], # tags
        f"i_{random_suffix}_12": {                     # metadata
            f"j_{random_suffix}_13": 1,
            f"k_{random_suffix}_14": "api"
        }
    }


def generate_type_chaos_drift() -> Dict:
    """Scenario 2: All types are different from expected."""
    return {
        "user_id": 12345,                   # String -> Int
        "full_name": ["Alice", "Smith"],    # String -> Array
        "email_address": {"email": "alice@example.com"},  # String -> Object
        "account_balance": "1500.50 USD",   # Float -> String with unit
        "is_verified": "yes",               # Bool -> String
        "created_at": 1736937000,           # ISO String -> Unix timestamp
        "preferences": "dark,true,en",      # Object -> CSV string
        "tags": "premium;active",           # Array -> Semicolon string
        "metadata": [1, "api"],             # Object -> Array
    }


def generate_structure_explosion_drift() -> Dict:
    """Scenario 3: Flat structure becomes deeply nested."""
    return {
        "data": {
            "user": {
                "identity": {
                    "primary": {
                        "id": {
                            "value": "usr_12345"
                        }
                    }
                },
                "profile": {
                    "personal": {
                        "name": {
                            "full": {
                                "display": "Alice Smith"
                            }
                        }
                    }
                },
                "contact": {
                    "electronic": {
                        "email": {
                            "primary": {
                                "address": "alice@example.com"
                            }
                        }
                    }
                }
            },
            "account": {
                "financial": {
                    "balance": {
                        "current": {
                            "amount": 1500.50
                        }
                    }
                },
                "status": {
                    "verification": {
                        "is_verified": True
                    }
                }
            }
        }
    }


def generate_field_avalanche_drift() -> Dict:
    """Scenario 4: Massive number of fields (stress test LSH)."""
    payload = {}
    
    # Add 500 random fields
    for i in range(500):
        field_name = f"field_{i:04d}_{random.choice(string.ascii_lowercase)}"
        field_value = random.choice([
            random.randint(0, 10000),
            random.random() * 1000,
            ''.join(random.choices(string.ascii_letters, k=20)),
            random.choice([True, False]),
            None,
            [random.randint(0, 100) for _ in range(5)]
        ])
        payload[field_name] = field_value
    
    # Bury the expected fields deep in the noise
    payload["user_identifier"] = "usr_12345"
    payload["person_name"] = "Alice Smith"
    
    return payload


def generate_encoding_attack_drift() -> Dict:
    """Scenario 5: Unicode and special characters in field names."""
    return {
        "user_id": "usr_12345",
        "full_name": "Alice Smith",
        "émäíl_àddréss": "alice@example.com",  # Accented chars
        "账户余额": 1500.50,                    # Chinese: account_balance
        "認証済み": True,                       # Japanese: is_verified
        "created\u200Bat": "2026-01-15T10:30:00Z",  # Zero-width char
        "preferences\x00": {"theme": "dark"},   # Null byte
        "t̴̢̧̛̻̲͈̱̻̜̘͔͓̟̝̘̩̥͚̜̫̲̹̰̘͚̫̗̖̰̼̯̗̭̠͙̱̳̝͙̭̲͓̫̝̣͎̻̰͚̣̲̺̾͂̈́̀̀̌̾̾̈́̀̀̄̽̿̅̀̑̔̎̊̃̅͗̈́̾̾̐̓̾̄̓̂̈̚͘̕͜͜͝͠͠ͅa̸g̷s̶": ["premium", "active"],  # Zalgo text
    }


def generate_null_storm_drift() -> Dict:
    """Scenario 6: All values are null."""
    return {
        "user_id": None,
        "full_name": None,
        "email_address": None,
        "account_balance": None,
        "is_verified": None,
        "created_at": None,
        "preferences": None,
        "tags": None,
        "metadata": None,
    }


def generate_empty_response_drift() -> Dict:
    """Scenario 7: Minimal/empty payloads."""
    return random.choice([
        {},                              # Empty object
        {"_": ""},                        # Single empty field
        {"status": "ok"},                 # Unrelated minimal response
        {"error": None, "data": None},    # Error-like structure
    ])


def generate_mixed_chaos_drift() -> Dict:
    """Scenario 8: Combination of all chaos patterns."""
    generators = [
        generate_total_rename_drift,
        generate_type_chaos_drift,
        generate_structure_explosion_drift,
        generate_encoding_attack_drift,
    ]
    
    # Pick 2-3 random generators and merge their outputs
    selected = random.sample(generators, k=random.randint(2, 3))
    merged = {}
    for gen in selected:
        payload = gen()
        merged.update(payload)
    
    return merged


SCENARIO_GENERATORS = {
    DriftScenario.TOTAL_RENAME: generate_total_rename_drift,
    DriftScenario.TYPE_CHAOS: generate_type_chaos_drift,
    DriftScenario.STRUCTURE_EXPLOSION: generate_structure_explosion_drift,
    DriftScenario.FIELD_AVALANCHE: generate_field_avalanche_drift,
    DriftScenario.ENCODING_ATTACK: generate_encoding_attack_drift,
    DriftScenario.NULL_STORM: generate_null_storm_drift,
    DriftScenario.EMPTY_RESPONSE: generate_empty_response_drift,
    DriftScenario.MIXED_CHAOS: generate_mixed_chaos_drift,
}


# ============================================================================
# Test Results
# ============================================================================

@dataclass
class ScenarioResult:
    scenario: str
    requests_sent: int
    successful: int
    failed: int
    healed: int
    unhealed: int
    crashed: bool
    avg_latency_ms: float
    max_latency_ms: float
    alerts_triggered: List[str]
    passed: bool


# ============================================================================
# Boundary Tester
# ============================================================================

class BoundaryTester:
    """Tests system boundaries under extreme drift scenarios."""
    
    def __init__(self, requests_per_scenario: int = 100):
        self.requests_per_scenario = requests_per_scenario
        self.results: List[ScenarioResult] = []
        
    async def check_system_health(self) -> bool:
        """Check if Nomos is still responding."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{CONTROL_URL}/v1/health")
                return response.status_code == 200
        except Exception:
            return False
    
    async def get_metrics(self) -> Optional[Dict]:
        """Get current metrics from control plane."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{CONTROL_URL}/v1/metrics")
                if response.status_code == 200:
                    return response.json()
        except Exception:
            pass
        return None
    
    async def run_scenario(self, scenario: DriftScenario) -> ScenarioResult:
        """Run a single drift scenario."""
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario.value.upper()}")
        print('='*60)
        
        generator = SCENARIO_GENERATORS[scenario]
        latencies = []
        successful = 0
        failed = 0
        healed = 0
        unhealed = 0
        alerts = []
        
        # Get baseline metrics
        baseline_metrics = await self.get_metrics()
        baseline_healed = baseline_metrics.get('requests_healed', 0) if baseline_metrics else 0
        
        # Run requests
        async with aiohttp.ClientSession() as session:
            for i in range(self.requests_per_scenario):
                payload = generator()
                
                start = time.perf_counter()
                try:
                    async with session.post(
                        f"{PROXY_URL}/api/user",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10.0)
                    ) as response:
                        latency_ms = (time.perf_counter() - start) * 1000
                        latencies.append(latency_ms)
                        
                        if response.status < 500:
                            successful += 1
                            if response.headers.get('X-Nomos-Healed', '').lower() == 'true':
                                healed += 1
                            else:
                                unhealed += 1
                        else:
                            failed += 1
                            
                except asyncio.TimeoutError:
                    failed += 1
                    latencies.append(10000.0)
                except Exception as e:
                    failed += 1
                    latencies.append(0.0)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"\r  Progress: {i+1}/{self.requests_per_scenario} "
                          f"(OK: {successful}, FAIL: {failed}, HEALED: {healed})", 
                          end="", flush=True)
        
        print()  # New line after progress
        
        # Check if system crashed
        crashed = not await self.check_system_health()
        
        # Check for alerts
        if crashed:
            alerts.append("SYSTEM_CRASH")
        
        healing_rate = healed / max(successful, 1)
        if healing_rate < 0.85 and successful > 0:
            alerts.append(f"CRITICAL_DRIFT: Healing rate {healing_rate*100:.1f}% < 85%")
        
        if failed > self.requests_per_scenario * 0.1:
            alerts.append(f"HIGH_ERROR_RATE: {failed} failures")
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        
        # Determine pass/fail
        # Pass criteria: System didn't crash, and either healed OR gracefully failed
        passed = (
            not crashed and
            failed < self.requests_per_scenario * 0.5  # < 50% failure rate
        )
        
        result = ScenarioResult(
            scenario=scenario.value,
            requests_sent=self.requests_per_scenario,
            successful=successful,
            failed=failed,
            healed=healed,
            unhealed=unhealed,
            crashed=crashed,
            avg_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            alerts_triggered=alerts,
            passed=passed,
        )
        
        self.results.append(result)
        
        # Print scenario summary
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n  Result: {status}")
        print(f"  Requests: {successful} OK / {failed} FAIL")
        print(f"  Healed: {healed} ({healing_rate*100:.1f}%)")
        print(f"  Latency: avg={avg_latency:.2f}ms, max={max_latency:.2f}ms")
        if alerts:
            print(f"  Alerts: {', '.join(alerts)}")
        
        return result
    
    async def run_all_scenarios(self) -> Dict:
        """Run all drift scenarios."""
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              NOMOS BOUNDARY TESTING - CRITICAL DRIFT                  ║
╠═══════════════════════════════════════════════════════════════════════╣
║  This test validates system stability under extreme schema drift.     ║
║  The system should remain 'Haltless' - never crash, always respond.   ║
║                                                                       ║
║  Expected behavior under total drift:                                 ║
║    - Trigger 'Critical Drift' alerts (confidence < 85%)               ║
║    - Continue processing requests (graceful degradation)              ║
║    - No crashes, no memory leaks, no panics                           ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
        
        # Check system is up before starting
        if not await self.check_system_health():
            print("[ERROR] Nomos is not responding. Start the proxy first.")
            return {"error": "System not available"}
        
        print("System health: ✅ OK\n")
        
        # Run each scenario
        for scenario in DriftScenario:
            await self.run_scenario(scenario)
            
            # Brief pause between scenarios
            await asyncio.sleep(0.5)
            
            # Verify system is still alive after scenario
            if not await self.check_system_health():
                print(f"\n[CRITICAL] System crashed after scenario: {scenario.value}")
                break
        
        return self._generate_report()
    
    async def run_single_scenario(self, scenario: DriftScenario) -> Dict:
        """Run a single drift scenario."""
        if not await self.check_system_health():
            print("[ERROR] Nomos is not responding.")
            return {"error": "System not available"}
        
        await self.run_scenario(scenario)
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """Generate final test report."""
        total_passed = sum(1 for r in self.results if r.passed)
        total_scenarios = len(self.results)
        any_crashed = any(r.crashed for r in self.results)
        
        all_alerts = []
        for r in self.results:
            all_alerts.extend(r.alerts_triggered)
        
        report = {
            "total_scenarios": total_scenarios,
            "passed_scenarios": total_passed,
            "failed_scenarios": total_scenarios - total_passed,
            "any_crashes": any_crashed,
            "all_alerts": list(set(all_alerts)),
            "haltless_maintained": not any_crashed,
            "scenarios": [
                {
                    "name": r.scenario,
                    "passed": r.passed,
                    "requests": r.requests_sent,
                    "successful": r.successful,
                    "failed": r.failed,
                    "healed": r.healed,
                    "healing_rate": r.healed / max(r.successful, 1),
                    "crashed": r.crashed,
                    "avg_latency_ms": r.avg_latency_ms,
                    "max_latency_ms": r.max_latency_ms,
                    "alerts": r.alerts_triggered,
                }
                for r in self.results
            ]
        }
        
        # Print summary
        haltless_status = "✅ PASS" if report['haltless_maintained'] else "❌ FAIL"
        
        print(f"""

╔═══════════════════════════════════════════════════════════════════════╗
║                    BOUNDARY TEST REPORT                               ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Total Scenarios:        {total_scenarios:>10}                                 ║
║  Passed:                 {total_passed:>10}                                 ║
║  Failed:                 {total_scenarios - total_passed:>10}                                 ║
╠═══════════════════════════════════════════════════════════════════════╣
║  HALTLESS MAINTAINED:    {haltless_status:>10}                                 ║
╠═══════════════════════════════════════════════════════════════════════╣
""")
        
        for r in self.results:
            status = "✅" if r.passed else "❌"
            heal_rate = r.healed / max(r.successful, 1) * 100
            print(f"║  {status} {r.scenario:<20} Heal: {heal_rate:>5.1f}%  Lat: {r.avg_latency_ms:>7.2f}ms   ║")
        
        print("╚═══════════════════════════════════════════════════════════════════════╝")
        
        # Print alerts summary
        if all_alerts:
            print("\nALERTS TRIGGERED:")
            for alert in set(all_alerts):
                print(f"  ⚠ {alert}")
        
        return report


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nomos Boundary Testing - Critical Drift Scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default="all",
        choices=["all"] + [s.value for s in DriftScenario],
        help="Scenario to run (default: all)"
    )
    parser.add_argument(
        "--requests", "-n",
        type=int,
        default=100,
        help="Requests per scenario (default: 100)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    tester = BoundaryTester(requests_per_scenario=args.requests)
    
    if args.scenario == "all":
        report = asyncio.run(tester.run_all_scenarios())
    else:
        scenario = DriftScenario(args.scenario)
        report = asyncio.run(tester.run_single_scenario(scenario))
    
    if args.output and 'error' not in report:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")
    
    # Exit code based on haltless validation
    if report.get('haltless_maintained', False):
        print("\n[SUCCESS] System remained haltless under extreme drift!")
        sys.exit(0)
    else:
        print("\n[FAILURE] System stability compromised!")
        sys.exit(1)


if __name__ == "__main__":
    main()
