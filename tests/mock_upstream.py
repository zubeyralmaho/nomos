#!/usr/bin/env python3
"""
Mock Upstream Server for Nomos Testing
========================================

A simple HTTP server that simulates an upstream API.
It accepts any JSON payload and echoes it back, optionally with schema drift.

Usage:
    python mock_upstream.py [--port 9090] [--drift-mode none|rename|chaos]
    
The server runs on port 9090 by default (Nomos proxy's default target).
"""

import argparse
import asyncio
import json
import random
import string
import sys
from typing import Dict, Optional

try:
    from aiohttp import web
except ImportError:
    print("Installing aiohttp...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    from aiohttp import web


# ============================================================================
# Drift Transformations
# ============================================================================

def apply_no_drift(payload: Dict) -> Dict:
    """Return payload unchanged."""
    return payload


def apply_rename_drift(payload: Dict) -> Dict:
    """Apply semantic field renames."""
    rename_map = {
        "user_id": "uuid",
        "full_name": "name",
        "email_address": "email",
        "account_balance": "balance",
        "is_verified": "verified",
        "created_at": "created",
    }
    
    result = {}
    for key, value in payload.items():
        new_key = rename_map.get(key, key)
        result[new_key] = value
    
    return result


def apply_camelcase_drift(payload: Dict) -> Dict:
    """Convert snake_case to camelCase."""
    def to_camel(s: str) -> str:
        parts = s.split('_')
        return parts[0] + ''.join(p.capitalize() for p in parts[1:])
    
    result = {}
    for key, value in payload.items():
        result[to_camel(key)] = value
    
    return result


def apply_chaos_drift(payload: Dict) -> Dict:
    """Apply random chaotic transformations."""
    result = {}
    
    for key, value in payload.items():
        # Random key transformation
        transform = random.choice(['keep', 'camel', 'abbreviate', 'prefix'])
        
        if transform == 'keep':
            new_key = key
        elif transform == 'camel':
            parts = key.split('_')
            new_key = parts[0] + ''.join(p.capitalize() for p in parts[1:])
        elif transform == 'abbreviate':
            new_key = ''.join(p[0] for p in key.split('_'))
        else:
            new_key = f"x_{key}"
        
        # Random value transformation
        if isinstance(value, str) and random.random() < 0.3:
            value = value.upper()
        elif isinstance(value, (int, float)) and random.random() < 0.3:
            value = str(value)
        
        result[new_key] = value
    
    return result


DRIFT_MODES = {
    "none": apply_no_drift,
    "rename": apply_rename_drift,
    "camel": apply_camelcase_drift,
    "chaos": apply_chaos_drift,
}


# ============================================================================
# HTTP Server
# ============================================================================

class MockUpstreamServer:
    """Mock upstream API server."""
    
    def __init__(self, port: int = 9090, drift_mode: str = "none"):
        self.port = port
        self.drift_mode = drift_mode
        self.drift_func = DRIFT_MODES.get(drift_mode, apply_no_drift)
        self.request_count = 0
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        """Configure routes."""
        # Catch-all routes for any path
        self.app.router.add_route('*', '/{path:.*}', self.handle_request)
    
    async def handle_request(self, request: web.Request) -> web.Response:
        """Handle incoming request."""
        self.request_count += 1
        
        # Try to parse JSON body
        try:
            if request.body_exists:
                body = await request.json()
            else:
                body = {}
        except json.JSONDecodeError:
            body = {}
        
        # Apply drift transformation
        response_body = self.drift_func(body)
        
        # Add some metadata
        response_body["_meta"] = {
            "server": "mock-upstream",
            "request_id": self.request_count,
            "drift_mode": self.drift_mode,
        }
        
        return web.json_response(response_body, headers={
            "X-Upstream-Server": "mock",
            "X-Request-Id": str(self.request_count),
        })
    
    async def run(self):
        """Run the server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, '127.0.0.1', self.port)
        await site.start()
        
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    MOCK UPSTREAM SERVER                               ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Listening on:       http://127.0.0.1:{self.port:<39} ║
║  Drift Mode:         {self.drift_mode:<52} ║
╠═══════════════════════════════════════════════════════════════════════╣
║  This server echoes back JSON payloads with optional schema drift.    ║
║  Press Ctrl+C to stop.                                                ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
        
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await runner.cleanup()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mock Upstream Server for Nomos Testing",
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=9090,
        help="Port to listen on (default: 9090)"
    )
    parser.add_argument(
        "--drift-mode", "-d",
        choices=list(DRIFT_MODES.keys()),
        default="rename",
        help="Drift transformation mode (default: rename)"
    )
    
    args = parser.parse_args()
    
    server = MockUpstreamServer(
        port=args.port,
        drift_mode=args.drift_mode,
    )
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
