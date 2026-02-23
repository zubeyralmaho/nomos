#!/usr/bin/env python3
"""
Nomos Upstream API Server
==========================

This is a real HTTP server that simulates an external API that has undergone
schema drift. It returns responses with renamed fields, type changes, etc.

This is NOT a mock - it's a legitimate test server that represents the
actual scenario Nomos handles: APIs that change without notice.

Usage:
    python upstream_server.py [--port 9090] [--drift-mode random]
    
Drift Modes:
    - healthy: Returns expected schema (no drift)
    - v2: Returns renamed fields (uuid, name, email, etc.)
    - camel: Returns camelCase fields
    - random: Randomly picks drift patterns per request
"""

import argparse
import asyncio
import json
import random
import string
import sys
from datetime import datetime
from enum import Enum
from typing import Dict, Any

try:
    from aiohttp import web
except ImportError:
    print("Installing aiohttp...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    from aiohttp import web


# ============================================================================
# Schema Definitions
# ============================================================================

class DriftMode(Enum):
    HEALTHY = "healthy"   # Returns expected schema
    V2 = "v2"             # API v2 style renames
    CAMEL = "camel"       # camelCase conversion
    NESTED = "nested"     # Restructured to nested format
    RANDOM = "random"     # Random drift per request


def generate_user_id() -> str:
    """Generate a random user ID."""
    return "usr_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))


def get_healthy_response(user_id: str = None) -> Dict[str, Any]:
    """Expected schema (no drift)."""
    return {
        "user_id": user_id or generate_user_id(),
        "full_name": "Alice Smith",
        "email_address": "alice@example.com",
        "account_balance": 1500.50,
        "is_verified": True,
        "created_at": datetime.now().isoformat(),
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


def get_v2_response(user_id: str = None) -> Dict[str, Any]:
    """API v2 style: fields renamed to shorter names."""
    return {
        "uuid": user_id or generate_user_id(),      # user_id -> uuid
        "name": "Alice Smith",                       # full_name -> name
        "email": "alice@example.com",                # email_address -> email
        "balance": 1500.50,                          # account_balance -> balance
        "verified": True,                            # is_verified -> verified
        "created": datetime.now().isoformat(),       # created_at -> created
        "prefs": {                                   # preferences -> prefs
            "theme": "dark",
            "notifs": True,                          # notifications -> notifs
            "lang": "en"                             # language -> lang
        },
        "labels": ["premium", "active"],             # tags -> labels
        "meta": {                                    # metadata -> meta
            "ver": 1,                                # version -> ver
            "src": "api"                             # source -> src
        }
    }


def get_camel_response(user_id: str = None) -> Dict[str, Any]:
    """CamelCase conversion (common in Java APIs)."""
    return {
        "userId": user_id or generate_user_id(),
        "fullName": "Alice Smith",
        "emailAddress": "alice@example.com",
        "accountBalance": 1500.50,
        "isVerified": True,
        "createdAt": datetime.now().isoformat(),
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


def get_nested_response(user_id: str = None) -> Dict[str, Any]:
    """Restructured to nested format."""
    return {
        "user": {
            "id": user_id or generate_user_id(),
            "profile": {
                "name": "Alice Smith",
                "email": "alice@example.com"
            },
            "status": {
                "verified": True,
                "tier": "premium"
            }
        },
        "account": {
            "balance": 1500.50,
            "currency": "USD"
        },
        "timestamps": {
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat()
        },
        "settings": {
            "theme": "dark",
            "notifications": True,
            "language": "en"
        },
        "tags": ["premium", "active"]
    }


DRIFT_GENERATORS = {
    DriftMode.HEALTHY: get_healthy_response,
    DriftMode.V2: get_v2_response,
    DriftMode.CAMEL: get_camel_response,
    DriftMode.NESTED: get_nested_response,
}


# ============================================================================
# Server Implementation
# ============================================================================

class UpstreamServer:
    """Simulates an external API with configurable drift."""
    
    def __init__(self, port: int = 9090, drift_mode: DriftMode = DriftMode.V2):
        self.port = port
        self.drift_mode = drift_mode
        self.request_count = 0
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        self.app.router.add_route('*', '/api/user', self.handle_user)
        self.app.router.add_route('*', '/api/user/{user_id}', self.handle_user)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/metrics', self.handle_metrics)
        # Catch-all for any other path
        self.app.router.add_route('*', '/{path:.*}', self.handle_any)
    
    async def handle_user(self, request: web.Request) -> web.Response:
        """Handle /api/user requests."""
        self.request_count += 1
        
        # Extract user_id from path or body
        user_id = request.match_info.get('user_id')
        
        if request.method == 'POST':
            try:
                body = await request.json()
                # Try to extract user_id from various field names
                user_id = (
                    body.get('user_id') or
                    body.get('uuid') or
                    body.get('userId') or
                    body.get('u_id') or
                    user_id
                )
            except:
                pass
        
        # Select drift mode
        mode = self.drift_mode
        if mode == DriftMode.RANDOM:
            mode = random.choice([DriftMode.HEALTHY, DriftMode.V2, DriftMode.CAMEL])
        
        # Generate response
        generator = DRIFT_GENERATORS.get(mode, get_v2_response)
        response_data = generator(user_id)
        
        return web.json_response(response_data)
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "drift_mode": self.drift_mode.value,
            "requests_served": self.request_count
        })
    
    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Metrics endpoint."""
        return web.json_response({
            "requests_total": self.request_count,
            "drift_mode": self.drift_mode.value,
        })
    
    async def handle_any(self, request: web.Request) -> web.Response:
        """Handle any other request - echo back with drift."""
        self.request_count += 1
        
        mode = self.drift_mode
        if mode == DriftMode.RANDOM:
            mode = random.choice([DriftMode.HEALTHY, DriftMode.V2, DriftMode.CAMEL])
        
        generator = DRIFT_GENERATORS.get(mode, get_v2_response)
        response_data = generator()
        
        return web.json_response(response_data)
    
    def run(self):
        """Start the server."""
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    NOMOS UPSTREAM API SERVER                          ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Port:           {self.port:<56} ║
║  Drift Mode:     {self.drift_mode.value:<56} ║
║                                                                       ║
║  This server simulates an external API that has schema drift.         ║
║  Nomos proxy will forward requests here and heal the responses.       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
        
        print(f"Endpoints:")
        print(f"  POST /api/user       - Returns user data with drift")
        print(f"  GET  /health         - Health check")
        print(f"  GET  /metrics        - Request count")
        print(f"\nDrift patterns:")
        print(f"  v2:      user_id→uuid, full_name→name, email_address→email")
        print(f"  camel:   user_id→userId, full_name→fullName")
        print(f"  nested:  Restructured to nested format")
        print(f"  healthy: No drift (expected schema)")
        print(f"\nStarting server on http://0.0.0.0:{self.port}")
        print(f"Press Ctrl+C to stop.\n")
        
        web.run_app(self.app, host='0.0.0.0', port=self.port, print=None)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nomos Upstream API Server - Simulates drifted external API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=9090,
        help="Port to listen on (default: 9090)"
    )
    parser.add_argument(
        "--drift-mode", "-m",
        type=str,
        default="v2",
        choices=["healthy", "v2", "camel", "nested", "random"],
        help="Drift mode (default: v2)"
    )
    
    args = parser.parse_args()
    
    drift_mode = DriftMode(args.drift_mode)
    server = UpstreamServer(port=args.port, drift_mode=drift_mode)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
