"""Kite Connect daily-login routine.

Kite access tokens expire every morning (~07:30 IST), so a fresh token
must be generated each trading day. This module persists today's token
and drives the one-time interactive login.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from monitoring.logging import get_logger

log = get_logger("core.auth")

TOKEN_FILE = Path("data_cache/kite_token.json")


def save_token(access_token: str) -> None:
    """Persist the access token stamped with today's date."""
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(
        json.dumps(
            {
                "access_token": access_token,
                "date": datetime.date.today().isoformat(),
            }
        )
    )
    log.info("Kite access token saved (valid until ~07:30 IST tomorrow)")


def load_valid_token() -> str | None:
    """Return today's cached access token, or None if missing/stale."""
    if not TOKEN_FILE.exists():
        return None
    try:
        data = json.loads(TOKEN_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("date") == datetime.date.today().isoformat():
        return data.get("access_token")
    log.info("Cached Kite token is stale - a new daily login is required")
    return None


def totp_now(secret: str) -> str:
    """Current TOTP code (optional helper for semi-automating the login)."""
    import pyotp

    return pyotp.TOTP(secret).now()


def interactive_login(client) -> str:
    """Walk the operator through the manual Kite login and save the token."""
    print("\n=== DALAL TERMINAL - Kite daily login ===")
    print("1. Open this URL in a browser and log in to Zerodha:\n")
    print("   ", client.login_url(), "\n")
    print("2. After login you are redirected to your registered redirect URL.")
    print("   Copy the value of the 'request_token' query parameter.\n")
    request_token = input("   request_token> ").strip()
    if not request_token:
        raise SystemExit("no request_token supplied - aborting login")
    access_token = client.generate_session(request_token)
    save_token(access_token)
    print("\nLogin complete. Start the terminal with DALAL_MODE=live.\n")
    return access_token
