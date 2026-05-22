"""DALAL TERMINAL - entry point.

  python main.py            start the web terminal
  python main.py --login    run the one-time daily Kite login routine

This is an analytics & research terminal. It surfaces buy-probability
estimates for human decision-making and never places orders.
"""
from __future__ import annotations

import argparse

from config.settings import get_settings
from monitoring.logging import get_logger, setup_logging


def run_login() -> None:
    """One-time-per-day Kite Connect login to mint an access token."""
    from core.auth import interactive_login
    from core.kite_client import KiteClient

    settings = get_settings()
    if not settings.has_kite_creds:
        raise SystemExit(
            "set KITE_API_KEY and KITE_API_SECRET in .env before --login"
        )
    interactive_login(KiteClient.from_settings(settings))


def run_server() -> None:
    import uvicorn

    settings = get_settings()
    log = get_logger("main")
    log.info("DALAL TERMINAL  ->  http://%s:%s  (mode=%s)",
             settings.host, settings.port, settings.mode)
    if not settings.is_live:
        log.info("running in DEMO mode - synthetic data, no API keys needed")
    uvicorn.run("ui.server:app", host=settings.host, port=settings.port,
                reload=False, log_level=settings.log_level.lower())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dalal-terminal", description="DALAL TERMINAL")
    parser.add_argument("--login", action="store_true",
                        help="run the Kite Connect daily login routine")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    if args.login:
        run_login()
    else:
        run_server()


if __name__ == "__main__":
    main()
