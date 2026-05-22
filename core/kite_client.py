"""Thin, retry-wrapped client around the Zerodha Kite Connect REST API.

`kiteconnect` is imported lazily so the rest of the terminal (demo mode,
the analysis engines, the test-suite) runs without the SDK installed.
Transient/network failures are retried with exponential backoff; an
expired daily token (TokenException) fails fast - it cannot be retried.
"""
from __future__ import annotations

import time
from datetime import datetime

from monitoring.logging import get_logger

log = get_logger("core.kite")

# per-request maximum lookback (calendar days) by candle interval
INTERVAL_CAP_DAYS: dict[str, int] = {
    "minute": 60, "3minute": 100, "5minute": 100, "10minute": 100,
    "15minute": 200, "30minute": 200, "60minute": 400, "day": 2000,
}


class KiteClient:
    """Wrapper over kiteconnect.KiteConnect with retry + clean accessors."""

    def __init__(self, api_key: str, api_secret: str, access_token: str = ""):
        from kiteconnect import KiteConnect  # lazy: keeps demo mode SDK-free

        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self._kite = KiteConnect(api_key=api_key)
        if access_token:
            self._kite.set_access_token(access_token)

    @classmethod
    def from_settings(cls, settings) -> "KiteClient":
        if not settings.has_kite_creds:
            raise RuntimeError(
                "Kite API key/secret missing - set KITE_API_KEY / KITE_API_SECRET"
            )
        return cls(settings.kite_api_key, settings.kite_api_secret,
                   settings.kite_access_token)

    # --- auth ---------------------------------------------------------------
    def login_url(self) -> str:
        return self._kite.login_url()

    def generate_session(self, request_token: str) -> str:
        """Exchange a request_token for the day's access_token."""
        data = self._kite.generate_session(request_token, api_secret=self.api_secret)
        self.set_access_token(data["access_token"])
        return self.access_token

    def set_access_token(self, token: str) -> None:
        self.access_token = token
        self._kite.set_access_token(token)

    # --- retry --------------------------------------------------------------
    def _retry(self, fn, *args, **kwargs):
        from kiteconnect.exceptions import TokenException

        delay, last = 2.0, None
        for attempt in range(4):
            try:
                return fn(*args, **kwargs)
            except TokenException:
                log.error("Kite token expired - run the daily login routine")
                raise
            except Exception as exc:  # transient / network
                last = exc
                log.warning("Kite call failed: %s (attempt %d/4)", exc, attempt + 1)
                if attempt < 3:
                    time.sleep(delay)
                    delay *= 2
        raise last  # type: ignore[misc]

    # --- market data --------------------------------------------------------
    def historical(
        self, instrument_token: int, frm: datetime, to: datetime,
        interval: str, continuous: bool = False, oi: bool = False,
    ) -> list[dict]:
        return self._retry(self._kite.historical_data, instrument_token,
                           frm, to, interval, continuous, oi)

    def ltp(self, instruments: list[str]) -> dict:
        return self._retry(self._kite.ltp, instruments)

    def quote(self, instruments: list[str]) -> dict:
        return self._retry(self._kite.quote, instruments)

    def ohlc(self, instruments: list[str]) -> dict:
        return self._retry(self._kite.ohlc, instruments)

    def instruments(self, exchange: str | None = None) -> list[dict]:
        return self._retry(self._kite.instruments, exchange)

    @property
    def raw(self):
        """Escape hatch to the underlying KiteConnect instance."""
        return self._kite
