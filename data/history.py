"""Historical OHLCV service.

Live mode picks a provider based on settings.history_provider:
- `kite` (or `auto` with Kite credentials) uses the Kite historical API,
  paginating across the per-interval date-range caps and persisting to
  the SQLite cache.
- `yfinance` (or `auto` without Kite credentials) falls back to free
  Yahoo Finance data (NSE listings as `<SYMBOL>.NS`). Lets users
  run the live pipeline + fit a real calibrator without a paid plan.

Demo mode always serves synthetic candles from DemoMarket.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import pandas as pd

from core.kite_client import INTERVAL_CAP_DAYS
from data.demo import get_demo_market
from data.models import validate_ohlcv
from data.store import CandleStore
from monitoring.logging import get_logger

log = get_logger("data.history")

# rough bar counts per trading day, used to size live fetch windows
_BARS_PER_DAY = {
    "minute": 375, "5minute": 75, "15minute": 25,
    "30minute": 13, "60minute": 7, "day": 1,
}

# yfinance interval / period maps
_YF_INTERVAL = {
    "minute": "1m", "5minute": "5m", "15minute": "15m",
    "30minute": "30m", "60minute": "60m", "day": "1d",
}
# Yahoo capped per interval; we ask for plenty then tail()
_YF_PERIOD = {
    "minute": "7d", "5minute": "60d", "15minute": "60d",
    "30minute": "60d", "60minute": "2y", "day": "5y",
}


class HistoryService:
    def __init__(self, settings, kite_client=None, instruments=None, store=None):
        self.settings = settings
        self._kite = kite_client
        self._instruments = instruments
        self._store = store or CandleStore(settings.db_path)
        self._demo = None if settings.is_live else get_demo_market()

    def candles(
        self, symbol: str, interval: str = "day", bars: int = 260
    ) -> pd.DataFrame:
        """Return up to `bars` recent OHLCV candles for `symbol`."""
        if not self.settings.is_live:
            demo_interval = "day" if interval == "day" else "minute"
            return self._demo.history(symbol, bars=bars, interval=demo_interval)
        provider = self._resolve_provider()
        if provider == "kite":
            return self._fetch_live(symbol, interval, bars)
        if provider == "yfinance":
            return self._fetch_yfinance(symbol, interval, bars)
        raise RuntimeError(f"unknown history provider {provider!r}")

    def _resolve_provider(self) -> str:
        """`auto` -> kite if creds present, else yfinance."""
        choice = (self.settings.history_provider or "auto").lower()
        if choice == "auto":
            return "kite" if self.settings.has_kite_creds else "yfinance"
        return choice

    def _fetch_live(self, symbol: str, interval: str, bars: int) -> pd.DataFrame:
        if self._kite is None or self._instruments is None:
            raise RuntimeError("live history needs a Kite client + instrument master")

        token = self._instruments.token_for(symbol)
        if token is None:
            raise ValueError(f"could not resolve symbol {symbol!r}")

        cap = INTERVAL_CAP_DAYS.get(interval, 2000)
        per_day = _BARS_PER_DAY.get(interval, 1)
        days_needed = math.ceil(bars / per_day) + 10

        end = datetime.now()
        cursor = end - timedelta(days=days_needed)
        frames: list[pd.DataFrame] = []
        while cursor < end:
            window_end = min(cursor + timedelta(days=cap), end)
            chunk = self._kite.historical(token, cursor, window_end, interval)
            if chunk:
                frames.append(pd.DataFrame(chunk))
            cursor = window_end + timedelta(days=1)

        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.concat(frames, ignore_index=True)
        df["ts"] = pd.to_datetime(df["date"])
        df = validate_ohlcv(df.set_index("ts"))
        self._store.save(symbol, interval, df)
        log.info("fetched %d %s candles for %s", len(df), interval, symbol)
        return df.tail(bars)

    def _fetch_yfinance(
        self, symbol: str, interval: str, bars: int
    ) -> pd.DataFrame:
        """Free Yahoo Finance fallback. NSE listings use `<SYMBOL>.NS`."""
        try:
            import yfinance as yf
        except ImportError:
            raise RuntimeError(
                "yfinance not installed - `pip install yfinance` "
                "or set HISTORY_PROVIDER=kite"
            )

        yf_int = _YF_INTERVAL.get(interval, "1d")
        period = _YF_PERIOD.get(interval, "5y")
        ticker = yf.Ticker(symbol + ".NS")
        df = ticker.history(period=period, interval=yf_int, auto_adjust=False)
        if df is None or df.empty:
            log.warning("yfinance returned no data for %s", symbol)
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"])

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]]
        df = validate_ohlcv(df)
        self._store.save(symbol, interval, df)
        log.info("fetched %d %s candles for %s via yfinance",
                 len(df), interval, symbol)
        return df.tail(bars)
