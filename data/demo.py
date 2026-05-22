"""Synthetic market data for DALAL TERMINAL's demo mode.

Lets the whole terminal - quotes, ticker tape, technical engine, TOP
leaderboard - run end-to-end with NO API keys. Data is deterministic per
symbol (seeded), so tests and screenshots are reproducible. Demo prices
are fictional and must never be treated as real market data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.models import Quote


def _symbol_seed(symbol: str, base_seed: int) -> int:
    return (base_seed ^ (abs(hash(symbol)) % (2**31))) & 0xFFFFFFFF


class DemoMarket:
    """Deterministic synthetic market - geometric random walk per symbol."""

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._live: dict[str, dict] = {}

    def history(
        self, symbol: str, bars: int = 260, interval: str = "day"
    ) -> pd.DataFrame:
        """Generate a reproducible OHLCV history for `symbol`."""
        rng = np.random.default_rng(_symbol_seed(symbol, self._seed))
        base = 100.0 + (abs(hash(symbol)) % 3900)
        mu = rng.normal(0.0005, 0.0004)          # per-bar drift
        sigma = rng.uniform(0.012, 0.026)        # per-bar volatility

        rets = rng.normal(mu, sigma, bars)
        close = base * np.cumprod(1.0 + rets)

        opens = np.empty(bars)
        highs = np.empty(bars)
        lows = np.empty(bars)
        opens[0] = base
        for i in range(bars):
            if i > 0:
                opens[i] = close[i - 1] * (1.0 + rng.normal(0.0, sigma * 0.25))
            hi_wick = abs(rng.normal(0.0, sigma * 0.6))
            lo_wick = abs(rng.normal(0.0, sigma * 0.6))
            top = max(opens[i], close[i])
            bot = min(opens[i], close[i])
            highs[i] = top * (1.0 + hi_wick)
            lows[i] = bot * (1.0 - lo_wick)

        rng_pct = (highs - lows) / close
        volume = (rng.uniform(2e5, 1e6, bars) * (1.0 + 8.0 * rng_pct)).astype(int)

        freq = "B" if interval == "day" else "min"
        index = pd.date_range(
            end=pd.Timestamp.now().normalize(), periods=bars, freq=freq
        )
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows,
             "close": close, "volume": volume},
            index=index,
        )

    def _ensure_live(self, symbol: str) -> dict:
        if symbol not in self._live:
            hist = self.history(symbol, bars=260)
            prev_close = float(hist["close"].iloc[-1])
            self._live[symbol] = {
                "prev_close": prev_close,
                "ltp": prev_close,
                "day_open": prev_close,
                "day_high": prev_close,
                "day_low": prev_close,
                "volume": 0,
                "rng": np.random.default_rng(_symbol_seed(symbol, self._seed + 7)),
            }
        return self._live[symbol]

    def step(self, symbols: list[str] | None = None) -> None:
        """Advance live prices one tick (random walk around prev close)."""
        targets = symbols or list(self._live.keys())
        for symbol in targets:
            s = self._ensure_live(symbol)
            shock = s["rng"].normal(0.0, 0.0018)
            s["ltp"] = round(max(0.05, s["ltp"] * (1.0 + shock)), 2)
            s["day_high"] = max(s["day_high"], s["ltp"])
            s["day_low"] = min(s["day_low"], s["ltp"])
            s["volume"] += int(s["rng"].integers(500, 5000))

    def quote(self, symbol: str) -> Quote:
        s = self._ensure_live(symbol)
        return Quote(
            symbol=symbol,
            ltp=s["ltp"],
            prev_close=round(s["prev_close"], 2),
            day_open=round(s["day_open"], 2),
            day_high=round(s["day_high"], 2),
            day_low=round(s["day_low"], 2),
            volume=s["volume"],
        )


_DEMO: DemoMarket | None = None


def get_demo_market() -> DemoMarket:
    """Process-wide demo market singleton."""
    global _DEMO
    if _DEMO is None:
        _DEMO = DemoMarket()
    return _DEMO
