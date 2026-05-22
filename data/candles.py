"""Rolling in-memory OHLC candle builder.

Aggregates a stream of ticks (price + traded volume) into fixed-interval
OHLC candles. The KiteTicker websocket consumer feeds ticks in; the
analysis engines read completed candles out. Keeps only a bounded
rolling window so memory stays flat over a long trading session.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

import pandas as pd

# supported intervals -> bar length
INTERVALS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1d": timedelta(days=1),
}


def _floor_time(ts: datetime, interval: timedelta) -> datetime:
    """Snap a timestamp down to the start of its bar."""
    if interval >= timedelta(days=1):
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    secs = int(interval.total_seconds())
    epoch = int(ts.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % secs))


class CandleBuilder:
    """Aggregate ticks into OHLC candles for one symbol / one interval."""

    def __init__(self, symbol: str, interval: str = "1m", max_bars: int = 500):
        if interval not in INTERVALS:
            raise ValueError(f"unknown interval {interval!r}; pick {list(INTERVALS)}")
        self.symbol = symbol
        self.interval = interval
        self._delta = INTERVALS[interval]
        self._completed: deque[dict] = deque(maxlen=max_bars)
        self._current: dict | None = None

    def add_tick(
        self, price: float, volume: float = 0.0, ts: datetime | None = None
    ) -> None:
        """Fold one tick into the current candle, rolling over on bar change."""
        ts = ts or datetime.now()
        bar_start = _floor_time(ts, self._delta)

        if self._current is None:
            self._current = self._new_bar(bar_start, price, volume)
            return

        if bar_start > self._current["ts"]:
            self._completed.append(self._current)
            self._current = self._new_bar(bar_start, price, volume)
            return

        bar = self._current
        bar["high"] = max(bar["high"], price)
        bar["low"] = min(bar["low"], price)
        bar["close"] = price
        bar["volume"] += volume

    @staticmethod
    def _new_bar(ts: datetime, price: float, volume: float) -> dict:
        return {"ts": ts, "open": price, "high": price,
                "low": price, "close": price, "volume": volume}

    def to_dataframe(self, include_current: bool = True) -> pd.DataFrame:
        """Return the rolling window as an OHLCV DataFrame."""
        rows = list(self._completed)
        if include_current and self._current is not None:
            rows.append(self._current)
        if not rows:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )
        df = pd.DataFrame(rows).set_index("ts")
        df.index.name = None
        return df

    @property
    def bar_count(self) -> int:
        return len(self._completed) + (1 if self._current else 0)
