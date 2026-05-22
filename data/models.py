"""Shared market-data types used across data feeds and analysis engines."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

# Canonical OHLCV column names. Every feed normalises to these so the
# analysis engines never have to guess at column casing.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Instrument:
    """A resolved tradable instrument from the Kite instrument master."""

    instrument_token: int
    tradingsymbol: str
    name: str
    exchange: str = "NSE"
    segment: str = "NSE"
    lot_size: int = 1
    tick_size: float = 0.05


@dataclass
class Quote:
    """A point-in-time market quote (live tape / snapshot)."""

    symbol: str
    ltp: float
    prev_close: float = 0.0
    day_open: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    volume: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def change(self) -> float:
        return self.ltp - self.prev_close

    @property
    def change_pct(self) -> float:
        if not self.prev_close:
            return 0.0
        return (self.ltp - self.prev_close) / self.prev_close * 100.0

    @property
    def direction(self) -> int:
        """+1 up, -1 down, 0 flat - drives green/red colouring in the UI."""
        if self.ltp > self.prev_close:
            return 1
        if self.ltp < self.prev_close:
            return -1
        return 0


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean OHLCV frame: lower-cased columns, sorted DatetimeIndex.

    Raises ValueError if required columns are missing. This is the single
    chokepoint every analysis engine relies on, so feeds can be sloppy.
    """
    if df is None or len(df) == 0:
        raise ValueError("OHLCV frame is empty")

    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing}")

    if not isinstance(out.index, pd.DatetimeIndex):
        # tolerate a 'date'/'datetime' column instead of an index
        for cand in ("date", "datetime", "timestamp"):
            if cand in out.columns:
                out = out.set_index(pd.DatetimeIndex(out[cand]))
                break

    out = out[OHLCV_COLUMNS].astype(float)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out
