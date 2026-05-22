"""NSE/BSE instrument master: cache the Kite instrument dump and resolve
trading symbols to instrument tokens.

The dump (~50k+ rows) changes daily, so it is cached per-day on disk and
only re-downloaded when stale.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

from data.models import Instrument
from monitoring.logging import get_logger

log = get_logger("core.instruments")


class InstrumentMaster:
    """Resolve symbols <-> instrument tokens from the cached Kite dump."""

    def __init__(self, kite_client=None, cache_dir: str = "data_cache"):
        self._client = kite_client
        self._cache_dir = Path(cache_dir)
        self._df: pd.DataFrame | None = None

    def _cache_path(self, exchange: str) -> Path:
        today = datetime.date.today().isoformat()
        return self._cache_dir / f"instruments_{exchange}_{today}.parquet"

    def load(self, exchange: str = "NSE", force: bool = False) -> pd.DataFrame:
        """Load the instrument dump for an exchange (cached per day)."""
        path = self._cache_path(exchange)
        if path.exists() and not force:
            self._df = pd.read_parquet(path)
            log.info("instrument master loaded from cache (%d rows)", len(self._df))
            return self._df

        if self._client is None:
            raise RuntimeError(
                "instrument master not cached and no Kite client available"
            )
        log.info("downloading instrument dump for %s ...", exchange)
        df = pd.DataFrame(self._client.instruments(exchange))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        self._df = df
        return df

    def resolve(self, symbol: str, exchange: str = "NSE") -> Instrument | None:
        """Resolve an NSE/BSE trading symbol to an Instrument."""
        if self._df is None:
            self.load(exchange)
        assert self._df is not None
        match = self._df[self._df["tradingsymbol"] == symbol.strip().upper()]
        if match.empty:
            log.warning("symbol %r not found in instrument master", symbol)
            return None
        row = match.iloc[0]
        return Instrument(
            instrument_token=int(row["instrument_token"]),
            tradingsymbol=str(row["tradingsymbol"]),
            name=str(row.get("name", "")),
            exchange=str(row.get("exchange", exchange)),
            segment=str(row.get("segment", exchange)),
            lot_size=int(row.get("lot_size", 1) or 1),
            tick_size=float(row.get("tick_size", 0.05) or 0.05),
        )

    def token_for(self, symbol: str, exchange: str = "NSE") -> int | None:
        inst = self.resolve(symbol, exchange)
        return inst.instrument_token if inst else None
