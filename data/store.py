"""On-disk OHLCV cache (SQLite).

Persists historical candles so the terminal does not re-hit the Kite
historical API on every restart (the API is rate-limited and metered).
Indicators and backtests read from here first, fetch-and-fill on a miss.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from data.models import validate_ohlcv

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol   TEXT NOT NULL,
    interval TEXT NOT NULL,
    ts       TEXT NOT NULL,
    open     REAL NOT NULL,
    high     REAL NOT NULL,
    low      REAL NOT NULL,
    close    REAL NOT NULL,
    volume   REAL NOT NULL,
    PRIMARY KEY (symbol, interval, ts)
);
"""


class CandleStore:
    """Thin SQLite wrapper for OHLCV persistence."""

    def __init__(self, db_path: str = "data_cache/dalal.sqlite"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def save(self, symbol: str, interval: str, df: pd.DataFrame) -> int:
        """Upsert an OHLCV frame; returns the number of rows written."""
        df = validate_ohlcv(df)
        rows = [
            (symbol, interval, ts.isoformat(),
             float(r.open), float(r.high), float(r.low),
             float(r.close), float(r.volume))
            for ts, r in df.iterrows()
        ]
        with closing(self._connect()) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO candles "
                "(symbol, interval, ts, open, high, low, close, volume) "
                "VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
        return len(rows)

    def load(
        self, symbol: str, interval: str, limit: int | None = None
    ) -> pd.DataFrame:
        """Load cached candles for a symbol/interval as an OHLCV frame."""
        query = (
            "SELECT ts, open, high, low, close, volume FROM candles "
            "WHERE symbol=? AND interval=? ORDER BY ts"
        )
        with closing(self._connect()) as conn:
            df = pd.read_sql_query(query, conn, params=(symbol, interval))
        if df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts").sort_index()
        df.index.name = None
        return df.tail(limit) if limit else df

    def latest_timestamp(self, symbol: str, interval: str) -> str | None:
        """Most recent cached bar timestamp - used to fetch only the gap."""
        with closing(self._connect()) as conn:
            cur = conn.execute(
                "SELECT MAX(ts) FROM candles WHERE symbol=? AND interval=?",
                (symbol, interval),
            )
            return cur.fetchone()[0]

    def symbols(self) -> list[str]:
        with closing(self._connect()) as conn:
            cur = conn.execute("SELECT DISTINCT symbol FROM candles ORDER BY symbol")
            return [r[0] for r in cur.fetchall()]
