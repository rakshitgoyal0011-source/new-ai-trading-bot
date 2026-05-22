"""Tests for the SQLite OHLCV cache."""
import numpy as np
import pandas as pd

from data.store import CandleStore


def _df(n=10):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    base = np.arange(n, dtype=float) + 100.0
    return pd.DataFrame(
        {"open": base, "high": base + 1.0, "low": base - 1.0,
         "close": base + 0.5, "volume": base * 100.0},
        index=idx,
    )


def test_save_and_load_roundtrip(tmp_path):
    store = CandleStore(str(tmp_path / "t.sqlite"))
    written = store.save("TEST", "day", _df(10))
    assert written == 10

    loaded = store.load("TEST", "day")
    assert len(loaded) == 10
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert abs(loaded["close"].iloc[-1] - 109.5) < 1e-9


def test_save_is_idempotent_upsert(tmp_path):
    store = CandleStore(str(tmp_path / "t.sqlite"))
    store.save("TEST", "day", _df(10))
    store.save("TEST", "day", _df(10))
    assert len(store.load("TEST", "day")) == 10  # no duplicate rows


def test_latest_timestamp_tracks_newest(tmp_path):
    store = CandleStore(str(tmp_path / "t.sqlite"))
    store.save("TEST", "day", _df(10))
    assert store.latest_timestamp("TEST", "day").startswith("2024-01-10")
    assert "TEST" in store.symbols()
