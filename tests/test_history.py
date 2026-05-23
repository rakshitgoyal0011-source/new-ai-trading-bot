"""Tests for HistoryService provider resolution and the yfinance fallback."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pandas as pd

from config.settings import Settings
from data.history import HistoryService


def test_resolve_provider_auto_without_kite_uses_yfinance(tmp_path):
    s = Settings(DALAL_MODE="live", HISTORY_PROVIDER="auto",
                 DB_PATH=str(tmp_path / "x.db"))
    hs = HistoryService(s)
    assert hs._resolve_provider() == "yfinance"


def test_resolve_provider_auto_with_kite_uses_kite(tmp_path):
    s = Settings(DALAL_MODE="live", HISTORY_PROVIDER="auto",
                 KITE_API_KEY="k", KITE_API_SECRET="s",
                 DB_PATH=str(tmp_path / "x.db"))
    hs = HistoryService(s)
    assert hs._resolve_provider() == "kite"


def test_resolve_provider_explicit_overrides_auto(tmp_path):
    s = Settings(DALAL_MODE="live", HISTORY_PROVIDER="yfinance",
                 KITE_API_KEY="k", KITE_API_SECRET="s",
                 DB_PATH=str(tmp_path / "x.db"))
    hs = HistoryService(s)
    assert hs._resolve_provider() == "yfinance"


def test_yfinance_fetch_with_mocked_module(monkeypatch, tmp_path):
    """Stand in for yfinance so the fallback path is exercised end-to-end
    without hitting the network."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    mock_df = pd.DataFrame({
        "Open":   [100.0] * 10,
        "High":   [105.0] * 10,
        "Low":    [ 95.0] * 10,
        "Close":  [102.0] * 10,
        "Volume": [1000]  * 10,
    }, index=idx)
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_df
    mock_yf = MagicMock()
    mock_yf.Ticker.return_value = mock_ticker
    monkeypatch.setitem(sys.modules, "yfinance", mock_yf)

    s = Settings(DALAL_MODE="live", HISTORY_PROVIDER="yfinance",
                 DB_PATH=str(tmp_path / "x.db"))
    hs = HistoryService(s)
    out = hs.candles("RELIANCE", "day", 100)

    assert len(out) == 10
    assert set(out.columns) == {"open", "high", "low", "close", "volume"}
    mock_yf.Ticker.assert_called_with("RELIANCE.NS")
    mock_ticker.history.assert_called_once()


def test_yfinance_returns_empty_when_no_data(monkeypatch, tmp_path):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_yf = MagicMock()
    mock_yf.Ticker.return_value = mock_ticker
    monkeypatch.setitem(sys.modules, "yfinance", mock_yf)

    s = Settings(DALAL_MODE="live", HISTORY_PROVIDER="yfinance",
                 DB_PATH=str(tmp_path / "x.db"))
    hs = HistoryService(s)
    out = hs.candles("UNKNOWN", "day", 100)
    assert len(out) == 0
