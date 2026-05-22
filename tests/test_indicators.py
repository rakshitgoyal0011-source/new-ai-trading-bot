"""Tests for technical indicators - deterministic, known-value checks."""
import numpy as np
import pandas as pd

from analysis.technical import indicators as ind


def _df(closes):
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    high = np.maximum(opens, closes) + 0.5
    low = np.minimum(opens, closes) - 0.5
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {"open": opens, "high": high, "low": low, "close": closes,
         "volume": np.full(len(closes), 1000.0)},
        index=idx,
    )


def _walk(n, seed):
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))


def test_sma_known_value():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert ind.sma(s, 3).iloc[-1] == 4.0


def test_ema_of_constant_is_constant():
    s = pd.Series(np.full(50, 7.0))
    assert abs(ind.ema(s, 10).iloc[-1] - 7.0) < 1e-9


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 60), dtype=float)
    assert ind.rsi(s).iloc[-1] > 99.0


def test_rsi_all_losses_is_zero():
    s = pd.Series(np.arange(60, 1, -1), dtype=float)
    assert ind.rsi(s).iloc[-1] < 1.0


def test_rsi_stays_in_bounds():
    r = ind.rsi(pd.Series(_walk(300, 1))).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_macd_histogram_identity():
    m = ind.macd(pd.Series(_walk(200, 2)))
    assert set(m.columns) == {"macd", "signal", "hist"}
    diff = (m["macd"] - m["signal"]).dropna()
    assert np.allclose(diff, m["hist"].dropna())


def test_atr_is_positive():
    a = ind.atr(_df(_walk(200, 3))).dropna()
    assert (a > 0).all()


def test_bollinger_band_ordering():
    bb = ind.bollinger_bands(pd.Series(_walk(200, 4))).dropna()
    assert (bb["upper"] >= bb["middle"]).all()
    assert (bb["middle"] >= bb["lower"]).all()


def test_adx_in_bounds():
    a = ind.adx(_df(_walk(300, 5)))["adx"].dropna()
    assert (a >= 0).all() and (a <= 100).all()


def test_supertrend_direction_is_signed():
    st = ind.supertrend(_df(_walk(200, 6)))
    assert set(st["direction"].unique()).issubset({-1.0, 1.0})


def test_williams_r_in_bounds():
    wr = ind.williams_r(_df(_walk(200, 7))).dropna()
    assert (wr >= -100.01).all() and (wr <= 0.01).all()


def test_mfi_in_bounds():
    m = ind.money_flow_index(_df(_walk(200, 8))).dropna()
    assert (m >= 0).all() and (m <= 100).all()


def test_vwap_positive():
    assert ind.vwap(_df(_walk(120, 9))).iloc[-1] > 0


def test_obv_responds_to_direction():
    rising = _df([10, 11, 12, 13, 14, 15])
    assert ind.obv(rising).iloc[-1] > 0
