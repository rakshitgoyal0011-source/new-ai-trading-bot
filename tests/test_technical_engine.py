"""Tests for the technical analysis engine."""
import json

import numpy as np
import pandas as pd

from analysis.technical import TechnicalEngine


def _trend_df(n, drift, seed):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.012, n)
    close = 100.0 * np.cumprod(1.0 + rets)
    opens = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(opens, close) * 1.01
    low = np.minimum(opens, close) * 0.99
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": opens, "high": high, "low": low, "close": close,
         "volume": rng.integers(1e5, 1e6, n).astype(float)},
        index=idx,
    )


def test_result_shape_and_bounds():
    res = TechnicalEngine().analyze("TEST", _trend_df(260, 0.0, 1))
    assert 0.0 <= res.score <= 100.0
    assert res.bias in ("bullish", "bearish", "neutral")
    assert res.trend in ("uptrend", "downtrend", "sideways")
    assert 0.0 <= res.confidence <= 1.0
    assert res.signals, "engine should emit signals"
    assert res.sub_scores, "engine should emit sub-scores"


def test_uptrend_scores_higher_than_downtrend():
    up = TechnicalEngine().analyze("UP", _trend_df(260, 0.004, 2))
    down = TechnicalEngine().analyze("DOWN", _trend_df(260, -0.004, 3))
    assert up.score > down.score
    assert up.score >= 55.0
    assert down.score <= 45.0


def test_uptrend_classified_as_uptrend():
    res = TechnicalEngine().analyze("UP", _trend_df(260, 0.004, 11))
    assert res.trend == "uptrend"
    assert res.bias == "bullish"


def test_insufficient_data_is_neutral():
    res = TechnicalEngine().analyze("X", _trend_df(12, 0.004, 4))
    assert res.score == 50.0
    assert res.confidence == 0.0


def test_every_signal_has_a_reason():
    res = TechnicalEngine().analyze("T", _trend_df(260, 0.002, 5))
    for sig in res.signals:
        assert sig.reason
        assert sig.direction in (-1, 0, 1)
        assert 0.0 <= sig.strength <= 1.0


def test_result_is_json_serialisable():
    res = TechnicalEngine().analyze("T", _trend_df(260, 0.002, 6))
    json.dumps(res.to_dict())  # must not raise


def test_multi_timeframe_confirmation():
    df = _trend_df(260, 0.004, 7)
    res = TechnicalEngine().analyze("T", df, higher_tf=df.iloc[::5])
    assert res.mtf_confirmed is True
