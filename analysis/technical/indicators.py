"""Technical indicators and oscillators.

Pure functions over pandas Series / OHLCV DataFrames - no I/O, no state -
so every one of them is trivially unit-testable. Wilder's smoothing
(RMA, alpha = 1/period) is used for RSI / ATR / ADX to match the values
charting platforms report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# moving averages
# --------------------------------------------------------------------------
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's moving average (a.k.a. RMA / SMMA)."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# --------------------------------------------------------------------------
# momentum oscillators
# --------------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100), Wilder smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # when there are no losses RSI is defined as 100
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R (-100 .. 0)."""
    hh = df["high"].rolling(period, min_periods=period).max()
    ll = df["low"].rolling(period, min_periods=period).min()
    rng = (hh - ll).replace(0.0, np.nan)
    return -100.0 * (hh - df["close"]) / rng


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Stochastic oscillator %K / %D."""
    hh = df["high"].rolling(k_period, min_periods=k_period).max()
    ll = df["low"].rolling(k_period, min_periods=k_period).min()
    rng = (hh - ll).replace(0.0, np.nan)
    k = 100.0 * (df["close"] - ll) / rng
    d = k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "hist": macd_line - signal_line,
        }
    )


def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index (0-100) - volume-weighted RSI."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_flow = tp * df["volume"]
    delta = tp.diff()
    pos_flow = raw_flow.where(delta > 0, 0.0)
    neg_flow = raw_flow.where(delta < 0, 0.0)
    pos_sum = pos_flow.rolling(period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(period, min_periods=period).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + ratio)
    return out.where(neg_sum != 0.0, 100.0)


# --------------------------------------------------------------------------
# volatility
# --------------------------------------------------------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, Wilder smoothing."""
    return rma(true_range(df), period)


def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands (upper / middle / lower) + %B + bandwidth."""
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower)
    pct_b = (close - lower) / width.replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "upper": upper,
            "middle": mid,
            "lower": lower,
            "bandwidth": width / mid.replace(0.0, np.nan),
            "pct_b": pct_b,
        }
    )


# --------------------------------------------------------------------------
# trend strength
# --------------------------------------------------------------------------
def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI / -DI, Wilder smoothing."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    atr_ = rma(true_range(df), period)
    plus_di = 100.0 * rma(plus_dm, period) / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * rma(minus_dm, period) / atr_.replace(0.0, np.nan)

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_ = rma(dx, period)
    return pd.DataFrame({"adx": adx_, "plus_di": plus_di, "minus_di": minus_di})


def supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """Supertrend line + direction (+1 bullish / -1 bearish)."""
    atr_ = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper = hl2 + multiplier * atr_
    lower = hl2 - multiplier * atr_

    close = df["close"].to_numpy()
    upper_a, lower_a = upper.to_numpy(), lower.to_numpy()
    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    trend = np.ones(n)  # +1 bull, -1 bear

    for i in range(1, n):
        # carry the band unless it tightens or price breaks it
        if np.isnan(upper_a[i]):
            continue
        fu_prev = final_upper[i - 1]
        fl_prev = final_lower[i - 1]
        final_upper[i] = (
            upper_a[i]
            if np.isnan(fu_prev) or upper_a[i] < fu_prev or close[i - 1] > fu_prev
            else fu_prev
        )
        final_lower[i] = (
            lower_a[i]
            if np.isnan(fl_prev) or lower_a[i] > fl_prev or close[i - 1] < fl_prev
            else fl_prev
        )
        if close[i] > final_upper[i]:
            trend[i] = 1
        elif close[i] < final_lower[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    line = np.where(trend == 1, final_lower, final_upper)
    return pd.DataFrame(
        {"supertrend": line, "direction": trend}, index=df.index
    )


# --------------------------------------------------------------------------
# volume
# --------------------------------------------------------------------------
def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"]).fillna(0.0).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP (reset per session by the caller for intraday)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol = df["volume"].cumsum().replace(0.0, np.nan)
    return (tp * df["volume"]).cumsum() / cum_vol


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume relative to its rolling average (>1 = above normal)."""
    avg = df["volume"].rolling(period, min_periods=period).mean()
    return df["volume"] / avg.replace(0.0, np.nan)
