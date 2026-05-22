"""Technical analysis engine.

Combines indicators, candlestick patterns, chart patterns and gap theory
into a single TECHNICAL SCORE (0-100) plus a list of human-readable
reasons. Every contributing check is a `Signal` with an explicit
direction / strength / weight, so the score is always explainable and
the engine stays unit-testable.

Score convention: 50 = neutral, >60 leans bullish, <40 leans bearish.
This is decision-support output, NOT a guarantee of future returns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

import pandas as pd

from data.models import validate_ohlcv

from . import candlesticks as cs
from . import chart_patterns as cp
from . import gaps as gp
from . import indicators as ind


# --------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------
@dataclass
class Signal:
    name: str
    category: str        # trend | momentum | volatility | volume | pattern
    direction: int       # +1 bullish, -1 bearish, 0 neutral
    strength: float      # 0..1
    weight: float        # relative importance
    reason: str

    @property
    def contribution(self) -> float:
        return self.direction * self.strength * self.weight


@dataclass
class TechnicalResult:
    symbol: str
    score: float                       # 0-100
    bias: str                          # bullish | bearish | neutral
    trend: str                         # uptrend | downtrend | sideways
    confidence: float                  # 0..1 (data depth + signal agreement)
    reasons: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    sub_scores: dict[str, float] = field(default_factory=dict)
    indicators: dict[str, float] = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)
    mtf_confirmed: bool = False
    as_of: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        return d


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------
class TechnicalEngine:
    """Stateless analyser: feed it OHLCV, get a scored, explained result."""

    MIN_BARS = 30

    # relative weights per signal family
    WEIGHTS = {
        "ema_cross": 1.2,
        "ema200": 1.0,
        "adx": 1.3,
        "supertrend": 1.2,
        "dow": 1.5,
        "rsi": 1.2,
        "macd": 1.3,
        "williams": 0.7,
        "stoch": 0.8,
        "mfi": 0.9,
        "bollinger": 0.9,
        "obv": 0.9,
        "volume": 0.8,
        "vwap": 0.6,
        "candlestick": 0.8,
        "chart": 1.0,
        "gap": 0.8,
        "mtf": 1.0,
    }

    def analyze(
        self,
        symbol: str,
        df: pd.DataFrame,
        higher_tf: pd.DataFrame | None = None,
    ) -> TechnicalResult:
        df = validate_ohlcv(df)
        if len(df) < self.MIN_BARS:
            return TechnicalResult(
                symbol=symbol, score=50.0, bias="neutral", trend="sideways",
                confidence=0.0,
                reasons=[f"only {len(df)} bars - need >= {self.MIN_BARS}"],
            )

        close = df["close"]
        signals: list[Signal] = []
        snapshot: dict[str, float] = {"close": round(float(close.iloc[-1]), 2)}

        trend, trend_reason = self._dow_trend(df)
        self._trend_signals(df, trend, trend_reason, signals, snapshot)
        self._momentum_signals(df, signals, snapshot)
        self._volatility_signals(df, signals, snapshot)
        self._volume_signals(df, signals, snapshot)
        patterns = self._pattern_signals(df, signals)

        mtf_confirmed = self._mtf_signal(higher_tf, trend, signals)

        score, sub_scores = self._aggregate(signals)
        bias = "bullish" if score >= 60 else "bearish" if score <= 40 else "neutral"

        active = [s for s in signals if s.direction != 0]
        agreement = (
            abs(sum(1 if s.direction > 0 else -1 for s in active)) / len(active)
            if active else 0.0
        )
        confidence = round(
            min(1.0, len(df) / 200.0) * (0.5 + 0.5 * agreement), 3
        )

        reasons = self._top_reasons(signals)
        return TechnicalResult(
            symbol=symbol, score=score, bias=bias, trend=trend,
            confidence=confidence, reasons=reasons, patterns=patterns,
            sub_scores=sub_scores, indicators=snapshot, signals=signals,
            mtf_confirmed=mtf_confirmed,
        )

    # ----- trend -----------------------------------------------------------
    def _dow_trend(self, df: pd.DataFrame) -> tuple[str, str]:
        highs = cp.pivot_highs(df, 3, 3)
        lows = cp.pivot_lows(df, 3, 3)
        if len(highs) >= 2 and len(lows) >= 2:
            hh = df["high"].iloc[highs[-1]] > df["high"].iloc[highs[-2]]
            hl = df["low"].iloc[lows[-1]] > df["low"].iloc[lows[-2]]
            lh = df["high"].iloc[highs[-1]] < df["high"].iloc[highs[-2]]
            ll = df["low"].iloc[lows[-1]] < df["low"].iloc[lows[-2]]
            if hh and hl:
                return "uptrend", "higher highs & higher lows (Dow)"
            if lh and ll:
                return "downtrend", "lower highs & lower lows (Dow)"
            return "sideways", "mixed swing structure"
        e20 = ind.ema(df["close"], 20)
        e50 = ind.ema(df["close"], 50)
        if pd.notna(e20.iloc[-1]) and pd.notna(e50.iloc[-1]):
            if e20.iloc[-1] > e50.iloc[-1]:
                return "uptrend", "EMA20 above EMA50"
            return "downtrend", "EMA20 below EMA50"
        return "sideways", "insufficient swing data"

    def _trend_signals(self, df, trend, trend_reason, signals, snap):
        close = df["close"]
        e20, e50 = ind.ema(close, 20), ind.ema(close, 50)
        e200 = ind.ema(close, 200)
        snap["ema20"] = self._r(e20)
        snap["ema50"] = self._r(e50)
        snap["ema200"] = self._r(e200)

        if pd.notna(e20.iloc[-1]) and pd.notna(e50.iloc[-1]):
            spread = (e20.iloc[-1] - e50.iloc[-1]) / e50.iloc[-1]
            crossed = (
                len(e20) > 2
                and (e20.iloc[-2] - e50.iloc[-2]) * spread < 0
            )
            direction = 1 if spread > 0 else -1
            strength = min(1.0, abs(spread) * 25.0)
            if crossed:
                strength = 1.0
            reason = (
                f"EMA20 {'crossed above' if crossed and direction > 0 else 'crossed below' if crossed else 'above' if direction > 0 else 'below'}"
                f" EMA50"
            )
            signals.append(Signal("ema_cross", "trend", direction, strength,
                                   self.WEIGHTS["ema_cross"], reason))

        if pd.notna(e200.iloc[-1]):
            direction = 1 if close.iloc[-1] > e200.iloc[-1] else -1
            gap = abs(close.iloc[-1] - e200.iloc[-1]) / e200.iloc[-1]
            signals.append(Signal(
                "ema200", "trend", direction, min(1.0, gap * 10.0),
                self.WEIGHTS["ema200"],
                f"price {'above' if direction > 0 else 'below'} 200-EMA"))

        adx_df = ind.adx(df)
        adx_v = adx_df["adx"].iloc[-1]
        plus_di = adx_df["plus_di"].iloc[-1]
        minus_di = adx_df["minus_di"].iloc[-1]
        snap["adx"] = self._r(adx_df["adx"])
        if pd.notna(adx_v):
            if adx_v >= 20:
                direction = 1 if plus_di > minus_di else -1
                strength = min(1.0, (adx_v - 20.0) / 30.0)
                signals.append(Signal(
                    "adx", "trend", direction, strength, self.WEIGHTS["adx"],
                    f"ADX {adx_v:.0f} - {'strong' if adx_v >= 25 else 'building'} "
                    f"{'+DI>-DI' if direction > 0 else '-DI>+DI'}"))
            else:
                signals.append(Signal("adx", "trend", 0, 0.3,
                                       self.WEIGHTS["adx"],
                                       f"ADX {adx_v:.0f} - no clear trend"))

        st = ind.supertrend(df)
        st_dir = st["direction"].iloc[-1]
        snap["supertrend_dir"] = float(st_dir)
        signals.append(Signal(
            "supertrend", "trend", int(st_dir), 0.8, self.WEIGHTS["supertrend"],
            f"Supertrend {'bullish' if st_dir > 0 else 'bearish'}"))

        dow_dir = 1 if trend == "uptrend" else -1 if trend == "downtrend" else 0
        signals.append(Signal("dow", "trend", dow_dir,
                               0.9 if dow_dir else 0.4,
                               self.WEIGHTS["dow"],
                               f"Dow trend: {trend} ({trend_reason})"))

    # ----- momentum --------------------------------------------------------
    def _momentum_signals(self, df, signals, snap):
        close = df["close"]
        rsi_v = ind.rsi(close).iloc[-1]
        snap["rsi"] = self._r(ind.rsi(close))
        if pd.notna(rsi_v):
            if rsi_v < 30:
                signals.append(Signal("rsi", "momentum", 1,
                                       min(1.0, (30 - rsi_v) / 20.0),
                                       self.WEIGHTS["rsi"],
                                       f"RSI {rsi_v:.0f} oversold"))
            elif rsi_v > 70:
                signals.append(Signal("rsi", "momentum", -1,
                                       min(1.0, (rsi_v - 70) / 20.0),
                                       self.WEIGHTS["rsi"],
                                       f"RSI {rsi_v:.0f} overbought"))
            else:
                d = 1 if rsi_v > 50 else -1
                signals.append(Signal("rsi", "momentum", d,
                                       abs(rsi_v - 50) / 20.0,
                                       self.WEIGHTS["rsi"],
                                       f"RSI {rsi_v:.0f} {'firm' if d > 0 else 'soft'}"))

        macd_df = ind.macd(close)
        macd_v = macd_df["macd"].iloc[-1]
        sig_v = macd_df["signal"].iloc[-1]
        hist = macd_df["hist"]
        snap["macd"] = self._r(macd_df["macd"])
        snap["macd_signal"] = self._r(macd_df["signal"])
        if pd.notna(macd_v) and pd.notna(sig_v):
            direction = 1 if macd_v > sig_v else -1
            crossed = len(hist) > 2 and hist.iloc[-1] * hist.iloc[-2] < 0
            signals.append(Signal(
                "macd", "momentum", direction, 1.0 if crossed else 0.6,
                self.WEIGHTS["macd"],
                f"MACD {'bullish cross' if crossed and direction > 0 else 'bearish cross' if crossed else 'above signal' if direction > 0 else 'below signal'}"))

        wr = ind.williams_r(df).iloc[-1]
        if pd.notna(wr):
            if wr < -80:
                signals.append(Signal("williams", "momentum", 1,
                                       min(1.0, (-80 - wr) / 20.0),
                                       self.WEIGHTS["williams"],
                                       f"Williams %R {wr:.0f} oversold"))
            elif wr > -20:
                signals.append(Signal("williams", "momentum", -1,
                                       min(1.0, (wr + 20) / 20.0),
                                       self.WEIGHTS["williams"],
                                       f"Williams %R {wr:.0f} overbought"))

        stoch = ind.stochastic(df)
        k = stoch["k"].iloc[-1]
        if pd.notna(k):
            if k < 20:
                signals.append(Signal("stoch", "momentum", 1,
                                       min(1.0, (20 - k) / 20.0),
                                       self.WEIGHTS["stoch"],
                                       f"Stochastic %K {k:.0f} oversold"))
            elif k > 80:
                signals.append(Signal("stoch", "momentum", -1,
                                       min(1.0, (k - 80) / 20.0),
                                       self.WEIGHTS["stoch"],
                                       f"Stochastic %K {k:.0f} overbought"))

        mfi_v = ind.money_flow_index(df).iloc[-1]
        snap["mfi"] = self._r(ind.money_flow_index(df))
        if pd.notna(mfi_v):
            if mfi_v < 20:
                signals.append(Signal("mfi", "momentum", 1,
                                       min(1.0, (20 - mfi_v) / 20.0),
                                       self.WEIGHTS["mfi"],
                                       f"MFI {mfi_v:.0f} - money flow oversold"))
            elif mfi_v > 80:
                signals.append(Signal("mfi", "momentum", -1,
                                       min(1.0, (mfi_v - 80) / 20.0),
                                       self.WEIGHTS["mfi"],
                                       f"MFI {mfi_v:.0f} - money flow overbought"))

    # ----- volatility ------------------------------------------------------
    def _volatility_signals(self, df, signals, snap):
        bb = ind.bollinger_bands(df["close"])
        pct_b = bb["pct_b"].iloc[-1]
        snap["bb_pct_b"] = self._r(bb["pct_b"])
        snap["atr"] = self._r(ind.atr(df))
        if pd.notna(pct_b):
            if pct_b < 0.0:
                signals.append(Signal("bollinger", "volatility", 1,
                                       min(1.0, -pct_b + 0.3),
                                       self.WEIGHTS["bollinger"],
                                       "price pierced lower Bollinger band"))
            elif pct_b > 1.0:
                signals.append(Signal("bollinger", "volatility", -1,
                                       min(1.0, pct_b - 1.0 + 0.3),
                                       self.WEIGHTS["bollinger"],
                                       "price pierced upper Bollinger band"))
            else:
                d = 1 if pct_b > 0.5 else -1
                signals.append(Signal("bollinger", "volatility", d,
                                       abs(pct_b - 0.5),
                                       self.WEIGHTS["bollinger"],
                                       f"price in {'upper' if d > 0 else 'lower'} Bollinger half"))

    # ----- volume ----------------------------------------------------------
    def _volume_signals(self, df, signals, snap):
        obv = ind.obv(df)
        if len(obv) > 10:
            obv_slope = obv.iloc[-1] - obv.iloc[-10]
            direction = 1 if obv_slope > 0 else -1 if obv_slope < 0 else 0
            signals.append(Signal("obv", "volume", direction, 0.6,
                                   self.WEIGHTS["obv"],
                                   f"OBV {'accumulation' if direction > 0 else 'distribution'}"))

        vr = ind.volume_ratio(df).iloc[-1]
        snap["volume_ratio"] = self._r(ind.volume_ratio(df))
        ret = df["close"].iloc[-1] - df["close"].iloc[-2]
        if pd.notna(vr) and vr > 1.5:
            direction = 1 if ret > 0 else -1
            signals.append(Signal("volume", "volume", direction,
                                   min(1.0, (vr - 1.5) / 2.0),
                                   self.WEIGHTS["volume"],
                                   f"volume {vr:.1f}x average confirms move"))

        vwap = ind.vwap(df).iloc[-1]
        snap["vwap"] = round(float(vwap), 2) if pd.notna(vwap) else 0.0
        if pd.notna(vwap):
            direction = 1 if df["close"].iloc[-1] > vwap else -1
            signals.append(Signal("vwap", "volume", direction, 0.5,
                                   self.WEIGHTS["vwap"],
                                   f"price {'above' if direction > 0 else 'below'} VWAP"))

    # ----- patterns --------------------------------------------------------
    def _pattern_signals(self, df, signals) -> list[str]:
        names: list[str] = []
        for hit in cs.recent_hits(df, lookback=3):
            direction = {"bullish": 1, "bearish": -1, "neutral": 0}[hit.bias]
            strength = max(0.3, 1.0 - 0.25 * hit.bars_ago)
            signals.append(Signal(
                f"candle:{hit.name}", "pattern", direction, strength,
                self.WEIGHTS["candlestick"],
                f"{hit.name.replace('_', ' ')} candlestick"
                f"{'' if hit.bars_ago == 0 else f' ({hit.bars_ago} bars ago)'}"))
            names.append(hit.name)

        for pat in cp.detect_patterns(df):
            direction = {"bullish": 1, "bearish": -1, "neutral": 0}[pat.bias]
            signals.append(Signal(
                f"chart:{pat.name}", "pattern", direction, 0.7,
                self.WEIGHTS["chart"],
                f"{pat.name.replace('_', ' ')} - {pat.detail}"))
            names.append(pat.name)

        for gap in gp.detect_gaps(df):
            if gap.kind == gp.COMMON or gap.bars_ago > 10:
                continue
            direction = {"bullish": 1, "bearish": -1, "neutral": 0}[gap.bias]
            signals.append(Signal(
                f"gap:{gap.kind}", "pattern", direction, 0.6,
                self.WEIGHTS["gap"],
                f"{gap.kind} gap {gap.direction} {gap.size_pct}%"))
            names.append(f"{gap.kind}_gap")
        return names

    # ----- multi-timeframe -------------------------------------------------
    def _mtf_signal(self, higher_tf, trend, signals) -> bool:
        if higher_tf is None or len(higher_tf) < self.MIN_BARS:
            return False
        htf_trend, _ = self._dow_trend(validate_ohlcv(higher_tf))
        if htf_trend == "sideways" or trend == "sideways":
            return False
        agree = htf_trend == trend
        direction = (1 if htf_trend == "uptrend" else -1) if agree else 0
        signals.append(Signal(
            "mtf", "trend", direction, 0.9 if agree else 0.5,
            self.WEIGHTS["mtf"],
            f"higher timeframe {htf_trend} "
            f"{'confirms' if agree else 'conflicts with'} signal"))
        return agree

    # ----- aggregation -----------------------------------------------------
    def _aggregate(self, signals: list[Signal]) -> tuple[float, dict[str, float]]:
        total_w = sum(s.weight for s in signals) or 1.0
        weighted = sum(s.contribution for s in signals)
        score = 50.0 + 50.0 * (weighted / total_w)
        score = round(max(0.0, min(100.0, score)), 1)

        sub: dict[str, float] = {}
        for cat in ("trend", "momentum", "volatility", "volume", "pattern"):
            cat_sigs = [s for s in signals if s.category == cat]
            if not cat_sigs:
                continue
            w = sum(s.weight for s in cat_sigs) or 1.0
            c = sum(s.contribution for s in cat_sigs)
            sub[cat] = round(max(0.0, min(100.0, 50.0 + 50.0 * (c / w))), 1)
        return score, sub

    def _top_reasons(self, signals: list[Signal], n: int = 6) -> list[str]:
        ranked = sorted(signals, key=lambda s: abs(s.contribution), reverse=True)
        out = []
        for s in ranked:
            if s.direction == 0:
                continue
            arrow = "+" if s.direction > 0 else "-"
            out.append(f"[{arrow}] {s.reason}")
            if len(out) >= n:
                break
        return out

    @staticmethod
    def _r(series: pd.Series) -> float:
        v = series.iloc[-1] if len(series) else None
        return round(float(v), 2) if v is not None and pd.notna(v) else 0.0
