"""Backtest prediction-point generation.

For every (symbol, t) we evaluate the technical engine on the price
window that ENDED at t (no look-ahead), pair it with the static
fundamental + sentiment scores, blend them through the same
CompositeEngine used live, and record the realised forward return over
H bars. Labels are 1 iff forward_return > 0.

Demo mode keeps fundamentals + sentiment static per symbol - their live,
point-in-time histories arrive once Kite + paid fundamentals land.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from analysis.composite.engine import CompositeEngine
from analysis.fundamental.engine import FundamentalEngine
from analysis.sentiment.engine import SentimentEngine
from analysis.technical import TechnicalEngine
from config import universe
from data import fundamentals as fund_data_mod
from data import news as news_data_mod
from data.history import HistoryService
from monitoring.logging import get_logger

log = get_logger("backtest.events")


@dataclass
class PredictionPoint:
    symbol: str
    date: str
    composite_score: float
    technical_score: float
    fundamental_score: float | None
    sentiment_score: float | None
    forward_return: float
    label: int

    def to_dict(self) -> dict:
        return asdict(self)


def generate_points(
    settings,
    symbols: Iterable[str] | None = None,
    horizon_bars: int = 10,
    lookback_bars: int = 200,
    history_bars: int = 260,
) -> list[PredictionPoint]:
    """Replay history and produce (composite_score, forward_return) pairs.

    `lookback_bars` is the minimum window the technical engine needs to
    settle (EMA50, MACD, RSI). `history_bars` is total bars pulled per
    symbol from HistoryService.
    """
    history = HistoryService(settings)
    tech_engine = TechnicalEngine()
    fund_engine = FundamentalEngine()
    sent_engine = SentimentEngine(model=settings.sentiment_model)
    composite = CompositeEngine()  # weights from defaults
    fund_provider = fund_data_mod.get_provider(settings)
    news_provider = news_data_mod.get_provider(settings)

    selected = list(symbols) if symbols else [s.symbol for s in universe.DEFAULT_UNIVERSE]

    # Static-in-demo fundamental + sentiment scores per symbol.
    fund_lookup = {sym: fund_provider.fetch(sym) for sym in selected}
    static: dict[str, tuple[float | None, float | None]] = {}
    for sym in selected:
        peers = [
            fund_lookup[p.symbol]
            for p in universe.peers(sym)
            if p.symbol in fund_lookup and fund_lookup[p.symbol].available
        ]
        fund_res = fund_engine.analyze(sym, fund_lookup.get(sym), peers)
        items = news_provider.fetch(sym, limit=5)
        sent_res = sent_engine.analyze(sym, items)
        static[sym] = (
            fund_res.score if fund_res.available else None,
            sent_res.score if items else None,
        )

    points: list[PredictionPoint] = []
    for sym in selected:
        try:
            df = history.candles(sym, "day", history_bars)
        except Exception as exc:
            log.warning("history fetch failed for %s: %s", sym, exc)
            continue
        if len(df) < lookback_bars + horizon_bars + 1:
            continue
        fund_score, sent_score = static[sym]

        closes = df["close"].to_numpy()
        for t in range(lookback_bars, len(df) - horizon_bars):
            window = df.iloc[: t + 1]
            try:
                tech_res = tech_engine.analyze(sym, window)
            except Exception:
                continue
            comp = composite.combine(
                sym, tech_res.score, fund_score, sent_score
            )
            entry = float(closes[t])
            exit_ = float(closes[t + horizon_bars])
            if entry <= 0:
                continue
            fwd = exit_ / entry - 1.0
            points.append(PredictionPoint(
                symbol=sym,
                date=str(df.index[t]),
                composite_score=comp.composite_score,
                technical_score=tech_res.score,
                fundamental_score=fund_score,
                sentiment_score=sent_score,
                forward_return=fwd,
                label=1 if fwd > 0 else 0,
            ))

    points.sort(key=lambda p: p.date)
    log.info("generated %d prediction points across %d symbols",
             len(points), len(selected))
    return points
