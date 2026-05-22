"""Composite buy-probability engine.

Blends the technical / fundamental / news engine scores into a single
composite using configurable weights. When fundamental or news scores
are missing the weights are renormalised over what IS available, so the
composite never silently treats a missing input as zero.

HONESTY CONTRACT
----------------
The composite score is NOT a probability. Converting it to a calibrated
"probability of a positive return over horizon H" needs a model trained
and backtested on history, then calibrated (Platt / isotonic). That work
is Milestone 6. Until then `calibrated_probability` stays None and the
UI must show the score as an uncalibrated estimate - never as a promise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from config.weights import DEFAULT_WEIGHTS, SignalWeights

DISCLAIMER = (
    "NOT FINANCIAL ADVICE. Scores and probabilities are model estimates "
    "from historical data, not guarantees. Do your own research."
)


@dataclass
class CompositeResult:
    symbol: str
    composite_score: float                       # 0-100, uncalibrated
    bias: str                                    # bullish | bearish | neutral
    components: dict[str, float] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    calibrated_probability: float | None = None  # set in Milestone 6
    probability_band: list[float] | None = None  # confidence band, Milestone 6
    historical_hit_rate: float | None = None     # similar past signals, M6
    horizon_days: int | None = None
    calibration_note: str = (
        "Uncalibrated composite. Calibrated probability + confidence band "
        "+ historical hit-rate are delivered in Milestone 6 (backtest)."
    )
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


class CompositeEngine:
    """Weighted blend of engine scores into one composite signal."""

    def __init__(self, weights: SignalWeights | None = None):
        self.weights = (weights or DEFAULT_WEIGHTS).normalized()

    def combine(
        self,
        symbol: str,
        technical_score: float,
        fundamental_score: float | None = None,
        news_score: float | None = None,
        weights: SignalWeights | None = None,
        reasons: list[str] | None = None,
    ) -> CompositeResult:
        """Blend available engine scores into a composite (0-100)."""
        w = (weights or self.weights).normalized()

        parts: dict[str, tuple[float, float]] = {
            "technical": (technical_score, w.technical)
        }
        if fundamental_score is not None:
            parts["fundamental"] = (fundamental_score, w.fundamental)
        if news_score is not None:
            parts["news"] = (news_score, w.news)

        total_w = sum(weight for _, weight in parts.values()) or 1.0
        composite = sum(score * weight for score, weight in parts.values()) / total_w
        composite = round(max(0.0, min(100.0, composite)), 1)

        bias = (
            "bullish" if composite >= 60
            else "bearish" if composite <= 40
            else "neutral"
        )
        return CompositeResult(
            symbol=symbol,
            composite_score=composite,
            bias=bias,
            components={k: round(v[0], 1) for k, v in parts.items()},
            weights_used={
                k: round(v[1] / total_w, 3) for k, v in parts.items()
            },
            reasons=list(reasons or []),
        )

    def calibrated_probability(self, *_args, **_kwargs):
        """Convert a composite score to a calibrated probability.

        Implemented in Milestone 6: a logistic / gradient-boosted model is
        trained on score features, validated walk-forward, then calibrated
        with Platt scaling or isotonic regression so the number is honest.
        """
        raise NotImplementedError(
            "calibrated probability requires the trained + calibrated model "
            "from Milestone 6 (backtest & validation)"
        )
