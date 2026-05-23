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
is Milestone 6. If a fitted calibrator exists on disk AND it shows lift
over the baseline, `calibrated_probability` is set; otherwise it stays
None and the UI must show the score as an uncalibrated estimate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from config.weights import DEFAULT_WEIGHTS, SignalWeights
from monitoring.logging import get_logger

log = get_logger("composite")

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
    calibrated_probability: float | None = None
    probability_band: list[float] | None = None
    historical_hit_rate: float | None = None
    horizon_days: int | None = None
    calibration_note: str = (
        "Uncalibrated composite. Run `python -m backtest.run` to fit a "
        "calibrator and turn this score into a probability."
    )
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


class CompositeEngine:
    """Weighted blend of engine scores into one composite signal."""

    def __init__(
        self,
        weights: SignalWeights | None = None,
        calibrator=None,
        calibrator_path: str | Path | None = None,
    ):
        self.weights = (weights or DEFAULT_WEIGHTS).normalized()
        self.calibrator = calibrator if calibrator is not None else self._load(calibrator_path)

    @staticmethod
    def _load(path: str | Path | None):
        """Best-effort load of a persisted calibrator."""
        try:
            from backtest.calibration import DEFAULT_CALIBRATOR_PATH, Calibrator
        except ImportError:
            return None
        p = Path(path) if path is not None else DEFAULT_CALIBRATOR_PATH
        if not p.exists():
            return None
        try:
            cal = Calibrator.load(p)
        except Exception as exc:
            log.warning("could not load calibrator from %s: %s", p, exc)
            return None
        if not cal.has_lift:
            log.info("calibrator at %s shows no lift - probability stays None", p)
        return cal

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

        cal_prob: float | None = None
        horizon_days: int | None = None
        cal_note = (
            "Uncalibrated composite. Run `python -m backtest.run` to fit a "
            "calibrator and turn this score into a probability."
        )
        if self.calibrator is not None:
            horizon_days = self.calibrator.horizon_bars
            if self.calibrator.has_lift:
                cal_prob = self.calibrator.predict(composite)
                if cal_prob is not None:
                    cal_prob = round(cal_prob, 3)
                cal_note = (
                    f"Calibrated on {self.calibrator.n_test} test points, "
                    f"horizon={self.calibrator.horizon_bars} bars, "
                    f"Brier {self.calibrator.brier_test:.4f} "
                    f"(baseline {self.calibrator.brier_baseline:.4f})."
                )
            else:
                cal_note = (
                    "Calibrator fit but showed no lift over baseline - "
                    "probability intentionally suppressed. "
                    f"({self.calibrator.lift_note})"
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
            calibrated_probability=cal_prob,
            horizon_days=horizon_days,
            calibration_note=cal_note,
        )

    def calibrated_probability(self, score: float) -> float | None:
        """Direct calibrator lookup for a 0-100 composite score."""
        if self.calibrator is None or not self.calibrator.has_lift:
            return None
        return self.calibrator.predict(score)
