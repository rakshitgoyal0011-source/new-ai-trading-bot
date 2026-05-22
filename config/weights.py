"""Configurable weights for the composite buy-probability signal.

The composite blends three engine scores (technical / fundamental / news).
Weights are user-tunable per command, e.g.  `TOP W=60/30/10`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalWeights:
    """Relative importance of each analysis engine in the composite score."""

    technical: float = 0.45
    fundamental: float = 0.35
    news: float = 0.20

    @property
    def total(self) -> float:
        return self.technical + self.fundamental + self.news

    def normalized(self) -> "SignalWeights":
        """Return weights rescaled to sum to 1.0 (defensive for user input)."""
        t = self.total or 1.0
        return SignalWeights(
            self.technical / t, self.fundamental / t, self.news / t
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "technical": self.technical,
            "fundamental": self.fundamental,
            "news": self.news,
        }

    @classmethod
    def from_settings(cls, settings) -> "SignalWeights":
        return cls(
            technical=settings.weight_technical,
            fundamental=settings.weight_fundamental,
            news=settings.weight_news,
        ).normalized()

    @classmethod
    def parse(cls, spec: str) -> "SignalWeights":
        """Parse a `45/35/20` style weight spec into normalized weights."""
        parts = [p for p in spec.replace(" ", "").split("/") if p]
        if len(parts) != 3:
            raise ValueError("weight spec must look like '45/35/20'")
        t, f, n = (float(p) for p in parts)
        return cls(t, f, n).normalized()


DEFAULT_WEIGHTS = SignalWeights()
