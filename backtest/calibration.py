"""Probability calibration: pool-adjacent-violators isotonic regression
plus the reliability diagnostics needed to decide whether a calibrator
shows real lift over a constant baseline.

We deliberately avoid scikit-learn here - PAV is ~40 lines and keeps the
dependency surface small. The Calibrator is JSON-serialisable so it can
be re-loaded by the CompositeEngine without re-running the backtest.

HONESTY CONTRACT
----------------
A Calibrator only feeds `calibrated_probability` through the
CompositeEngine when `has_lift` is True. "Lift" means it beats the
constant-base-rate Brier baseline by a meaningful margin AND the
out-of-sample AUC clears 0.52. If neither holds we keep
`calibrated_probability = None` rather than fabricate a probability.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# ----- core algorithm ------------------------------------------------------


def pav_isotonic(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators non-decreasing isotonic regression.

    Returns (breakpoints, values): contiguous blocks indexed by the
    upper bound of each block's x range. Predict by searching the
    breakpoints with `isotonic_predict`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    y_sorted = y[order]

    sums: list[float] = []
    weights: list[float] = []
    maxes: list[float] = []
    for xi, yi in zip(x_sorted, y_sorted):
        s, w = float(yi), 1.0
        while sums and (sums[-1] / weights[-1]) > (s / w):
            s += sums.pop()
            w += weights.pop()
            maxes.pop()
        sums.append(s)
        weights.append(w)
        maxes.append(float(xi))
    values = np.array([s / w for s, w in zip(sums, weights)])
    return np.array(maxes), values


def isotonic_predict(scores, breakpoints, values) -> np.ndarray:
    breakpoints = np.asarray(breakpoints)
    values = np.asarray(values)
    scores = np.asarray(scores, dtype=float)
    if breakpoints.size == 0:
        return np.full(scores.shape, 0.5)
    idx = np.searchsorted(breakpoints, scores, side="left")
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


# ----- diagnostics ---------------------------------------------------------


def brier_score(probs, labels) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2)) if p.size else 0.0


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p < hi) if i < n_bins - 1 else (p <= hi))
        if not mask.any():
            continue
        ece += abs(p[mask].mean() - y[mask].mean()) * mask.sum()
    return float(ece / p.size)


def reliability_diagram(probs, labels, n_bins: int = 10) -> list[dict]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p < hi) if i < n_bins - 1 else (p <= hi))
        if not mask.any():
            out.append({
                "bin_lo": float(lo), "bin_hi": float(hi),
                "predicted": float((lo + hi) / 2),
                "observed": None, "count": 0,
            })
        else:
            out.append({
                "bin_lo": float(lo), "bin_hi": float(hi),
                "predicted": float(p[mask].mean()),
                "observed": float(y[mask].mean()),
                "count": int(mask.sum()),
            })
    return out


def auc(probs, labels) -> float:
    """Wilcoxon-Mann-Whitney AUC, ties get average rank. O(n log n)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    if p.size == 0:
        return 0.5
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    sorted_y = y[order]
    ranks = np.empty_like(sorted_p, dtype=float)
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[i: j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos_rank_sum = float(ranks[sorted_y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ----- Calibrator ----------------------------------------------------------

# thresholds beyond which we trust the fit
_BRIER_LIFT_MIN = 0.005
_AUC_LIFT_MIN = 0.52


@dataclass
class Calibrator:
    breakpoints: list[float]
    values: list[float]
    horizon_bars: int
    n_train: int
    n_test: int
    base_rate: float
    brier_train: float
    brier_test: float
    brier_baseline: float
    ece_test: float
    auc_test: float
    reliability: list[dict] = field(default_factory=list)
    has_lift: bool = False
    lift_note: str = ""
    fitted_at: str = ""

    def predict(self, score: float) -> float | None:
        """Return calibrated probability for a 0-100 score, or None if
        the calibrator is not trusted (no demonstrated lift)."""
        if not self.has_lift:
            return None
        bp = np.array(self.breakpoints)
        vs = np.array(self.values)
        return float(isotonic_predict([score], bp, vs)[0])

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(**d)


def fit_calibrator(
    train_scores,
    train_labels,
    test_scores,
    test_labels,
    horizon_bars: int,
) -> Calibrator:
    """Fit an isotonic calibrator and capture the diagnostics."""
    bp, vs = pav_isotonic(train_scores, train_labels)
    train_probs = isotonic_predict(train_scores, bp, vs)
    test_probs = isotonic_predict(test_scores, bp, vs)

    base_rate = float(np.mean(train_labels)) if len(train_labels) else 0.5
    brier_baseline = base_rate * (1.0 - base_rate)
    brier_train = brier_score(train_probs, train_labels)
    brier_test = brier_score(test_probs, test_labels)
    ece_test = expected_calibration_error(test_probs, test_labels)
    auc_test = auc(test_probs, test_labels)

    beats_brier = brier_test < (brier_baseline - _BRIER_LIFT_MIN)
    beats_auc = auc_test > _AUC_LIFT_MIN
    has_lift = bool(beats_brier and beats_auc)
    if has_lift:
        note = "calibrator beats baseline - probability exposed via composite"
    else:
        why = []
        if not beats_brier:
            why.append(f"Brier {brier_test:.4f} not below baseline {brier_baseline:.4f} - {_BRIER_LIFT_MIN}")
        if not beats_auc:
            why.append(f"AUC {auc_test:.3f} not above {_AUC_LIFT_MIN}")
        note = "no predictive lift detected (" + "; ".join(why) + ") - probability stays None"

    return Calibrator(
        breakpoints=bp.tolist(),
        values=vs.tolist(),
        horizon_bars=horizon_bars,
        n_train=int(len(train_labels)),
        n_test=int(len(test_labels)),
        base_rate=base_rate,
        brier_train=brier_train,
        brier_test=brier_test,
        brier_baseline=brier_baseline,
        ece_test=ece_test,
        auc_test=auc_test,
        reliability=reliability_diagram(test_probs, test_labels),
        has_lift=has_lift,
        lift_note=note,
        fitted_at=datetime.now().isoformat(timespec="seconds"),
    )


# Anchored at the project root so the running server, pytest, and CLI
# invocations all read/write the same calibrator regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATOR_PATH = _PROJECT_ROOT / "runs" / "calibrator.json"
