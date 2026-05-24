"""Backtest harness: replay history, fit a calibrator, persist it.

Chronological train/test split (default 70/30) is the simplest honest
walk-forward: fit on the older half of points, evaluate on the newer
half, decide whether to trust the calibrator from the test-set
diagnostics alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from backtest.calibration import (
    DEFAULT_CALIBRATOR_PATH,
    Calibrator,
    fit_calibrator,
)
from backtest.events import PredictionPoint, generate_points
from monitoring.logging import get_logger

log = get_logger("backtest.harness")


@dataclass
class BacktestReport:
    calibrator: Calibrator
    n_points: int
    horizon_bars: int


def run_backtest(
    settings,
    symbols: Iterable[str] | None = None,
    horizon_bars: int = 10,
    train_frac: float = 0.7,
    save_path: str | Path | None = DEFAULT_CALIBRATOR_PATH,
    points: list[PredictionPoint] | None = None,
) -> BacktestReport:
    """Run the full backtest pipeline and (optionally) persist the calibrator.

    `points` lets tests inject prefab data without paying the cost of
    iterating real history. In production it stays None.
    """
    points = points or generate_points(
        settings, symbols=symbols, horizon_bars=horizon_bars,
    )
    if len(points) < 20:
        raise RuntimeError(
            f"only {len(points)} prediction points - "
            "need >= 20 to fit a calibrator"
        )

    split = max(10, int(len(points) * train_frac))
    train = points[:split]
    test = points[split:]
    if len(test) < 5:
        raise RuntimeError(
            f"test split has {len(test)} points - increase history or "
            "lower train_frac"
        )

    train_scores = np.array([p.composite_score for p in train])
    train_labels = np.array([p.label for p in train])
    test_scores = np.array([p.composite_score for p in test])
    test_labels = np.array([p.label for p in test])

    calibrator = fit_calibrator(
        train_scores, train_labels,
        test_scores, test_labels,
        horizon_bars=horizon_bars,
    )
    log.info(
        "backtest: %d train + %d test points, Brier %.4f (baseline %.4f), "
        "AUC %.3f, lift=%s",
        len(train), len(test),
        calibrator.brier_test, calibrator.brier_baseline,
        calibrator.auc_test, calibrator.has_lift,
    )
    if save_path is not None:
        calibrator.save(save_path)
        log.info("saved calibrator to %s", save_path)
    return BacktestReport(
        calibrator=calibrator,
        n_points=len(points),
        horizon_bars=horizon_bars,
    )
