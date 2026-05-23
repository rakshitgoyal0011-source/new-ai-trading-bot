"""CLI entry point for the backtest + calibration pipeline.

Default mode is a single chronological 70/30 split:
  python -m backtest.run --horizon 10 --train-frac 0.7

Add `--walk` to use rolling-window walk-forward instead (per-fold
diagnostics + a stability gate on the final calibrator):
  python -m backtest.run --walk --horizon 10 --train 1000 --test 200 --step 200
"""
from __future__ import annotations

import argparse

from backtest.calibration import DEFAULT_CALIBRATOR_PATH
from backtest.harness import run_backtest
from backtest.walkforward import run_walk_forward
from config.settings import get_settings


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizon", type=int, default=10,
                   help="forward horizon in trading bars (default 10)")
    p.add_argument("--train-frac", type=float, default=0.7,
                   help="single-split train fraction (default 0.7)")
    p.add_argument("--walk", action="store_true",
                   help="use rolling-window walk-forward instead of a single split")
    p.add_argument("--train", type=int, default=1000,
                   help="walk-forward train window size (default 1000)")
    p.add_argument("--test", type=int, default=200,
                   help="walk-forward test window size (default 200)")
    p.add_argument("--step", type=int, default=200,
                   help="walk-forward step size (default 200)")
    p.add_argument("--save", default=str(DEFAULT_CALIBRATOR_PATH),
                   help="path to write the fitted calibrator JSON")
    args = p.parse_args()

    settings = get_settings()
    if args.walk:
        return _run_walk(settings, args)
    return _run_single(settings, args)


def _run_single(settings, args) -> int:
    report = run_backtest(
        settings,
        horizon_bars=args.horizon,
        train_frac=args.train_frac,
        save_path=args.save,
    )
    cal = report.calibrator
    print()
    print("=" * 60)
    print(f"BACKTEST   horizon={cal.horizon_bars} bars   points={report.n_points}")
    print("=" * 60)
    print(f"  train n          {cal.n_train}")
    print(f"  test  n          {cal.n_test}")
    print(f"  base rate        {cal.base_rate:.3f}")
    print(f"  brier (train)    {cal.brier_train:.4f}")
    print(f"  brier (test)     {cal.brier_test:.4f}")
    print(f"  brier baseline   {cal.brier_baseline:.4f}")
    print(f"  ECE   (test)     {cal.ece_test:.4f}")
    print(f"  AUC   (test)     {cal.auc_test:.3f}")
    print(f"  has lift?        {cal.has_lift}")
    print(f"  note             {cal.lift_note}")
    print()
    print("  reliability (predicted -> observed | n):")
    for row in cal.reliability:
        obs = (f"{row['observed']:.3f}" if row["observed"] is not None
               else "    -")
        bar = ""
        if row["observed"] is not None:
            bar = "#" * max(0, int(round(row["observed"] * 30)))
        print(f"    {row['predicted']:.3f} -> {obs}  n={row['count']:>4}  {bar}")
    print()
    print(f"saved to {args.save}")
    return 0


def _run_walk(settings, args) -> int:
    report = run_walk_forward(
        settings,
        horizon_bars=args.horizon,
        train_size=args.train,
        test_size=args.test,
        step=args.step,
        save_path=args.save,
    )
    cal = report.final_calibrator
    print()
    print("=" * 60)
    print(f"WALK-FORWARD   horizon={report.horizon_bars} bars   "
          f"points={report.n_points}   folds={report.fold_count}")
    print("=" * 60)
    print(f"  window           train={args.train}  test={args.test}  step={args.step}")
    print(f"  lift in folds    {report.lift_folds}/{report.fold_count}")
    print(f"  stable lift?     {report.stable_lift}")
    print(f"  avg Brier (test) {report.avg_brier_test:.4f}")
    print(f"  avg Brier base   {report.avg_brier_baseline:.4f}")
    print(f"  avg AUC   (test) {report.avg_auc_test:.3f}")
    print(f"  avg ECE   (test) {report.avg_ece_test:.4f}")
    print()
    print("  per-fold:")
    print("    fold   n_train  n_test  Brier   base    AUC     ECE     lift")
    for f in report.folds:
        print(f"    {f.fold:>4d}   {f.n_train:>7d}  {f.n_test:>6d}  "
              f"{f.brier_test:.4f}  {f.brier_baseline:.4f}  "
              f"{f.auc_test:.3f}   {f.ece_test:.4f}  {f.has_lift}")
    print()
    print(f"  final calibrator (most-recent window):")
    print(f"    has_lift  {cal.has_lift}")
    print(f"    note      {cal.lift_note}")
    print()
    print(f"saved to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
