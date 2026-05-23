"""CLI entry point for the backtest + calibration pipeline.

Run with:  python -m backtest.run --horizon 10 --train-frac 0.7
"""
from __future__ import annotations

import argparse

from backtest.calibration import DEFAULT_CALIBRATOR_PATH
from backtest.harness import run_backtest
from config.settings import get_settings


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizon", type=int, default=10,
                   help="forward horizon in trading bars (default 10)")
    p.add_argument("--train-frac", type=float, default=0.7,
                   help="chronological train fraction (default 0.7)")
    p.add_argument("--save", default=str(DEFAULT_CALIBRATOR_PATH),
                   help="path to write the fitted calibrator JSON")
    args = p.parse_args()

    settings = get_settings()
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
    return 0 if cal.has_lift else 0   # exit zero either way - "no lift" is honest output


if __name__ == "__main__":
    raise SystemExit(main())
