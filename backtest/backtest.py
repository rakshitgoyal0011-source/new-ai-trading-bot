"""Backtesting & validation  (Milestone 6).

The realistic Indian-market cost model below is functional now - it is
needed for honest backtests and is unit-tested. The backtest loop,
walk-forward / out-of-sample validation and the calibration curve are
delivered in Milestone 6: that is what turns the composite score into a
trustworthy calibrated probability before it is exposed in the UI.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Round-trip trading costs for Indian equities (rates are approximate
    and configurable - verify against your broker's latest schedule)."""

    brokerage_pct: float = 0.03          # % per leg (intraday-style); 0 for delivery
    brokerage_cap: float = 20.0          # Rs cap per leg
    stt_pct: float = 0.025               # securities txn tax, % on sell leg (intraday)
    exchange_txn_pct: float = 0.00297    # NSE transaction charge %
    sebi_pct: float = 0.0001             # SEBI turnover fee %
    stamp_duty_pct: float = 0.003        # % on buy leg (intraday)
    gst_pct: float = 18.0                # GST % on brokerage + txn + sebi
    slippage_bps: float = 5.0            # modelled slippage per leg, basis points

    def _brokerage(self, value: float) -> float:
        return min(value * self.brokerage_pct / 100.0, self.brokerage_cap)

    def round_trip(self, buy_value: float, sell_value: float) -> float:
        """Total cost (Rs) of buying then selling - charges + slippage."""
        brokerage = self._brokerage(buy_value) + self._brokerage(sell_value)
        stt = sell_value * self.stt_pct / 100.0
        turnover = buy_value + sell_value
        exch = turnover * self.exchange_txn_pct / 100.0
        sebi = turnover * self.sebi_pct / 100.0
        stamp = buy_value * self.stamp_duty_pct / 100.0
        gst = (brokerage + exch + sebi) * self.gst_pct / 100.0
        slippage = turnover * self.slippage_bps / 10_000.0
        return round(brokerage + stt + exch + sebi + stamp + gst + slippage, 2)


@dataclass
class BacktestResult:
    """Summary statistics for a composite-signal backtest (Milestone 6)."""

    trades: int = 0
    hit_rate: float = 0.0
    avg_return_pct: float = 0.0
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    calibration_curve: list[tuple[float, float]] | None = None


class Backtester:
    """Walk-forward backtester for the composite signal (Milestone 6)."""

    def __init__(self, cost_model: CostModel | None = None):
        self.cost_model = cost_model or CostModel()

    def run(self, *_args, **_kwargs) -> BacktestResult:
        raise NotImplementedError(
            "the backtest loop + walk-forward validation + calibration curve "
            "are implemented in Milestone 6"
        )
