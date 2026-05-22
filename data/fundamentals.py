"""Fundamental data ingestion.

Defines the data shape and provider interface the fundamental engine
consumes. The concrete providers (yfinance free baseline, then
TwelveData / FMP) are wired up in Milestone 4; the interface is fixed
here so the rest of the system can be built against it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class FundamentalData:
    """Point-in-time fundamental snapshot for one company."""

    symbol: str
    pe: float | None = None              # price / earnings
    pb: float | None = None              # price / book
    peg: float | None = None             # PE / earnings growth
    roe: float | None = None             # return on equity %
    roce: float | None = None            # return on capital employed %
    debt_to_equity: float | None = None
    interest_coverage: float | None = None
    revenue_growth: float | None = None  # YoY %
    eps_growth: float | None = None      # YoY %
    operating_margin: float | None = None
    net_margin: float | None = None
    dividend_yield: float | None = None
    market_cap: float | None = None
    promoter_holding: float | None = None
    promoter_pledge: float | None = None
    sector: str = ""
    available: bool = False              # True once a provider has populated it

    def to_dict(self) -> dict:
        return asdict(self)


class FundamentalsProvider:
    """Interface for fundamental data sources.

    Implementations: YFinanceFundamentals (free), TwelveDataFundamentals,
    FMPFundamentals - delivered in Milestone 4.
    """

    name: str = "base"

    def fetch(self, symbol: str) -> FundamentalData:
        raise NotImplementedError("fundamentals ingestion lands in Milestone 4")


def get_provider(settings) -> FundamentalsProvider:
    """Factory dispatching on settings.fundamentals_provider (Milestone 4)."""
    raise NotImplementedError(
        "fundamentals provider wiring lands in Milestone 4 "
        f"(requested provider: {settings.fundamentals_provider!r})"
    )
