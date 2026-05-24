"""Fundamental data ingestion.

`FundamentalsProvider` is the interface; two implementations ship today:
- `DemoFundamentals`      synthetic, deterministic per symbol (no network)
- `YFinanceFundamentals`  free baseline via the yfinance package
                          (Yahoo's NSE listings are `<SYMBOL>.NS`)

The factory `get_provider(settings)` picks one based on DALAL_MODE and
FUNDAMENTALS_PROVIDER. Both providers cache per-symbol with a TTL so we
do not hammer Yahoo every refresh.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from config import universe
from data._seeding import deterministic_seed
from monitoring.logging import get_logger

log = get_logger("data.fundamentals")


@dataclass
class FundamentalData:
    symbol: str
    pe: float | None = None
    pb: float | None = None
    peg: float | None = None
    roe: float | None = None              # %
    roce: float | None = None             # %
    debt_to_equity: float | None = None
    interest_coverage: float | None = None
    revenue_growth: float | None = None   # YoY %
    eps_growth: float | None = None       # YoY %
    operating_margin: float | None = None # %
    net_margin: float | None = None       # %
    dividend_yield: float | None = None   # %
    market_cap: float | None = None       # Rs.
    promoter_holding: float | None = None # %
    promoter_pledge: float | None = None  # %
    sector: str = ""
    available: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class FundamentalsProvider:
    name: str = "base"

    def fetch(self, symbol: str) -> FundamentalData:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Demo (synthetic) fundamentals - always available, no network.
# --------------------------------------------------------------------------
class DemoFundamentals(FundamentalsProvider):
    name = "demo"

    def fetch(self, symbol: str) -> FundamentalData:
        rng = np.random.default_rng(deterministic_seed("fa_", symbol))
        stock = universe.get(symbol)
        return FundamentalData(
            symbol=symbol,
            pe=float(rng.uniform(8, 60)),
            pb=float(rng.uniform(0.8, 14)),
            peg=float(rng.uniform(0.5, 4)),
            roe=float(rng.uniform(-5, 40)),
            roce=float(rng.uniform(-3, 45)),
            debt_to_equity=float(rng.uniform(0, 3.2)),
            interest_coverage=float(rng.uniform(1, 28)),
            revenue_growth=float(rng.uniform(-15, 40)),
            eps_growth=float(rng.uniform(-25, 55)),
            operating_margin=float(rng.uniform(-5, 38)),
            net_margin=float(rng.uniform(-10, 28)),
            dividend_yield=float(rng.uniform(0, 5)),
            market_cap=float(rng.uniform(1e10, 4e12)),
            promoter_holding=float(rng.uniform(20, 75)),
            promoter_pledge=float(rng.uniform(0, 25)),
            sector=stock.sector if stock else "",
            available=True,
        )


# --------------------------------------------------------------------------
# yfinance - free Yahoo Finance fundamentals (.NS suffix for NSE).
# --------------------------------------------------------------------------
class YFinanceFundamentals(FundamentalsProvider):
    name = "yfinance"

    def __init__(self, ttl_seconds: int = 6 * 3600):
        self._cache: dict[str, tuple[float, FundamentalData]] = {}
        self._ttl = ttl_seconds

    def fetch(self, symbol: str) -> FundamentalData:
        cached = self._cache.get(symbol)
        if cached and time.time() - cached[0] < self._ttl:
            return cached[1]

        data = self._fetch_uncached(symbol)
        self._cache[symbol] = (time.time(), data)
        return data

    def _fetch_uncached(self, symbol: str) -> FundamentalData:
        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance not installed - falling back to unavailable")
            return FundamentalData(symbol=symbol, available=False)

        try:
            ticker = yf.Ticker(symbol + ".NS")
            info = ticker.info or {}
        except Exception as exc:
            log.warning("yfinance fetch failed for %s: %s", symbol, exc)
            return FundamentalData(symbol=symbol, available=False)

        stock = universe.get(symbol)
        roe = info.get("returnOnEquity")
        margin_op = info.get("operatingMargins")
        margin_net = info.get("profitMargins")
        rev_g = info.get("revenueGrowth")
        eps_g = info.get("earningsGrowth")
        div_y = info.get("dividendYield")
        # yfinance reports debtToEquity as a percentage (150 == 1.5x), not
        # the ratio our engine + UI expect; normalise to ratio here.
        de_raw = info.get("debtToEquity")

        return FundamentalData(
            symbol=symbol,
            pe=info.get("trailingPE"),
            pb=info.get("priceToBook"),
            peg=info.get("pegRatio"),
            roe=(roe * 100.0) if roe is not None else None,
            debt_to_equity=(de_raw / 100.0) if de_raw is not None else None,
            revenue_growth=(rev_g * 100.0) if rev_g is not None else None,
            eps_growth=(eps_g * 100.0) if eps_g is not None else None,
            operating_margin=(margin_op * 100.0) if margin_op is not None else None,
            net_margin=(margin_net * 100.0) if margin_net is not None else None,
            dividend_yield=(div_y * 100.0) if div_y is not None else None,
            market_cap=info.get("marketCap"),
            sector=info.get("sector") or (stock.sector if stock else ""),
            available=bool(info.get("trailingPE") or info.get("priceToBook")),
        )


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------
def get_provider(settings) -> FundamentalsProvider:
    """Pick a fundamentals provider based on settings."""
    if not settings.is_live or settings.fundamentals_provider == "demo":
        return DemoFundamentals()
    if settings.fundamentals_provider == "yfinance":
        return YFinanceFundamentals()
    # paid providers (twelvedata, fmp) land here once wired up
    log.warning("provider %r not yet wired - using yfinance baseline",
                settings.fundamentals_provider)
    return YFinanceFundamentals()
