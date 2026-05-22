"""Tradable universe + sector map for DALAL TERMINAL.

Ships with the NIFTY 50 as a sane default. The full NIFTY 500 can be loaded
at runtime from the Kite instrument dump (see core/instruments.py) or an
index constituents CSV; sector-relative scoring only needs the sector map.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stock:
    symbol: str          # NSE trading symbol
    name: str            # company name
    sector: str          # coarse sector bucket for peer-relative scoring


# NIFTY 50 constituents (representative snapshot - refresh from the index
# periodically; constituents rotate ~twice a year).
NIFTY50: list[Stock] = [
    Stock("RELIANCE", "Reliance Industries", "Oil & Gas"),
    Stock("TCS", "Tata Consultancy Services", "IT"),
    Stock("HDFCBANK", "HDFC Bank", "Bank"),
    Stock("ICICIBANK", "ICICI Bank", "Bank"),
    Stock("INFY", "Infosys", "IT"),
    Stock("HINDUNILVR", "Hindustan Unilever", "FMCG"),
    Stock("ITC", "ITC", "FMCG"),
    Stock("SBIN", "State Bank of India", "Bank"),
    Stock("BHARTIARTL", "Bharti Airtel", "Telecom"),
    Stock("BAJFINANCE", "Bajaj Finance", "Financial Services"),
    Stock("KOTAKBANK", "Kotak Mahindra Bank", "Bank"),
    Stock("LT", "Larsen & Toubro", "Infrastructure"),
    Stock("HCLTECH", "HCL Technologies", "IT"),
    Stock("AXISBANK", "Axis Bank", "Bank"),
    Stock("MARUTI", "Maruti Suzuki India", "Auto"),
    Stock("SUNPHARMA", "Sun Pharmaceutical", "Pharma"),
    Stock("TITAN", "Titan Company", "Consumer"),
    Stock("ASIANPAINT", "Asian Paints", "Consumer"),
    Stock("ULTRACEMCO", "UltraTech Cement", "Cement"),
    Stock("NESTLEIND", "Nestle India", "FMCG"),
    Stock("WIPRO", "Wipro", "IT"),
    Stock("ONGC", "Oil & Natural Gas Corp", "Oil & Gas"),
    Stock("NTPC", "NTPC", "Power"),
    Stock("POWERGRID", "Power Grid Corp", "Power"),
    Stock("M&M", "Mahindra & Mahindra", "Auto"),
    Stock("TATAMOTORS", "Tata Motors", "Auto"),
    Stock("TATASTEEL", "Tata Steel", "Metal"),
    Stock("JSWSTEEL", "JSW Steel", "Metal"),
    Stock("ADANIENT", "Adani Enterprises", "Infrastructure"),
    Stock("ADANIPORTS", "Adani Ports & SEZ", "Infrastructure"),
    Stock("COALINDIA", "Coal India", "Energy"),
    Stock("GRASIM", "Grasim Industries", "Cement"),
    Stock("HINDALCO", "Hindalco Industries", "Metal"),
    Stock("BAJAJFINSV", "Bajaj Finserv", "Financial Services"),
    Stock("BAJAJ-AUTO", "Bajaj Auto", "Auto"),
    Stock("BRITANNIA", "Britannia Industries", "FMCG"),
    Stock("CIPLA", "Cipla", "Pharma"),
    Stock("DRREDDY", "Dr. Reddy's Laboratories", "Pharma"),
    Stock("EICHERMOT", "Eicher Motors", "Auto"),
    Stock("HEROMOTOCO", "Hero MotoCorp", "Auto"),
    Stock("HDFCLIFE", "HDFC Life Insurance", "Financial Services"),
    Stock("SBILIFE", "SBI Life Insurance", "Financial Services"),
    Stock("INDUSINDBK", "IndusInd Bank", "Bank"),
    Stock("TECHM", "Tech Mahindra", "IT"),
    Stock("APOLLOHOSP", "Apollo Hospitals", "Healthcare"),
    Stock("TATACONSUM", "Tata Consumer Products", "FMCG"),
    Stock("BPCL", "Bharat Petroleum", "Oil & Gas"),
    Stock("LTIM", "LTIMindtree", "IT"),
    Stock("SHRIRAMFIN", "Shriram Finance", "Financial Services"),
    Stock("TRENT", "Trent", "Consumer"),
]

DEFAULT_UNIVERSE: list[Stock] = NIFTY50

_BY_SYMBOL: dict[str, Stock] = {s.symbol: s for s in NIFTY50}


def get(symbol: str) -> Stock | None:
    """Look up a stock by NSE symbol (case-insensitive)."""
    return _BY_SYMBOL.get(symbol.strip().upper())


def symbols() -> list[str]:
    return [s.symbol for s in DEFAULT_UNIVERSE]


def sectors() -> list[str]:
    return sorted({s.sector for s in DEFAULT_UNIVERSE})


def by_sector(sector: str) -> list[Stock]:
    return [s for s in DEFAULT_UNIVERSE if s.sector.lower() == sector.lower()]


def peers(symbol: str) -> list[Stock]:
    """Sector peers of a stock (excluding the stock itself)."""
    stock = get(symbol)
    if stock is None:
        return []
    return [s for s in by_sector(stock.sector) if s.symbol != stock.symbol]
