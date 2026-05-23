"""Corporate calendar: upcoming earnings, ex-div, board meetings.

`CalendarProvider` is the interface; ships with:
- `DemoCalendar`      synthetic events, deterministic per symbol
- `YFinanceCalendar`  next earnings date via yfinance Ticker.calendar
                      (free, soft-fails when Yahoo throttles or omits a row)

The factory `get_provider(settings)` picks one. Live mode falls back to
demo when yfinance is unavailable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import numpy as np

from monitoring.logging import get_logger

log = get_logger("data.calendar")

EVENT_TYPES: tuple[str, ...] = (
    "results", "ex_dividend", "board_meeting", "agm", "split",
)


@dataclass
class CalendarEvent:
    symbol: str
    date: datetime
    event_type: str
    description: str
    source: str = ""

    @property
    def days_away(self) -> int:
        return max(0, (self.date.date() - datetime.now().date()).days)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        d["days_away"] = self.days_away
        return d


class CalendarProvider:
    name: str = "base"

    def fetch(self, symbol: str, horizon_days: int = 30) -> list[CalendarEvent]:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Demo (synthetic) calendar - deterministic per symbol.
# --------------------------------------------------------------------------
_DEMO_DESCRIPTIONS = {
    "results":       "Q{q} results, board meeting",
    "ex_dividend":   "ex-dividend, Rs.{d} per share",
    "board_meeting": "Board to consider quarterly update",
    "agm":           "Annual general meeting (AGM)",
    "split":         "Stock split, ratio {r}:1",
}


class DemoCalendar(CalendarProvider):
    name = "demo"

    def fetch(self, symbol: str, horizon_days: int = 30) -> list[CalendarEvent]:
        rng = np.random.default_rng(abs(hash("cal_" + symbol)) % (2**31))
        n = int(rng.integers(1, 4))
        now = datetime.now()
        events: list[CalendarEvent] = []
        used_days: set[int] = set()
        for _ in range(n):
            etype = EVENT_TYPES[rng.integers(0, len(EVENT_TYPES))]
            days = int(rng.integers(1, horizon_days + 1))
            while days in used_days:
                days = (days + 1) % (horizon_days + 1) or 1
            used_days.add(days)
            desc = _DEMO_DESCRIPTIONS[etype].format(
                q=int(rng.integers(1, 5)),
                d=round(float(rng.uniform(1, 25)), 1),
                r=int(rng.integers(2, 6)),
            )
            events.append(CalendarEvent(
                symbol=symbol, date=now + timedelta(days=days),
                event_type=etype, description=desc, source="demo",
            ))
        events.sort(key=lambda e: e.date)
        return events


# --------------------------------------------------------------------------
# yfinance - free upcoming earnings + recent dividends/splits.
# --------------------------------------------------------------------------
class YFinanceCalendar(CalendarProvider):
    name = "yfinance"

    def __init__(self, ttl_seconds: int = 6 * 3600):
        self._cache: dict[str, tuple[float, list[CalendarEvent]]] = {}
        self._ttl = ttl_seconds

    def fetch(self, symbol: str, horizon_days: int = 30) -> list[CalendarEvent]:
        import time

        cached = self._cache.get(symbol)
        if cached and time.time() - cached[0] < self._ttl:
            return [e for e in cached[1] if e.days_away <= horizon_days]

        events = self._fetch_uncached(symbol, horizon_days)
        self._cache[symbol] = (time.time(), events)
        return events

    def _fetch_uncached(
        self, symbol: str, horizon_days: int
    ) -> list[CalendarEvent]:
        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance not installed - YFinanceCalendar disabled")
            return []

        now = datetime.now()
        horizon_end = now + timedelta(days=horizon_days)
        events: list[CalendarEvent] = []
        try:
            ticker = yf.Ticker(symbol + ".NS")
            cal = ticker.calendar
        except Exception as exc:
            log.warning("yfinance calendar failed for %s: %s", symbol, exc)
            return []

        next_earnings = self._extract_earnings_date(cal)
        if next_earnings is not None and now <= next_earnings <= horizon_end:
            events.append(CalendarEvent(
                symbol=symbol, date=next_earnings, event_type="results",
                description="Next earnings (per yfinance)", source="yfinance",
            ))
        events.sort(key=lambda e: e.date)
        return events

    @staticmethod
    def _extract_earnings_date(cal) -> datetime | None:
        if cal is None:
            return None
        try:
            # newer yfinance returns a dict
            if isinstance(cal, dict):
                d = cal.get("Earnings Date") or cal.get("earningsDate")
                if isinstance(d, (list, tuple)) and d:
                    d = d[0]
                if d is None:
                    return None
                if hasattr(d, "to_pydatetime"):
                    d = d.to_pydatetime()
                if isinstance(d, datetime):
                    return d
            # older returns a DataFrame
            if hasattr(cal, "iloc") and "Earnings Date" in getattr(cal, "index", []):
                d = cal.loc["Earnings Date"].iloc[0]
                if hasattr(d, "to_pydatetime"):
                    d = d.to_pydatetime()
                return d if isinstance(d, datetime) else None
        except Exception:
            return None
        return None


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------
def get_provider(settings) -> CalendarProvider:
    if not settings.is_live:
        return DemoCalendar()
    return YFinanceCalendar()
