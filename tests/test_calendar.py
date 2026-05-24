"""Tests for the corporate calendar providers."""
from __future__ import annotations

from datetime import datetime, timedelta

from data.calendar import CalendarEvent, DemoCalendar, EVENT_TYPES


def test_demo_calendar_returns_events_in_horizon():
    events = DemoCalendar().fetch("RELIANCE", horizon_days=30)
    assert 1 <= len(events) <= 3
    now = datetime.now()
    for ev in events:
        assert ev.symbol == "RELIANCE"
        assert ev.event_type in EVENT_TYPES
        assert now <= ev.date <= now + timedelta(days=31)
        assert ev.description


def test_demo_calendar_is_deterministic():
    a = DemoCalendar().fetch("TCS", horizon_days=30)
    b = DemoCalendar().fetch("TCS", horizon_days=30)
    assert [(e.event_type, e.description) for e in a] == [
        (e.event_type, e.description) for e in b
    ]


def test_demo_calendar_sorted_by_date():
    events = DemoCalendar().fetch("INFY", horizon_days=60)
    dates = [e.date for e in events]
    assert dates == sorted(dates)


def test_calendar_event_days_away_is_non_negative():
    ev = CalendarEvent(
        symbol="X",
        date=datetime.now() + timedelta(days=7),
        event_type="results",
        description="Q2 results",
    )
    assert ev.days_away >= 6


def test_calendar_event_to_dict_serialises_date():
    ev = CalendarEvent(
        symbol="X",
        date=datetime(2026, 6, 1, 9, 30),
        event_type="ex_dividend",
        description="ex-div Rs.5",
    )
    d = ev.to_dict()
    assert d["date"].startswith("2026-06-01")
    assert d["event_type"] == "ex_dividend"
    assert "days_away" in d


def test_demo_calendar_handles_small_horizon_without_hanging():
    """Regression: original `while days in used_days: days = (days+1) %
    (horizon+1) or 1` looped forever once horizon_days filled up."""
    events1 = DemoCalendar().fetch("RELIANCE", horizon_days=1)
    assert len(events1) <= 1
    events2 = DemoCalendar().fetch("TCS", horizon_days=2)
    assert len(events2) <= 2
    # also stress with horizon equal to max possible n
    events3 = DemoCalendar().fetch("INFY", horizon_days=3)
    assert len(events3) <= 3


def test_demo_calendar_zero_horizon_returns_empty():
    assert DemoCalendar().fetch("RELIANCE", horizon_days=0) == []
