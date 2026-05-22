"""Tests for the demo news provider (RSS path is exercised via integration)."""
from data.news import DemoNews, NewsItem


def test_demo_news_returns_items_with_metadata():
    items = DemoNews().fetch("RELIANCE", limit=5)
    assert 1 <= len(items) <= 5
    for item in items:
        assert isinstance(item, NewsItem)
        assert item.symbol == "RELIANCE"
        assert item.headline
        assert item.source == "demo wire"
        assert item.published is not None


def test_demo_news_is_deterministic():
    a = DemoNews().fetch("TCS", limit=5)
    b = DemoNews().fetch("TCS", limit=5)
    assert [i.headline for i in a] == [i.headline for i in b]


def test_demo_news_respects_limit():
    items = DemoNews().fetch("INFY", limit=2)
    assert len(items) <= 2
