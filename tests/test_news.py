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


def test_build_needles_uses_two_words_for_conglomerate_prefix():
    """Regression: `name.split()[0]` alone produced needle 'tata' for
    TATASTEEL, which matched every Tata Motors / Tata Consumer headline."""
    from data.news import build_needles
    needles = build_needles("TATASTEEL", "Tata Steel")
    assert "tata steel" in needles
    assert "tata" not in needles


def test_build_needles_keeps_first_word_when_unique_in_universe():
    """Names whose first word doesn't collide with other tickers still
    use the single-word needle for natural-language matching."""
    from data.news import build_needles
    needles = build_needles("RELIANCE", "Reliance Industries")
    assert "reliance" in needles
    assert needles >= {"reliance", "reliance industries", "reliance"}


def test_build_needles_includes_symbol_and_full_name():
    from data.news import build_needles
    needles = build_needles("INFY", "Infosys")
    assert "infy" in needles
    assert "infosys" in needles
