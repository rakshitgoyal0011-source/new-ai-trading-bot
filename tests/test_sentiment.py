"""Tests for the news / sentiment engine."""
from datetime import datetime, timedelta

from analysis.sentiment.engine import (
    SentimentEngine,
    is_material,
    lexicon_score,
)
from data.news import NewsItem


def test_lexicon_strongly_positive_headline():
    assert lexicon_score("Company beats estimates with strong profit surge") > 0.5


def test_lexicon_strongly_negative_headline():
    s = lexicon_score("Company missed guidance amid weak losses and downgrade")
    assert s < -0.5


def test_lexicon_no_finance_words_is_zero():
    assert lexicon_score("CEO held a press meeting in Mumbai") == 0.0


def test_material_event_detection():
    assert is_material("Q3 earnings beat the street")
    assert not is_material("CEO met with press in Mumbai")


def test_engine_no_items_is_neutral():
    res = SentimentEngine().analyze("X", [])
    assert res.score == 50.0
    assert res.bias == "neutral"


def test_engine_positive_news_is_bullish():
    now = datetime.now()
    items = [
        NewsItem("X", "Company beats estimates with strong profit growth",
                 "wire", published=now),
        NewsItem("X", "Stock surges on rating upgrade", "wire",
                 published=now - timedelta(days=1)),
    ]
    res = SentimentEngine().analyze("X", items)
    assert res.score >= 60.0
    assert res.bias == "bullish"
    assert res.headline_count == 2


def test_engine_negative_news_is_bearish():
    now = datetime.now()
    items = [
        NewsItem("X", "Company missed guidance amid weak demand and downgrade",
                 "wire", published=now),
        NewsItem("X", "CEO resigned, shares fell sharply", "wire",
                 published=now),
    ]
    res = SentimentEngine().analyze("X", items)
    assert res.score <= 40.0
    assert res.bias == "bearish"


def test_recency_weighting_recent_news_dominates():
    now = datetime.now()
    # an old positive item should be outweighed by a recent negative one
    items = [
        NewsItem("X", "Strong profit growth", "wire",
                 published=now - timedelta(days=30)),
        NewsItem("X", "Weak guidance and downgrade", "wire", published=now),
    ]
    res = SentimentEngine().analyze("X", items)
    assert res.score < 50.0


def test_material_events_listed():
    now = datetime.now()
    items = [
        NewsItem("X", "Q3 earnings beat estimates", "wire", published=now),
    ]
    res = SentimentEngine().analyze("X", items)
    assert any("earnings" in m.lower() for m in res.material_events)
