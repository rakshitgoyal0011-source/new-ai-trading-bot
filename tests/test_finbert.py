"""Tests for the FinBERT scorer + SentimentEngine model selection.

We never download the real FinBERT model in tests - everything is
monkey-patched so the suite stays offline + sub-second.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from analysis.sentiment import finbert as fb
from analysis.sentiment.engine import SentimentEngine


@pytest.fixture(autouse=True)
def _reset_module():
    fb._reset_for_tests()
    yield
    fb._reset_for_tests()


def test_scorer_returns_zero_when_transformers_missing(monkeypatch):
    # Hide transformers so the lazy import fails
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert fb.finbert_scorer("Company beats estimates") == 0.0
    assert fb.is_available() is False


def test_scorer_returns_zero_on_empty_text():
    assert fb.finbert_scorer("") == 0.0


def _install_mock_finbert(monkeypatch, pos: float, neg: float, neu: float):
    """Set up a mock transformers + torch so finbert_scorer is exercised."""
    import torch  # may or may not exist; if not, skip
    mock_logits = torch.tensor([[float(pos), float(neg), float(neu)]])

    mock_outputs = MagicMock()
    mock_outputs.logits = mock_logits

    mock_model = MagicMock()
    mock_model.return_value = mock_outputs
    mock_model.eval = MagicMock(return_value=mock_model)

    mock_tokenizer = MagicMock()
    # the tokenizer is called with kwargs; return a dict that ** unpacks
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    mock_module = MagicMock()
    mock_module.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock_module.AutoModelForSequenceClassification.from_pretrained.return_value = mock_model
    monkeypatch.setitem(sys.modules, "transformers", mock_module)


def test_scorer_with_mocked_transformers_positive(monkeypatch):
    torch = pytest.importorskip("torch")
    _install_mock_finbert(monkeypatch, pos=5.0, neg=0.0, neu=0.0)
    score = fb.finbert_scorer("Company beats estimates with strong profit")
    assert score > 0.9  # softmax(5,0,0)[0] - softmax(5,0,0)[1] ~ 0.99 - 0.007


def test_scorer_with_mocked_transformers_negative(monkeypatch):
    torch = pytest.importorskip("torch")
    _install_mock_finbert(monkeypatch, pos=0.0, neg=5.0, neu=0.0)
    score = fb.finbert_scorer("Company misses guidance and downgrades outlook")
    assert score < -0.9


def test_sentiment_engine_picks_finbert_when_configured():
    calls = []

    def fake_scorer(text):
        calls.append(text)
        return 0.8

    engine = SentimentEngine(model="finbert", scorer=fake_scorer)
    assert engine.model == "finbert"
    assert engine.scorer is fake_scorer


def test_sentiment_engine_defaults_to_lexicon():
    from analysis.sentiment.engine import lexicon_score
    engine = SentimentEngine()
    assert engine.scorer is lexicon_score
