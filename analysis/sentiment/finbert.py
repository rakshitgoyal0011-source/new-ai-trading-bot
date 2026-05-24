"""FinBERT sentiment scorer (ProsusAI/finbert).

Lazy-loads the model on first call. SENTIMENT_MODEL=finbert in .env
turns it on; the lexicon stays the default so the demo experience and
the test suite don't drag in transformers + torch (~ 400MB).

`finbert_scorer(text)` returns a value in [-1, +1] (positive minus
negative probability). If the model cannot be loaded (transformers /
torch not installed, network blocked, disk full) the function returns
0.0 and the SentimentEngine smoothly falls back to a neutral signal -
the warning is logged once.
"""
from __future__ import annotations

from monitoring.logging import get_logger

log = get_logger("analysis.sentiment.finbert")

_MODEL = None
_TOKENIZER = None
_LOAD_FAILED = False
_MODEL_NAME = "ProsusAI/finbert"
# FinBERT label indices: 0 = positive, 1 = negative, 2 = neutral
_POS_IDX, _NEG_IDX = 0, 1


def _ensure_loaded() -> None:
    """Load the model + tokenizer once. Sets _LOAD_FAILED on any error."""
    global _MODEL, _TOKENIZER, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return
    try:
        from transformers import (
            AutoModelForSequenceClassification, AutoTokenizer,
        )
        _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        _MODEL.eval()
        log.info("FinBERT model loaded (%s)", _MODEL_NAME)
    except Exception as exc:
        _LOAD_FAILED = True
        log.warning(
            "FinBERT unavailable (%s: %s) - scoring will return 0.0 "
            "and SentimentEngine will fall back to neutral",
            type(exc).__name__, exc,
        )


def finbert_scorer(text: str) -> float:
    """Return a -1..+1 sentiment score for `text` using FinBERT.

    Returns 0.0 if the model couldn't load, so callers can treat the
    output uniformly regardless of which backend ran.
    """
    if not text:
        return 0.0
    _ensure_loaded()
    if _LOAD_FAILED or _MODEL is None or _TOKENIZER is None:
        return 0.0
    try:
        import torch

        inputs = _TOKENIZER(
            text, return_tensors="pt",
            truncation=True, max_length=128, padding=True,
        )
        with torch.no_grad():
            logits = _MODEL(**inputs).logits[0]
        probs = torch.softmax(logits, dim=0)
        return float(probs[_POS_IDX]) - float(probs[_NEG_IDX])
    except Exception as exc:
        log.warning("FinBERT inference failed: %s", exc)
        return 0.0


def is_available() -> bool:
    """True if FinBERT scored a real prediction (model loaded OK)."""
    _ensure_loaded()
    return not _LOAD_FAILED and _MODEL is not None


def _reset_for_tests() -> None:
    """Reset the module-level cache (used by the test suite)."""
    global _MODEL, _TOKENIZER, _LOAD_FAILED
    _MODEL = None
    _TOKENIZER = None
    _LOAD_FAILED = False
