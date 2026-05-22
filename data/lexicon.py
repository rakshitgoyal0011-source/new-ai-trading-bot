"""Finance sentiment lexicon - curated positive / negative / material keyword
sets used by the lexicon-based SentimentEngine.

Single-word tokens only; the engine matches them in lower-cased headlines
via word-boundary regex. Designed to be small enough to read in one sitting
and easy to extend. FinBERT can be swapped in via SENTIMENT_MODEL=finbert
once the heavy ML deps are installed.
"""
from __future__ import annotations

POSITIVE: frozenset[str] = frozenset({
    # earnings / growth
    "beat", "beats", "surpass", "surpassed", "exceeded", "exceeds",
    "outperform", "outperforms", "outpaced", "ahead",
    "growth", "rising", "rises", "surge", "surges", "rally", "rallies",
    "gain", "gains", "jump", "jumps", "soar", "soars", "expand", "expansion",
    "accelerate", "accelerates", "improved", "improving", "recovery", "upturn",
    "record", "highs",
    # corporate actions / positive events
    "upgrade", "upgrades", "upgraded", "raised", "raises", "boost", "boosted",
    "approval", "approved", "win", "wins", "awarded", "awards",
    "contract", "order", "orders", "tender",
    "milestone", "launch", "launches", "breakthrough",
    "dividend", "bonus", "buyback", "split",
    # qualitative
    "strong", "robust", "stellar", "blockbuster", "solid", "healthy",
    "positive", "optimistic", "bullish", "confident", "favourable",
    "profit", "profits", "profitable", "lucrative",
})

NEGATIVE: frozenset[str] = frozenset({
    # earnings / growth
    "miss", "missed", "misses", "decline", "declines", "declined",
    "fall", "falls", "fell", "drop", "drops", "plunge", "plunges",
    "slump", "slumps", "crash", "crashes", "loss", "losses",
    "weakness", "weak", "weaker", "slowdown", "sluggish", "downturn",
    "worse", "worsen", "worsened", "challenging", "headwind", "pressure",
    # corporate actions / negative events
    "downgrade", "downgrades", "downgraded", "cut", "cuts", "reduced",
    "reduces", "warning", "warns", "warned",
    "probe", "investigation", "investigated", "fraud", "scam", "scandal",
    "default", "defaulted", "bankruptcy", "insolvency",
    "delist", "delisted", "penalty", "penalised", "fined", "violation",
    "breach", "rejected", "blocked", "banned", "halted", "suspended",
    "resign", "resigned", "resignation", "exit", "exits",
    "strike", "layoff", "layoffs",
    # qualitative
    "disappoint", "disappointed", "disappointing", "negative", "bearish",
    "lows", "low", "concerns", "worried",
})

MATERIAL: frozenset[str] = frozenset({
    "results", "earnings", "quarterly", "q1", "q2", "q3", "q4",
    "ebitda", "revenue", "guidance",
    "order", "contract", "acquisition", "merger", "demerger",
    "stake", "ipo", "fpo", "rights", "preferential",
    "dividend", "bonus", "split", "buyback",
    "sebi", "rbi", "cci", "ncl", "ncdrc",
    "downgrade", "upgrade", "rating",
})
