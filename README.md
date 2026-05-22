# DALAL TERMINAL

A Bloomberg-style web terminal for the **Indian stock market (NSE / BSE)**.
Streams live prices, ranks stocks by a composite buy-probability score
combining **technical + fundamental + news/sentiment** analysis, and
frames every suggestion with an ATR-based stop, target and budget-aware
position size.

> ## ⚠ NOT FINANCIAL ADVICE
> DALAL TERMINAL is an **analytics and research** tool. It surfaces
> probability **estimates** derived from historical data for a human to
> consider. It is **not** a guarantee of future returns, it is **not**
> investment advice, and it **never places orders**. Trade only with
> capital you can afford to lose, and do your own research.

---

## Status — milestone tracker

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Scaffold + technical engine + web terminal (demo mode) end-to-end | ✅ shipped |
| 2 | Kite auth + live quotes + ticker tape (live mode wiring) | code-complete, awaiting your Kite keys |
| 3 | TUI command bar — Bloomberg-style multi-panel polish | partially shipped (web command bar + tape + watchlist + movers) |
| 4 | Fundamental engine + provider (yfinance baseline) | interface frozen, scoring pending |
| 5 | News & sentiment engine (RSS + lexicon, FinBERT optional) | interface frozen, scoring pending |
| 6 | Backtest + walk-forward validation + calibration curve | cost model done, loop pending |
| 7 | Screeners + budget-sizing UX | ranking + budget pairing done; richer scans pending |

Milestone 1 ships with **71 passing tests** and a runnable web terminal
that works with zero API keys.

---

## Quick start (demo mode, no keys needed)

```bash
git clone <this repo>
cd new-ai-trading-bot

python -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt

python main.py
# open http://127.0.0.1:8000
```

The terminal boots with synthetic but deterministic prices for the
NIFTY 50, so every command and the live ticker tape work out of the
box. Type `HELP` once it loads.

Run the test suite:

```bash
pytest -q
```

---

## Going live — Kite Connect setup

Live data uses **Zerodha Kite Connect v3** (the official Python SDK is
`kiteconnect`, currently 5.2.0).

### 1. Create the Kite app

1. Sign up at <https://developers.kite.trade> (≈ ₹500 / month per app).
2. Create an app. Note the **API Key** and **API Secret**.
3. Set the redirect URL to `http://127.0.0.1:8000/kite/callback`
   (or whatever you intend to use).
4. **Historical candles** (needed for indicators + backtests) require
   the separate **Historical Data add-on** (≈ ₹2,000 / month — verify
   current pricing at the developer portal).

### 2. Configure `.env`

```bash
cp .env.example .env
# edit .env and fill in:
#   DALAL_MODE=live
#   KITE_API_KEY=...
#   KITE_API_SECRET=...
```

### 3. Daily login routine (Kite tokens expire ~07:30 IST every morning)

```bash
python main.py --login
# 1. Open the printed URL, log in to Zerodha
# 2. Copy the `request_token` from the redirect URL
# 3. Paste it back into the prompt
# -> access_token is saved to data_cache/kite_token.json for the day
```

Then start the terminal normally with `python main.py`.

---

## Configuration checklist

| Setting | Where | Required for |
|---------|-------|--------------|
| `KITE_API_KEY` / `KITE_API_SECRET` | `.env` | live mode |
| `KITE_REDIRECT_URL` | `.env` | live mode (must match the Kite app) |
| `KITE_TOTP_SECRET` | `.env` | optional, semi-automates the daily login |
| `DALAL_MODE` | `.env` | `demo` (default) or `live` |
| `DALAL_HOST` / `DALAL_PORT` | `.env` | web server bind |
| `FUNDAMENTALS_PROVIDER` | `.env` | Milestone 4 (`yfinance` default) |
| `TWELVEDATA_API_KEY` / `FMP_API_KEY` | `.env` | Milestone 4, paid providers |
| `NEWS_PROVIDER` | `.env` | Milestone 5 (`rss` default) |
| `NEWSAPI_KEY` / `FINNHUB_API_KEY` | `.env` | Milestone 5, paid providers |
| `SENTIMENT_MODEL` | `.env` | Milestone 5 (`lexicon` default, `finbert` opt-in) |
| `WEIGHT_TECHNICAL/FUNDAMENTAL/NEWS` | `.env` | composite blend (auto-normalised) |
| `RISK_PER_TRADE_PCT` | `.env` | default risk per suggested trade |
| `DEFAULT_CAPITAL` | `.env` | default budget for sizing |

---

## Data sources

| Layer | Source | Cost | Notes |
|-------|--------|------|-------|
| Live quotes + websocket | Zerodha Kite Connect | ₹500/mo | included in the Kite app |
| Historical OHLCV | Zerodha Kite Historical | ~₹2,000/mo | add-on |
| Fundamentals (M4) | `yfinance` (`.NS`) | free | baseline; provider is pluggable |
| Fundamentals (M4 paid) | TwelveData / FMP / EODHD | paid | structured NSE financials |
| News & announcements (M5) | NSE / BSE RSS, Moneycontrol, ET RSS | free | needs headers + retry/backoff |
| News (M5 paid) | NewsAPI / Finnhub / stockinsights.ai | paid | richer coverage / pre-tagged |
| Sentiment (M5) | Finance lexicon (default) → FinBERT (opt-in) | free | model is swappable |

---

## Command reference

| Command | What it does |
|---------|--------------|
| `<TICKER> TA` | Technical breakdown: 0-100 score, bias, trend, sub-scores, reasons, full indicator snapshot, ATR-based trade frame |
| `<TICKER> GP` | Price chart (sparkline) + key indicators |
| `<TICKER> FA` | Fundamental snapshot (Milestone 4) |
| `<TICKER> N` | Latest news + sentiment (Milestone 5) |
| `TOP [n] [SECTOR=x] [BUDGET=n]` | Ranked buy-probability leaderboard with optional sector filter and budget-aware sizing |
| `SCAN` / `SCAN <key>` | List or run prebuilt screeners (`momentum`, `oversold`, `volume_spike`, `strong_bull` ready; fundamental scans land in M4/M5) |
| `WATCH a,b,c` | Set the live watchlist tape |
| `RULES` | Risk-discipline checklist |
| `HELP` | Command reference |

Examples: `RELIANCE TA`, `TOP 10 SECTOR=IT`, `TOP BUDGET=100000`, `SCAN momentum`, `WATCH RELIANCE,TCS,INFY`.

---

## Architecture

```
config/        run mode, weights, NIFTY 50 universe with sectors
core/          Kite Connect client, daily-login routine, instrument master
data/          live quotes, historical fetch + SQLite cache, candle builder,
               demo market, fundamental & news provider interfaces
analysis/
  technical/   indicators + candlestick patterns + chart patterns + gaps
               + scored TechnicalEngine (centerpiece)
  fundamental/ FundamentalEngine (M4)
  sentiment/   SentimentEngine    (M5)
  composite/   CompositeEngine    - weighted blend; calibration M6
screener/      ranking + prebuilt scans + budget-aware picks (TOP)
risk/          ATR stops, R:R, position sizing, RULES
backtest/      realistic Indian-market cost model; walk-forward loop M6
ui/            FastAPI server + websocket tape + command router
  web/         dark monospace terminal frontend (index.html + css + js)
monitoring/    centralised logging
tests/         71 tests covering indicators, candlesticks, technical engine,
               composite, risk, screener, candles, store, demo, backtest,
               and the command router end-to-end
```

The data and analysis layers are UI-agnostic, so a TUI or richer
candlestick-chart frontend can be added later without touching engines.

---

## Honesty contract on the buy-probability

The composite score is **not** a probability yet. Turning it into a
calibrated *probability of a positive return over horizon H* needs a
model trained and backtested on historical Kite data, then calibrated
(Platt scaling / isotonic regression) with a calibration curve. That
work is **Milestone 6**. Until then:

- `CompositeResult.calibrated_probability` stays `None`
- the UI reports the **uncalibrated composite score**, never a fake probability
- every analytical response carries the **NOT FINANCIAL ADVICE** disclaimer

This is by design — surfacing a fake probability is worse than no probability.

---

## License & scope

For personal research use. No automated order placement is implemented
or planned: this is a decision-support terminal for a human operator.
