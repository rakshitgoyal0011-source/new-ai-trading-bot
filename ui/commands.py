"""Command router for the DALAL TERMINAL command bar.

Parses Bloomberg-style commands (`<TICKER> TA`, `TOP`, `SCAN`, ...) and
returns a structured response of render-able blocks that the web
frontend paints into the terminal. Engines stay UI-agnostic; this module
is the only place that knows about command syntax.
"""
from __future__ import annotations

import time

from analysis.composite.engine import DISCLAIMER, CompositeEngine
from analysis.technical import TechnicalEngine
from config import universe
from config.weights import SignalWeights
from data.history import HistoryService
from data.quotes import QuoteService
from monitoring.logging import get_logger
from risk.risk import build_trade_plan, risk_rules
from screener import screener

log = get_logger("ui.commands")

_LEADERBOARD_TTL = 30.0  # seconds


# --- block helpers ---------------------------------------------------------
def _text(s: str) -> dict:
    return {"type": "text", "text": s}


def _note(s: str) -> dict:
    return {"type": "note", "text": s}


def _keyval(pairs: list[tuple[str, str]]) -> dict:
    return {"type": "keyval", "pairs": [[k, str(v)] for k, v in pairs]}


def _table(headers: list[str], rows: list[list]) -> dict:
    return {"type": "table", "headers": headers,
            "rows": [[str(c) for c in r] for r in rows]}


def _bars(items: list[tuple[str, float]]) -> dict:
    return {"type": "bars", "items": [[k, round(v, 1)] for k, v in items]}


def _spark(data: list[float], label: str = "") -> dict:
    return {"type": "spark", "data": [round(float(x), 2) for x in data],
            "label": label}


def _disclaimer() -> dict:
    return {"type": "disclaimer", "text": DISCLAIMER}


def _ok(title: str, blocks: list[dict], subtitle: str = "") -> dict:
    return {"ok": True, "title": title, "subtitle": subtitle, "blocks": blocks}


def _err(msg: str) -> dict:
    return {"ok": False, "title": "ERROR", "subtitle": "",
            "blocks": [_note(msg), _text("type HELP for the command reference")]}


class CommandRouter:
    """Holds the data services + engines and dispatches command strings."""

    def __init__(self, settings):
        self.settings = settings
        self.history = HistoryService(settings)
        self.quotes = QuoteService(settings)
        self.tech = TechnicalEngine()
        self.composite = CompositeEngine(SignalWeights.from_settings(settings))
        self.watchlist = universe.symbols()[:12]
        self.quotes.track(universe.symbols())
        self._lb_cache: list[screener.LeaderboardEntry] | None = None
        self._lb_ts = 0.0

    # --- dispatch ----------------------------------------------------------
    def dispatch(self, raw: str) -> dict:
        parts = raw.strip().split()
        if not parts:
            return _err("empty command")
        head = parts[0].upper()
        try:
            if head in ("HELP", "?"):
                return self._help()
            if head == "RULES":
                return self._rules()
            if head == "TOP":
                return self._top(parts[1:])
            if head == "SCAN":
                return self._scan(parts[1:])
            if head == "WATCH":
                return self._watch(parts[1:])
            if len(parts) >= 2:
                return self._ticker(parts[0].upper(), parts[1].upper(), parts[2:])
            return _err(f"unrecognised command: {raw!r}")
        except Exception as exc:  # never let the terminal crash on bad input
            log.exception("command failed: %s", raw)
            return _err(f"command failed: {exc}")

    # --- ticker commands ---------------------------------------------------
    def _ticker(self, symbol: str, verb: str, args: list[str]) -> dict:
        if verb in ("TA", "GP", "FA", "N"):
            stock = universe.get(symbol)
            name = stock.name if stock else symbol
            if verb == "TA":
                return self._technical(symbol, name)
            if verb == "GP":
                return self._chart(symbol, name)
            if verb == "FA":
                return self._fundamental(symbol, name)
            if verb == "N":
                return self._news(symbol, name)
        return _err(f"unknown action {verb!r} - try TA / GP / FA / N")

    def _technical(self, symbol: str, name: str) -> dict:
        df = self.history.candles(symbol, "day", 260)
        higher = df.iloc[::5]  # crude weekly view for MTF confirmation
        res = self.tech.analyze(symbol, df, higher_tf=higher)

        blocks: list[dict] = [
            _keyval([
                ("TECHNICAL SCORE", f"{res.score} / 100"),
                ("BIAS", res.bias.upper()),
                ("TREND", res.trend),
                ("CONFIDENCE", f"{res.confidence:.0%}"),
                ("MTF CONFIRMED", "yes" if res.mtf_confirmed else "no"),
            ]),
        ]
        if res.sub_scores:
            blocks.append(_bars(list(res.sub_scores.items())))
        if res.reasons:
            blocks.append(_table(["why this score"], [[r] for r in res.reasons]))
        if res.patterns:
            blocks.append(_text("patterns: " + ", ".join(
                p.replace("_", " ") for p in res.patterns)))

        ind = res.indicators
        blocks.append(_keyval([
            ("Close", ind.get("close", 0)),
            ("RSI(14)", ind.get("rsi", 0)),
            ("MACD", ind.get("macd", 0)),
            ("ADX(14)", ind.get("adx", 0)),
            ("ATR(14)", ind.get("atr", 0)),
            ("EMA20", ind.get("ema20", 0)),
            ("EMA50", ind.get("ema50", 0)),
            ("EMA200", ind.get("ema200", 0)),
            ("Supertrend", "bull" if ind.get("supertrend_dir", 0) > 0 else "bear"),
            ("BB %B", ind.get("bb_pct_b", 0)),
            ("MFI(14)", ind.get("mfi", 0)),
            ("VWAP", ind.get("vwap", 0)),
            ("Vol vs avg", ind.get("volume_ratio", 0)),
        ]))

        plan = build_trade_plan(
            symbol, ind.get("close", 0.0), ind.get("atr", 0.0),
            self.settings.default_capital,
            risk_pct=self.settings.risk_per_trade_pct,
        )
        blocks.append(_note(
            f"RISK FRAME (capital Rs.{self.settings.default_capital:,.0f}, "
            f"{plan.risk_pct}% risk): entry {plan.entry}  stop {plan.stop}  "
            f"targets {plan.targets}  R:R {plan.reward_risk}  size {plan.shares} sh"))
        blocks.append(_disclaimer())
        return _ok(f"{symbol} - TECHNICAL ANALYSIS", blocks, subtitle=name)

    def _chart(self, symbol: str, name: str) -> dict:
        df = self.history.candles(symbol, "day", 180)
        closes = df["close"].tolist()
        last = df.iloc[-1]
        from analysis.technical import indicators as ind

        e20 = ind.ema(df["close"], 20).iloc[-1]
        e50 = ind.ema(df["close"], 50).iloc[-1]
        rsi = ind.rsi(df["close"]).iloc[-1]
        blocks = [
            _spark(closes[-90:], f"{symbol} close - last 90 bars"),
            _keyval([
                ("Last", round(float(last["close"]), 2)),
                ("Open", round(float(last["open"]), 2)),
                ("High", round(float(last["high"]), 2)),
                ("Low", round(float(last["low"]), 2)),
                ("52-ish High", round(float(df["high"].tail(252).max()), 2)),
                ("52-ish Low", round(float(df["low"].tail(252).min()), 2)),
                ("EMA20", round(float(e20), 2)),
                ("EMA50", round(float(e50), 2)),
                ("RSI(14)", round(float(rsi), 1)),
            ]),
            _spark(ind.rsi(df["close"]).dropna().tolist()[-90:], "RSI(14)"),
        ]
        return _ok(f"{symbol} - PRICE & INDICATORS", blocks, subtitle=name)

    def _fundamental(self, symbol: str, name: str) -> dict:
        return _ok(f"{symbol} - FUNDAMENTAL ANALYSIS", [
            _note("Fundamental scoring is delivered in Milestone 4."),
            _text("Will show: PE / PB / PEG vs sector, ROE / ROCE, margins,"),
            _text("revenue & EPS growth, debt/equity, promoter pledge,"),
            _text("a sector-relative FUNDAMENTAL SCORE (0-100) and reasons."),
        ], subtitle=name)

    def _news(self, symbol: str, name: str) -> dict:
        return _ok(f"{symbol} - NEWS & SENTIMENT", [
            _note("News & sentiment scoring is delivered in Milestone 5."),
            _text("Will show: recent headlines + corporate announcements,"),
            _text("per-item sentiment (finance lexicon / FinBERT), material"),
            _text("event flags and a NEWS/SENTIMENT SCORE (0-100)."),
        ], subtitle=name)

    # --- leaderboard / screens --------------------------------------------
    def _leaderboard(self) -> list[screener.LeaderboardEntry]:
        now = time.monotonic()
        if self._lb_cache is not None and now - self._lb_ts < _LEADERBOARD_TTL:
            return self._lb_cache
        entries: list[screener.LeaderboardEntry] = []
        for stock in universe.DEFAULT_UNIVERSE:
            try:
                df = self.history.candles(stock.symbol, "day", 260)
                res = self.tech.analyze(stock.symbol, df)
                comp = self.composite.combine(stock.symbol, res.score)
                quote = self.quotes.snapshot(stock.symbol)
                entries.append(screener.LeaderboardEntry(
                    symbol=stock.symbol, name=stock.name, sector=stock.sector,
                    composite_score=comp.composite_score,
                    technical_score=res.score, bias=comp.bias, trend=res.trend,
                    ltp=quote.ltp if quote else res.indicators.get("close", 0.0),
                    change_pct=quote.change_pct if quote else 0.0,
                    atr=res.indicators.get("atr", 0.0),
                    top_reason=res.reasons[0] if res.reasons else "",
                    metrics={"rsi": res.indicators.get("rsi", 50.0),
                             "volume_ratio": res.indicators.get("volume_ratio", 1.0)},
                ))
            except Exception as exc:
                log.warning("leaderboard skip %s: %s", stock.symbol, exc)
        self._lb_cache, self._lb_ts = entries, now
        return entries

    def _top(self, args: list[str]) -> dict:
        sector = None
        budget = None
        max_results = 15
        for a in args:
            up = a.upper()
            if up.startswith("SECTOR="):
                sector = a.split("=", 1)[1]
            elif up.startswith("BUDGET="):
                budget = float(a.split("=", 1)[1])
            elif up.isdigit():
                max_results = int(up)
            elif universe.get(up) is None and up.title() in universe.sectors():
                sector = up.title()
        rows = screener.rank(self._leaderboard(), sector=sector,
                             max_results=max_results)
        if not rows:
            return _err(f"no stocks matched (sector={sector})")

        table = _table(
            ["#", "SYM", "SECTOR", "COMPOSITE", "BIAS", "TREND", "LTP", "CHG%"],
            [[i + 1, e.symbol, e.sector, e.composite_score, e.bias,
              e.trend, round(e.ltp, 2), round(e.change_pct, 2)]
             for i, e in enumerate(rows)],
        )
        blocks = [
            _note("Composite = technical only for now; fundamental & news "
                  "weights activate in Milestone 4 / 5."),
            table,
        ]
        if budget:
            blocks.append(_text(f"--- budget plan for Rs.{budget:,.0f} "
                                f"({self.settings.risk_per_trade_pct}% risk/trade) ---"))
            picks = screener.budget_picks(
                rows, budget, risk_pct=self.settings.risk_per_trade_pct)
            blocks.append(_table(
                ["SYM", "ENTRY", "STOP", "T1/T2/T3", "R:R", "SHARES", "OUTLAY"],
                [[e.symbol, p.entry, p.stop,
                  "/".join(str(t) for t in p.targets), p.reward_risk,
                  p.shares, p.position_value] for e, p in picks]))
        blocks.append(_disclaimer())
        return _ok("TOP - BUY-PROBABILITY LEADERBOARD", blocks,
                   subtitle=f"{len(rows)} stocks"
                            + (f" | sector {sector}" if sector else ""))

    def _scan(self, args: list[str]) -> dict:
        if not args:
            rows = [[s.key, s.label, "ready" if s.ready else "pending",
                     s.description] for s in screener.SCANS]
            return _ok("SCAN - PREBUILT SCREENERS", [
                _table(["KEY", "NAME", "STATUS", "WHAT IT FINDS"], rows),
                _text("run one with:  SCAN <key>   e.g.  SCAN momentum"),
            ])
        key = args[0].lower()
        scan = next((s for s in screener.SCANS if s.key == key), None)
        if scan is None:
            return _err(f"unknown scan {key!r}")
        if not scan.ready:
            return _ok(f"SCAN - {scan.label.upper()}",
                       [_note(f"'{scan.label}' {scan.description}.")])
        hits = screener.run_scan(key, self._leaderboard())
        hits.sort(key=lambda e: e.composite_score, reverse=True)
        if not hits:
            return _ok(f"SCAN - {scan.label.upper()}",
                       [_text("no stocks currently match this scan")])
        return _ok(f"SCAN - {scan.label.upper()}", [
            _text(scan.description),
            _table(["SYM", "SECTOR", "SCORE", "TREND", "REASON"],
                   [[e.symbol, e.sector, e.composite_score, e.trend,
                     e.top_reason] for e in hits]),
            _disclaimer(),
        ])

    def _watch(self, args: list[str]) -> dict:
        if args:
            syms = [s.strip().upper() for s in " ".join(args).replace(",", " ").split()]
            self.watchlist = [s for s in syms if s]
        rows = []
        for sym in self.watchlist:
            q = self.quotes.snapshot(sym)
            if q is None:
                continue
            arrow = "^" if q.direction > 0 else "v" if q.direction < 0 else "-"
            rows.append([sym, round(q.ltp, 2), round(q.change, 2),
                         f"{q.change_pct:+.2f}%", arrow])
        return _ok("WATCH - LIVE WATCHLIST", [
            _table(["SYM", "LTP", "CHG", "CHG%", ""], rows),
            _text("set your own:  WATCH RELIANCE,TCS,INFY"),
        ], subtitle=f"{len(rows)} symbols")

    # --- static ------------------------------------------------------------
    def _help(self) -> dict:
        return _ok("DALAL TERMINAL - COMMAND REFERENCE", [
            _table(["COMMAND", "DESCRIPTION"], [
                ["<TICKER> TA", "technical signal breakdown + 0-100 score"],
                ["<TICKER> GP", "price chart (sparkline) + key indicators"],
                ["<TICKER> FA", "fundamental snapshot (Milestone 4)"],
                ["<TICKER> N", "latest news + sentiment (Milestone 5)"],
                ["TOP [n] [SECTOR=x] [BUDGET=n]", "ranked buy-probability board"],
                ["SCAN [key]", "list / run prebuilt screeners"],
                ["WATCH [a,b,c]", "live watchlist tape"],
                ["RULES", "risk & discipline checklist"],
                ["HELP", "this reference"],
            ]),
            _text("examples:  RELIANCE TA   |   TOP 10 SECTOR=IT   |   "
                  "TOP BUDGET=100000   |   SCAN momentum"),
            _note("Analytics & research only - this terminal never places orders."),
        ])

    def _rules(self) -> dict:
        return _ok("RULES - RISK & DISCIPLINE", [
            _table(["#", "RULE"],
                   [[i + 1, r] for i, r in enumerate(risk_rules())]),
            _disclaimer(),
        ])
