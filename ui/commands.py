"""Command router for the DALAL TERMINAL command bar.

Parses Bloomberg-style commands (`<TICKER> TA`, `TOP`, `SCAN`, ...) and
returns a structured response of render-able blocks that the web
frontend paints into the terminal. Engines stay UI-agnostic; this module
is the only place that knows about command syntax.
"""
from __future__ import annotations

import time

from analysis.composite.engine import DISCLAIMER, CompositeEngine
from analysis.fundamental.engine import FundamentalEngine
from analysis.sentiment.engine import SentimentEngine
from analysis.technical import TechnicalEngine
from config import universe
from config.weights import SignalWeights
from data import calendar as calendar_mod
from data import fundamentals as fund_data_mod
from data import news as news_data_mod
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
        self.fund_provider = fund_data_mod.get_provider(settings)
        self.news_provider = news_data_mod.get_provider(settings)
        self.cal_provider = calendar_mod.get_provider(settings)
        self.fund_engine = FundamentalEngine()
        self.sent_engine = SentimentEngine(model=settings.sentiment_model)
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
            if head == "BT":
                return self._bt(parts[1:])
            if head == "HEAT":
                return self._heat()
            if head == "KITE":
                return self._kite()
            if head == "CAL":
                return self._calendar_all(parts[1:])
            if len(parts) >= 2:
                return self._ticker(parts[0].upper(), parts[1].upper(), parts[2:])
            return _err(f"unrecognised command: {raw!r}")
        except Exception as exc:  # never let the terminal crash on bad input
            log.exception("command failed: %s", raw)
            return _err(f"command failed: {exc}")

    # --- ticker commands ---------------------------------------------------
    def _ticker(self, symbol: str, verb: str, args: list[str]) -> dict:
        if verb in ("TA", "GP", "FA", "N", "CAL"):
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
            if verb == "CAL":
                return self._calendar_symbol(symbol, name, args)
        return _err(f"unknown action {verb!r} - try TA / GP / FA / N / CAL")

    def _score_symbol(self, symbol: str, technical_score: float):
        """Compute the live composite (tech + fund + sent) for one symbol.

        Used by ticker commands that want the calibrated probability
        alongside their main view. The leaderboard has its own batched
        path that pre-fetches fundamentals once.
        """
        fund_data = self.fund_provider.fetch(symbol)
        fund_score: float | None = None
        if fund_data.available:
            peers = [self.fund_provider.fetch(p.symbol)
                     for p in universe.peers(symbol)]
            fund_res = self.fund_engine.analyze(symbol, fund_data, peers)
            if fund_res.available:
                fund_score = fund_res.score
        items = self.news_provider.fetch(symbol, limit=5)
        sent_score = (
            self.sent_engine.analyze(symbol, items).score if items else None
        )
        return self.composite.combine(
            symbol, technical_score, fund_score, sent_score
        )

    def _technical(self, symbol: str, name: str) -> dict:
        df = self.history.candles(symbol, "day", 260)
        # crude weekly view for MTF confirmation - anchor at the latest
        # bar so the most recent week is never silently dropped
        offset = (len(df) - 1) % 5 if len(df) else 0
        higher = df.iloc[offset::5]
        res = self.tech.analyze(symbol, df, higher_tf=higher)
        comp = self._score_symbol(symbol, res.score)

        header_pairs = [
            ("TECHNICAL SCORE", f"{res.score} / 100"),
            ("COMPOSITE", f"{comp.composite_score} / 100"),
            ("BIAS", res.bias.upper()),
            ("TREND", res.trend),
            ("CONFIDENCE", f"{res.confidence:.0%}"),
            ("MTF CONFIRMED", "yes" if res.mtf_confirmed else "no"),
        ]
        if comp.calibrated_probability is not None:
            header_pairs.insert(2, (
                f"P(+) over {comp.horizon_days}d",
                f"{comp.calibrated_probability * 100:.0f}%",
            ))
        blocks: list[dict] = [_keyval(header_pairs)]
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
        recent = df.tail(60)
        candles = [
            [round(float(r.open), 2), round(float(r.high), 2),
             round(float(r.low), 2), round(float(r.close), 2)]
            for r in recent.itertuples()
        ]
        last = df.iloc[-1]
        from analysis.technical import indicators as ind

        e20 = ind.ema(df["close"], 20).iloc[-1]
        e50 = ind.ema(df["close"], 50).iloc[-1]
        rsi = ind.rsi(df["close"]).iloc[-1]
        blocks = [
            {"type": "candles", "ohlc": candles,
             "label": f"{symbol} - last 60 daily candles"},
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
            _spark(closes[-90:], "close - last 90 bars"),
            _spark(ind.rsi(df["close"]).dropna().tolist()[-90:], "RSI(14)"),
        ]
        return _ok(f"{symbol} - PRICE & INDICATORS", blocks, subtitle=name)

    def _fundamental(self, symbol: str, name: str) -> dict:
        data = self.fund_provider.fetch(symbol)
        if not data.available:
            return _ok(f"{symbol} - FUNDAMENTAL ANALYSIS", [
                _note("No fundamentals available from the current provider."),
                _text(f"provider: {self.fund_provider.name} | "
                      f"try a different FUNDAMENTALS_PROVIDER in .env"),
            ], subtitle=name)
        peers_fund = [self.fund_provider.fetch(p.symbol)
                      for p in universe.peers(symbol)]
        res = self.fund_engine.analyze(symbol, data, peers_fund)

        blocks: list[dict] = [
            _keyval([
                ("FUNDAMENTAL SCORE", f"{res.score} / 100"),
                ("BIAS", res.bias.upper()),
                ("PEERS COMPARED", res.peers_compared),
                ("PROVIDER", self.fund_provider.name),
            ]),
        ]
        if res.sub_scores:
            blocks.append(_bars(list(res.sub_scores.items())))

        pairs: list[tuple[str, str]] = []
        for label, val, fmt in [
            ("PE", data.pe, "{:.2f}"), ("PB", data.pb, "{:.2f}"),
            ("PEG", data.peg, "{:.2f}"), ("ROE %", data.roe, "{:.2f}"),
            ("ROCE %", data.roce, "{:.2f}"),
            ("Op margin %", data.operating_margin, "{:.2f}"),
            ("Net margin %", data.net_margin, "{:.2f}"),
            ("Rev growth %", data.revenue_growth, "{:.2f}"),
            ("EPS growth %", data.eps_growth, "{:.2f}"),
            ("D/E", data.debt_to_equity, "{:.2f}"),
            ("Div yld %", data.dividend_yield, "{:.2f}"),
            ("Promoter %", data.promoter_holding, "{:.1f}"),
            ("Pledge %", data.promoter_pledge, "{:.1f}"),
        ]:
            if val is not None:
                pairs.append((label, fmt.format(val)))
        if data.market_cap:
            pairs.append(("Mkt cap (Rs.B)", f"{data.market_cap / 1e9:.1f}"))
        if pairs:
            blocks.append(_keyval(pairs))

        if res.reasons:
            blocks.append(_table(["why this score"], [[r] for r in res.reasons]))
        if res.quality_flags:
            blocks.append(_text("flags: " + " | ".join(res.quality_flags)))
        blocks.append(_disclaimer())
        return _ok(
            f"{symbol} - FUNDAMENTAL ANALYSIS", blocks,
            subtitle=f"{name} | {data.sector or 'sector ?'}"
        )

    def _news(self, symbol: str, name: str) -> dict:
        items = self.news_provider.fetch(symbol, limit=10)
        res = self.sent_engine.analyze(symbol, items)

        blocks: list[dict] = [
            _keyval([
                ("SENTIMENT SCORE", f"{res.score} / 100"),
                ("BIAS", res.bias.upper()),
                ("HEADLINES", res.headline_count),
                ("MODEL", res.model),
                ("PROVIDER", self.news_provider.name),
            ]),
        ]
        if not items:
            blocks.append(_note(
                "No headlines found in the news window for this symbol."))
        else:
            rows = []
            for i in items[:12]:
                date = i.published.strftime("%b %d") if i.published else "?"
                s = i.sentiment or 0
                tone = "+" if s > 0 else "-" if s < 0 else " "
                rows.append([date, i.source, f"{tone} {i.headline}"])
            blocks.append(_table(["DATE", "SOURCE", "HEADLINE"], rows))
        if res.material_events:
            blocks.append(_text("material events:"))
            blocks.append(_table(["headline"],
                                  [[m] for m in res.material_events[:5]]))
        blocks.append(_disclaimer())
        return _ok(f"{symbol} - NEWS & SENTIMENT", blocks, subtitle=name)

    # --- leaderboard / screens --------------------------------------------
    def _leaderboard(self) -> list[screener.LeaderboardEntry]:
        now = time.monotonic()
        if self._lb_cache is not None and now - self._lb_ts < _LEADERBOARD_TTL:
            return self._lb_cache

        # pre-fetch fundamentals once so peer comparisons are cheap
        fund_lookup = {
            s.symbol: self.fund_provider.fetch(s.symbol)
            for s in universe.DEFAULT_UNIVERSE
        }
        entries: list[screener.LeaderboardEntry] = []
        for stock in universe.DEFAULT_UNIVERSE:
            try:
                df = self.history.candles(stock.symbol, "day", 260)
                res = self.tech.analyze(stock.symbol, df)

                peers_fund = [
                    fund_lookup[p.symbol]
                    for p in universe.peers(stock.symbol)
                    if fund_lookup.get(p.symbol) and fund_lookup[p.symbol].available
                ]
                fund_res = self.fund_engine.analyze(
                    stock.symbol, fund_lookup.get(stock.symbol), peers_fund)
                news_items = self.news_provider.fetch(stock.symbol, limit=5)
                sent_res = self.sent_engine.analyze(stock.symbol, news_items)

                comp = self.composite.combine(
                    stock.symbol, res.score,
                    fund_res.score if fund_res.available else None,
                    sent_res.score if news_items else None,
                )
                quote = self.quotes.snapshot(stock.symbol)
                ltp = quote.ltp if quote else float(res.indicators.get("close", 0.0))
                atr = float(res.indicators.get("atr", 0.0))
                atr_pct = (atr / ltp * 100.0) if ltp > 0 else 0.0
                high52 = float(df["high"].tail(252).max()) if len(df) else 0.0
                from_high = (ltp / high52 * 100.0) if high52 > 0 else 0.0
                gap_pct = 0.0
                if len(df) >= 2:
                    prev_close = float(df.iloc[-2]["close"])
                    today_open = float(df.iloc[-1]["open"])
                    if prev_close > 0:
                        gap_pct = (today_open - prev_close) / prev_close * 100.0
                entries.append(screener.LeaderboardEntry(
                    symbol=stock.symbol, name=stock.name, sector=stock.sector,
                    composite_score=comp.composite_score,
                    technical_score=res.score, bias=comp.bias, trend=res.trend,
                    ltp=ltp,
                    change_pct=quote.change_pct if quote else 0.0,
                    atr=atr,
                    top_reason=res.reasons[0] if res.reasons else "",
                    metrics={
                        "rsi": res.indicators.get("rsi", 50.0),
                        "volume_ratio": res.indicators.get("volume_ratio", 1.0),
                        "fundamental_score": (
                            fund_res.score if fund_res.available else 50.0),
                        "sentiment_score": (
                            sent_res.score if news_items else 50.0),
                        "gap_pct": round(gap_pct, 2),
                        "from_52w_high_pct": round(from_high, 2),
                        "atr_pct": round(atr_pct, 2),
                    },
                    calibrated_probability=comp.calibrated_probability,
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
            elif universe.get(up) is None:
                # case-insensitive sector match so `TOP IT` and `TOP FMCG`
                # don't get crushed by str.title() lowercasing acronyms.
                match = next(
                    (s for s in universe.sectors() if s.lower() == up.lower()),
                    None,
                )
                if match:
                    sector = match
        rows = screener.rank(self._leaderboard(), sector=sector,
                             max_results=max_results)
        if not rows:
            return _err(f"no stocks matched (sector={sector})")

        show_prob = any(e.calibrated_probability is not None for e in rows)
        if show_prob:
            headers = ["#", "SYM", "SECTOR", "COMPOSITE", "P(+)",
                       "BIAS", "TREND", "LTP", "CHG%"]
            body = [
                [i + 1, e.symbol, e.sector, e.composite_score,
                 (f"{e.calibrated_probability * 100:.0f}%"
                  if e.calibrated_probability is not None else "-"),
                 e.bias, e.trend, round(e.ltp, 2), round(e.change_pct, 2)]
                for i, e in enumerate(rows)
            ]
        else:
            headers = ["#", "SYM", "SECTOR", "COMPOSITE",
                       "BIAS", "TREND", "LTP", "CHG%"]
            body = [
                [i + 1, e.symbol, e.sector, e.composite_score, e.bias,
                 e.trend, round(e.ltp, 2), round(e.change_pct, 2)]
                for i, e in enumerate(rows)
            ]
        table = _table(headers, body)

        cal_note = (
            "Composite = technical + fundamental + sentiment | "
            f"providers: tech=engine, fund={self.fund_provider.name}, "
            f"news={self.news_provider.name}"
        )
        if self.composite.calibrator is not None:
            cal = self.composite.calibrator
            cal_note += (
                f" | calibrator: horizon={cal.horizon_bars}d, "
                f"lift={'yes' if cal.has_lift else 'no'}"
            )
        else:
            cal_note += " | calibrator: not fit yet - run BT FIT"
        blocks = [_note(cal_note), table]
        if budget:
            from backtest.backtest import CostModel

            cost_model = CostModel()
            blocks.append(_text(f"--- budget plan for Rs.{budget:,.0f} "
                                f"({self.settings.risk_per_trade_pct}% risk/trade) ---"))
            picks = screener.budget_picks(
                rows, budget, risk_pct=self.settings.risk_per_trade_pct)
            plan_rows = []
            for e, p in picks:
                target1 = p.targets[0] if p.targets else e.ltp
                outlay = p.position_value
                exit_value = target1 * p.shares
                rt_cost = cost_model.round_trip(outlay, exit_value)
                rt_pct = (rt_cost / outlay * 100.0) if outlay > 0 else 0.0
                plan_rows.append([
                    e.symbol, p.entry, p.stop,
                    "/".join(str(t) for t in p.targets), p.reward_risk,
                    p.shares, outlay, f"{rt_cost:.0f}", f"{rt_pct:.2f}%",
                ])
            blocks.append(_table(
                ["SYM", "ENTRY", "STOP", "T1/T2/T3", "R:R", "SHARES",
                 "OUTLAY", "RT COST", "COST %"],
                plan_rows,
            ))
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
        if args and args[0].upper() in ("SAVE", "LOAD", "LIST"):
            return self._watch_persist(args)
        if args:
            syms = [s.strip().upper() for s in
                    " ".join(args).replace(",", " ").split()]
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
            _text("set:  WATCH RELIANCE,TCS,INFY   |   "
                  "save:  WATCH SAVE <name>   |   load:  WATCH LOAD <name>"),
        ], subtitle=f"{len(rows)} symbols")

    def _watch_persist(self, args: list[str]) -> dict:
        import json
        from pathlib import Path

        base = Path("runs/watchlists")
        op = args[0].upper()
        if op == "LIST":
            base.mkdir(parents=True, exist_ok=True)
            names = sorted(p.stem for p in base.glob("*.json"))
            if not names:
                return _ok("WATCH - SAVED LISTS",
                           [_note("no saved watchlists yet")])
            return _ok("WATCH - SAVED LISTS", [
                _table(["NAME"], [[n] for n in names]),
                _text("load with:  WATCH LOAD <name>"),
            ])
        if len(args) < 2:
            return _err(f"WATCH {op} needs a name")
        name = args[1].strip().lower()
        safe = "".join(c for c in name if c.isalnum() or c in "-_")
        if not safe:
            return _err(f"invalid name {name!r}")
        path = base / f"{safe}.json"
        if op == "SAVE":
            base.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"symbols": self.watchlist}, indent=2))
            return _ok("WATCH - SAVED", [
                _text(f"saved {len(self.watchlist)} symbols to {path}")
            ])
        if op == "LOAD":
            if not path.exists():
                return _err(f"no watchlist named {safe!r}")
            data = json.loads(path.read_text())
            self.watchlist = list(data.get("symbols", []))
            return self._watch([])
        return _err(f"unknown WATCH op {op!r}")

    # --- heatmap / kite diagnostic ----------------------------------------
    def _heat(self) -> dict:
        rows = self._leaderboard()
        if not rows:
            return _err("no leaderboard data available")
        by_sector: dict[str, list[screener.LeaderboardEntry]] = {}
        for e in rows:
            by_sector.setdefault(e.sector or "?", []).append(e)
        sectors_out = []
        for sec, items in by_sector.items():
            items.sort(key=lambda x: x.composite_score, reverse=True)
            avg = sum(x.composite_score for x in items) / len(items)
            cells = [{
                "symbol": x.symbol,
                "score": round(x.composite_score, 1),
                "change_pct": round(x.change_pct, 2),
            } for x in items]
            sectors_out.append({"name": sec, "avg": round(avg, 1), "cells": cells})
        sectors_out.sort(key=lambda s: s["avg"], reverse=True)
        return _ok("HEAT - SECTOR HEATMAP", [
            _note("Composite score 0-100 per stock, grouped by sector. "
                  "Greener = stronger."),
            {"type": "heatmap", "sectors": sectors_out},
            _disclaimer(),
        ], subtitle=(
            f"{len(sectors_out)} sectors | "
            f"{sum(len(s['cells']) for s in sectors_out)} stocks"
        ))

    def _kite(self) -> dict:
        import importlib.util

        s = self.settings
        has_yf = importlib.util.find_spec("yfinance") is not None
        has_kc = importlib.util.find_spec("kiteconnect") is not None
        resolved = (
            self.history._resolve_provider() if s.is_live else "demo"
        )
        pairs = [
            ("MODE", s.mode),
            ("KITE_API_KEY", "set" if s.kite_api_key else "MISSING"),
            ("KITE_API_SECRET", "set" if s.kite_api_secret else "MISSING"),
            ("KITE_ACCESS_TOKEN", "set" if s.kite_access_token else "MISSING"),
            ("HISTORY_PROVIDER", s.history_provider),
            ("resolved history", resolved),
            ("yfinance installed", "yes" if has_yf else "no"),
            ("kiteconnect installed", "yes" if has_kc else "no"),
        ]
        blocks: list[dict] = [_keyval(pairs)]
        if not s.is_live:
            blocks.append(_note(
                "Running in DEMO mode. Set DALAL_MODE=live to engage "
                "Kite / yfinance providers."))
        elif not s.has_kite_creds:
            blocks.append(_note(
                "Live mode without Kite credentials. HistoryService is on "
                "the yfinance fallback (free Yahoo). Set KITE_API_KEY + "
                "KITE_API_SECRET + a fresh KITE_ACCESS_TOKEN to enable "
                "live quotes + Kite historical."))
        elif not s.kite_access_token:
            blocks.append(_note(
                "API key + secret set but ACCESS_TOKEN is missing. "
                "Visit  GET /kite/login  to start the daily login flow."))
        else:
            try:
                from core.kite_client import KiteClient

                client = KiteClient.from_settings(s)
                profile = client._kite.profile()
                blocks.append(_keyval([
                    ("user_name", profile.get("user_name", "?")),
                    ("broker", profile.get("broker", "?")),
                    ("email", profile.get("email", "?")),
                    ("user_type", profile.get("user_type", "?")),
                ]))
            except Exception as exc:
                blocks.append(_note(f"Kite profile probe failed: {exc}"))
                blocks.append(_text(
                    "If this is a TokenException the daily access token is "
                    "stale - regenerate via /kite/login."))
        blocks.append(_text(
            "Auth flow:  GET /kite/login  ->  /kite/callback?request_token=..."))
        return _ok("KITE - CONNECTION DIAGNOSTIC", blocks)

    # --- calendar ----------------------------------------------------------
    def _calendar_symbol(
        self, symbol: str, name: str, args: list[str]
    ) -> dict:
        days = self._parse_days(args, default=30)
        events = self.cal_provider.fetch(symbol, horizon_days=days)
        if not events:
            return _ok(
                f"{symbol} - CALENDAR",
                [_note(f"no events in the next {days} days for {symbol}."),
                 _text(f"provider: {self.cal_provider.name}")],
                subtitle=name,
            )
        rows = [
            [e.date.strftime("%b %d"), e.days_away,
             e.event_type.replace("_", " "), e.description]
            for e in events
        ]
        return _ok(f"{symbol} - CALENDAR ({days}d)", [
            _table(["DATE", "AWAY", "TYPE", "DETAIL"], rows),
            _text(f"provider: {self.cal_provider.name}"),
            _disclaimer(),
        ], subtitle=name)

    def _calendar_all(self, args: list[str]) -> dict:
        days = self._parse_days(args, default=14)
        pairs = []
        for stock in universe.DEFAULT_UNIVERSE:
            try:
                events = self.cal_provider.fetch(
                    stock.symbol, horizon_days=days)
            except Exception as exc:
                log.warning("CAL skip %s: %s", stock.symbol, exc)
                continue
            for ev in events:
                pairs.append((stock, ev))
        if not pairs:
            return _ok("CALENDAR - UPCOMING EVENTS", [
                _note(f"no events scheduled in the next {days} days."),
                _text(f"provider: {self.cal_provider.name}"),
            ])
        pairs.sort(key=lambda pair: pair[1].date)
        rows = [
            [ev.date.strftime("%b %d"), ev.days_away, stock.symbol,
             stock.sector, ev.event_type.replace("_", " "), ev.description]
            for stock, ev in pairs[:60]
        ]
        return _ok(f"CALENDAR - UPCOMING EVENTS ({days}d)", [
            _table(["DATE", "AWAY", "SYM", "SECTOR", "TYPE", "DETAIL"], rows),
            _text(f"provider: {self.cal_provider.name}"),
            _disclaimer(),
        ], subtitle=f"{len(pairs)} events")

    @staticmethod
    def _parse_days(args: list[str], default: int) -> int:
        for a in args:
            if a.upper().startswith("DAYS="):
                try:
                    return int(a.split("=", 1)[1])
                except ValueError:
                    pass
        return default

    # --- backtest / calibration -------------------------------------------
    def _bt(self, args: list[str]) -> dict:
        if args and args[0].upper() == "FIT":
            return self._bt_fit(args[1:])
        if args and args[0].upper() == "WALK":
            return self._bt_walk(args[1:])
        return self._bt_status()

    def _bt_walk(self, args: list[str]) -> dict:
        params = {"horizon": 10, "train": 1000, "test": 200, "step": 200}
        for a in args:
            for key in ("HORIZON", "TRAIN", "TEST", "STEP"):
                if a.upper().startswith(key + "="):
                    try:
                        params[key.lower()] = int(a.split("=", 1)[1])
                    except ValueError:
                        pass
        from backtest.walkforward import run_walk_forward

        try:
            report = run_walk_forward(
                self.settings,
                horizon_bars=params["horizon"],
                train_size=params["train"],
                test_size=params["test"],
                step=params["step"],
            )
        except Exception as exc:
            log.exception("walk-forward failed")
            return _err(f"walk-forward failed: {exc}")
        # hot-reload the calibrator in the running composite engine
        self.composite.calibrator = report.final_calibrator
        self._lb_cache = None

        cal = report.final_calibrator
        return _ok("WALK-FORWARD CALIBRATION", [
            _keyval([
                ("HORIZON (bars)", report.horizon_bars),
                ("POINTS", report.n_points),
                ("FOLDS", report.fold_count),
                ("LIFT FOLDS", f"{report.lift_folds}/{report.fold_count}"),
                ("STABLE LIFT?", "YES - probability exposed"
                 if report.stable_lift else "no - probability suppressed"),
                ("avg Brier (test)", round(report.avg_brier_test, 4)),
                ("avg Brier base", round(report.avg_brier_baseline, 4)),
                ("avg AUC (test)", round(report.avg_auc_test, 3)),
                ("avg ECE (test)", round(report.avg_ece_test, 4)),
            ]),
            _text(cal.lift_note if cal else ""),
            _table(
                ["FOLD", "N_TRAIN", "N_TEST", "BRIER", "BASE", "AUC", "ECE", "LIFT"],
                [[f.fold, f.n_train, f.n_test, round(f.brier_test, 4),
                  round(f.brier_baseline, 4), round(f.auc_test, 3),
                  round(f.ece_test, 4), "yes" if f.has_lift else "no"]
                 for f in report.folds],
            ),
            _disclaimer(),
        ])

    def _bt_status(self) -> dict:
        from backtest.calibration import DEFAULT_CALIBRATOR_PATH, Calibrator

        if not DEFAULT_CALIBRATOR_PATH.exists():
            return _ok("CALIBRATION STATUS", [
                _note("No calibrator on disk yet."),
                _text("Fit one with:  BT FIT   (or `python -m backtest.run`)"),
                _text("Until then `calibrated_probability` stays None - by design."),
            ])
        try:
            cal = Calibrator.load(DEFAULT_CALIBRATOR_PATH)
        except Exception as exc:
            return _err(f"could not load calibrator: {exc}")

        header = _keyval([
            ("HORIZON (bars)", cal.horizon_bars),
            ("TRAIN N", cal.n_train),
            ("TEST N", cal.n_test),
            ("BASE RATE", round(cal.base_rate, 3)),
            ("BRIER train", round(cal.brier_train, 4)),
            ("BRIER test", round(cal.brier_test, 4)),
            ("BRIER baseline", round(cal.brier_baseline, 4)),
            ("ECE test", round(cal.ece_test, 4)),
            ("AUC test", round(cal.auc_test, 3)),
            ("LIFT?", "YES - probability exposed" if cal.has_lift
                       else "no - probability suppressed"),
            ("FITTED AT", cal.fitted_at),
        ])
        rows = []
        for b in cal.reliability:
            if b["count"] == 0:
                rows.append([f"{b['bin_lo']:.1f}-{b['bin_hi']:.1f}",
                             "-", 0, ""])
                continue
            bar = "#" * max(0, int(round(b["observed"] * 30)))
            rows.append([
                f"{b['bin_lo']:.1f}-{b['bin_hi']:.1f}",
                f"{b['observed']:.3f}",
                b["count"], bar,
            ])
        return _ok("CALIBRATION REPORT", [
            header,
            _text(cal.lift_note),
            _text("reliability (predicted bin -> observed positive rate):"),
            _table(["PRED BIN", "OBSERVED", "N", "BAR"], rows),
            _disclaimer(),
        ])

    def _bt_fit(self, args: list[str]) -> dict:
        horizon = 10
        for a in args:
            if a.upper().startswith("HORIZON="):
                try:
                    horizon = int(a.split("=", 1)[1])
                except ValueError:
                    pass
        from backtest.harness import run_backtest

        try:
            report = run_backtest(self.settings, horizon_bars=horizon)
        except Exception as exc:
            log.exception("backtest failed")
            return _err(f"backtest failed: {exc}")
        # reload the calibrator in the running composite engine so the
        # next TOP / TA reflects it immediately.
        self.composite.calibrator = report.calibrator
        self._lb_cache = None
        log.info("fit complete - %d points, lift=%s",
                 report.n_points, report.calibrator.has_lift)
        return self._bt_status()

    # --- static ------------------------------------------------------------
    def _help(self) -> dict:
        return _ok("DALAL TERMINAL - COMMAND REFERENCE", [
            _table(["COMMAND", "DESCRIPTION"], [
                ["<TICKER> TA", "technical signal breakdown + 0-100 score"],
                ["<TICKER> GP", "price chart (sparkline) + key indicators"],
                ["<TICKER> FA", "fundamental snapshot (sector-relative)"],
                ["<TICKER> N", "latest news + sentiment score"],
                ["<TICKER> CAL  |  CAL [DAYS=n]", "upcoming results / ex-div / meetings"],
                ["TOP [n] [SECTOR=x] [BUDGET=n]", "ranked buy-probability board"],
                ["HEAT", "sector composite heat-map"],
                ["SCAN [key]", "list / run prebuilt screeners"],
                ["WATCH [a,b,c] | SAVE/LOAD/LIST", "live watchlist tape + persistence"],
                ["BT [FIT|WALK]", "calibration report / single fit / walk-forward"],
                ["KITE", "data-provider + Kite connection diagnostic"],
                ["RULES", "risk & discipline checklist"],
                ["HELP", "this reference"],
            ]),
            _text("examples:  RELIANCE TA   |   TOP 10 SECTOR=IT BUDGET=100000   |   "
                  "SCAN momentum   |   HEAT   |   BT FIT   |   WATCH SAVE morning"),
            _note("Analytics & research only - this terminal never places orders."),
        ])

    def _rules(self) -> dict:
        return _ok("RULES - RISK & DISCIPLINE", [
            _table(["#", "RULE"],
                   [[i + 1, r] for i, r in enumerate(risk_rules())]),
            _disclaimer(),
        ])
