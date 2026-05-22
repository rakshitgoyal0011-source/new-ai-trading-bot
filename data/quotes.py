"""Live quote service.

Demo mode steps the synthetic DemoMarket each refresh. Live mode opens
the Kite websocket (KiteTicker), subscribes the tracked symbols and
keeps a dict of the latest Quote per symbol. The web terminal reads
snapshots from here for the ticker tape and watchlist.
"""
from __future__ import annotations

from data.demo import get_demo_market
from data.models import Quote
from monitoring.logging import get_logger

log = get_logger("data.quotes")


class QuoteService:
    def __init__(self, settings, kite_client=None, instruments=None):
        self.settings = settings
        self._kite = kite_client
        self._instruments = instruments
        self._quotes: dict[str, Quote] = {}
        self._token_symbol: dict[int, str] = {}
        self._symbols: list[str] = []
        self._ticker = None
        self._demo = None if settings.is_live else get_demo_market()

    def track(self, symbols: list[str]) -> None:
        """Set the symbol set to keep quotes for (starts the ticker if live)."""
        self._symbols = [s.strip().upper() for s in symbols]
        if self.settings.is_live and self._ticker is None:
            self._start_ticker()

    def step(self) -> None:
        """Advance demo prices one tick (no-op in live mode; the ticker pushes)."""
        if self._demo is not None:
            self._demo.step(self._symbols)

    def snapshot(self, symbol: str) -> Quote | None:
        symbol = symbol.strip().upper()
        if self._demo is not None:
            return self._demo.quote(symbol)
        return self._quotes.get(symbol)

    def all(self) -> list[Quote]:
        if self._demo is not None:
            return [self._demo.quote(s) for s in self._symbols]
        return [self._quotes[s] for s in self._symbols if s in self._quotes]

    # --- live websocket ----------------------------------------------------
    def _start_ticker(self) -> None:
        from kiteconnect import KiteTicker

        tokens: list[int] = []
        for sym in self._symbols:
            inst = self._instruments.resolve(sym)
            if inst is not None:
                self._token_symbol[inst.instrument_token] = sym
                tokens.append(inst.instrument_token)

        ticker = KiteTicker(self._kite.api_key, self._kite.access_token)

        def on_connect(ws, _response):
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_QUOTE, tokens)
            log.info("KiteTicker connected - subscribed %d instruments", len(tokens))

        def on_ticks(_ws, ticks):
            for tick in ticks:
                self._ingest(tick)

        def on_error(_ws, code, reason):
            log.warning("KiteTicker error %s: %s", code, reason)

        ticker.on_connect = on_connect
        ticker.on_ticks = on_ticks
        ticker.on_error = on_error
        ticker.connect(threaded=True)
        self._ticker = ticker

    def _ingest(self, tick: dict) -> None:
        symbol = self._token_symbol.get(tick.get("instrument_token"))
        if not symbol:
            return
        ohlc = tick.get("ohlc", {}) or {}
        self._quotes[symbol] = Quote(
            symbol=symbol,
            ltp=float(tick.get("last_price", 0.0) or 0.0),
            prev_close=float(ohlc.get("close", 0.0) or 0.0),
            day_open=float(ohlc.get("open", 0.0) or 0.0),
            day_high=float(ohlc.get("high", 0.0) or 0.0),
            day_low=float(ohlc.get("low", 0.0) or 0.0),
            volume=int(tick.get("volume_traded", 0) or 0),
        )
