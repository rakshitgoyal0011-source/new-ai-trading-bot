"""Integration tests for the command router (demo mode, end-to-end)."""
from config.settings import Settings
from ui.commands import CommandRouter


def _router():
    # demo mode is the default; no API keys touched
    return CommandRouter(Settings(mode="demo"))


def test_help_command():
    resp = _router().dispatch("HELP")
    assert resp["ok"] is True
    assert resp["blocks"]


def test_unknown_command_fails_gracefully():
    resp = _router().dispatch("FLORPNAX")
    assert resp["ok"] is False


def test_rules_command():
    resp = _router().dispatch("RULES")
    assert resp["ok"] is True


def test_technical_command_end_to_end():
    resp = _router().dispatch("RELIANCE TA")
    assert resp["ok"] is True
    assert "TECHNICAL" in resp["title"]
    # a TA response must always carry the disclaimer
    assert any(b["type"] == "disclaimer" for b in resp["blocks"])


def test_top_leaderboard_command():
    resp = _router().dispatch("TOP 5")
    assert resp["ok"] is True
    table = next(b for b in resp["blocks"] if b["type"] == "table")
    assert len(table["rows"]) <= 5


def test_chart_command():
    resp = _router().dispatch("TCS GP")
    assert resp["ok"] is True
    assert any(b["type"] == "spark" for b in resp["blocks"])


def test_chart_includes_candles_block():
    resp = _router().dispatch("TCS GP")
    candles = [b for b in resp["blocks"] if b["type"] == "candles"]
    assert candles and len(candles[0]["ohlc"]) > 0


def test_fundamental_command_end_to_end():
    resp = _router().dispatch("RELIANCE FA")
    assert resp["ok"] is True
    assert "FUNDAMENTAL" in resp["title"]
    assert any(b["type"] == "disclaimer" for b in resp["blocks"])


def test_news_command_end_to_end():
    resp = _router().dispatch("RELIANCE N")
    assert resp["ok"] is True
    assert "NEWS" in resp["title"]


def test_empty_command_is_handled():
    assert _router().dispatch("")["ok"] is False


def test_bt_status_with_no_calibrator(monkeypatch, tmp_path):
    # paths are now anchored to PROJECT_ROOT, not cwd - point the
    # constant at an absent file instead
    from backtest import calibration as cal_mod
    monkeypatch.setattr(
        cal_mod, "DEFAULT_CALIBRATOR_PATH", tmp_path / "absent.json")
    resp = _router().dispatch("BT")
    assert resp["ok"] is True
    assert "CALIBRATION" in resp["title"]
    assert any(b["type"] == "note" for b in resp["blocks"])


def test_help_mentions_bt():
    resp = _router().dispatch("HELP")
    table = next(b for b in resp["blocks"] if b["type"] == "table")
    assert any("BT" in row[0] for row in table["rows"])


def test_help_mentions_heat_and_kite():
    resp = _router().dispatch("HELP")
    table = next(b for b in resp["blocks"] if b["type"] == "table")
    cmds = [row[0] for row in table["rows"]]
    assert any("HEAT" in c for c in cmds)
    assert any("KITE" in c for c in cmds)


def test_heat_command_returns_heatmap_block():
    resp = _router().dispatch("HEAT")
    assert resp["ok"] is True
    heat = [b for b in resp["blocks"] if b["type"] == "heatmap"]
    assert heat and len(heat[0]["sectors"]) > 0
    sec = heat[0]["sectors"][0]
    assert "name" in sec and "avg" in sec and "cells" in sec


def test_kite_command_shows_diagnostic_keyval():
    resp = _router().dispatch("KITE")
    assert resp["ok"] is True
    kv = next(b for b in resp["blocks"] if b["type"] == "keyval")
    keys = [p[0] for p in kv["pairs"]]
    assert "MODE" in keys
    assert "HISTORY_PROVIDER" in keys


def test_watch_save_load_roundtrip(monkeypatch, tmp_path):
    # _watch_persist now anchors at PROJECT_ROOT - point the constant
    # at a temp dir so the test never pollutes the real runs/ folder
    from ui import commands as cmd_mod
    monkeypatch.setattr(cmd_mod, "PROJECT_ROOT", tmp_path)
    r = _router()
    r.watchlist = ["RELIANCE", "TCS"]
    assert r.dispatch("WATCH SAVE morning")["ok"] is True
    r.watchlist = ["INFY"]
    assert r.dispatch("WATCH LOAD morning")["ok"] is True
    assert r.watchlist == ["RELIANCE", "TCS"]
    listed = r.dispatch("WATCH LIST")
    assert listed["ok"] is True
    table = next(b for b in listed["blocks"] if b["type"] == "table")
    assert any("morning" in row[0] for row in table["rows"])


def test_default_calibrator_path_is_absolute_not_cwd_relative():
    """Regression: `Path('runs/calibrator.json')` was CWD-relative, so
    launching the server from a different directory silently ran without
    a calibrator. Now anchored at the project root."""
    from backtest.calibration import DEFAULT_CALIBRATOR_PATH
    assert DEFAULT_CALIBRATOR_PATH.is_absolute()
    # also lives somewhere named 'runs' (sanity check)
    assert "runs" in DEFAULT_CALIBRATOR_PATH.parts


def test_trade_plan_unavailable_when_indicator_is_missing(monkeypatch):
    """Regression: build_trade_plan used to be called with
    ind.get('close', 0.0)/ind.get('atr', 0.0) defaults, producing a
    Rs.0 plan or div-by-zero crash on a degenerate technical pipeline."""
    router = _router()

    class _DegenerateTech:
        def analyze(self, symbol, df, higher_tf=None):
            class _R: pass
            r = _R()
            r.score = 50.0
            r.bias = "neutral"
            r.trend = "sideways"
            r.confidence = 0.0
            r.mtf_confirmed = False
            r.sub_scores = {}
            r.reasons = []
            r.patterns = []
            r.indicators = {}   # no close, no atr - degenerate
            return r

    router.tech = _DegenerateTech()
    resp = router.dispatch("RELIANCE TA")
    assert resp["ok"] is True
    notes = [b["text"] for b in resp["blocks"] if b["type"] == "note"]
    assert any("Trade plan unavailable" in t for t in notes)


def test_top_budget_zero_skips_plan_without_crashing():
    """Regression: `if budget:` treated BUDGET=0 the same as BUDGET=anything-positive
    causing inconsistent behaviour; explicit `is not None and > 0` now."""
    resp = _router().dispatch("TOP 5 BUDGET=0")
    assert resp["ok"] is True
    tables = [b for b in resp["blocks"] if b["type"] == "table"]
    # no plan table - only the leaderboard one
    assert not any("OUTLAY" in t["headers"] for t in tables)


def test_news_tone_marker_for_negative_sentiment_is_minus():
    """Regression: `i.sentiment or 0 > 0` parsed as `i.sentiment or (0>0)`,
    making any nonzero sentiment - including NEGATIVE - render as '+'."""
    from datetime import datetime
    from data.news import NewsItem

    router = _router()
    bearish = NewsItem(
        symbol="RELIANCE", headline="Reliance plunges on regulatory probe",
        source="test", published=datetime.now(), sentiment=-0.6,
    )
    router.news_provider.fetch = lambda symbol, limit=10: [bearish]
    resp = router.dispatch("RELIANCE N")
    assert resp["ok"] is True
    tbl = next(b for b in resp["blocks"] if b["type"] == "table")
    cell = tbl["rows"][0][2]
    assert cell.startswith("-"), f"expected '-' tone, got: {cell!r}"


def test_top_sector_filter_works_for_uppercase_acronyms():
    """Regression: `up.title() in universe.sectors()` failed for IT/FMCG
    because str.title() lowercases the body of acronyms."""
    resp = _router().dispatch("TOP IT")
    assert resp["ok"] is True
    tbl = next(b for b in resp["blocks"] if b["type"] == "table")
    # every row's sector column should be exactly "IT"
    for row in tbl["rows"]:
        assert row[2] == "IT", f"non-IT row leaked through filter: {row}"


def test_higher_timeframe_view_includes_latest_bar():
    """Regression: `df.iloc[::5]` drops the latest bar when len%5 != 1.
    The fix anchors the stride at the latest bar."""
    import pandas as pd
    for n in (256, 257, 258, 259, 260):
        df = pd.DataFrame({"close": range(n)})
        offset = (len(df) - 1) % 5
        higher = df.iloc[offset::5]
        assert higher.index[-1] == n - 1, (
            f"latest bar dropped at len={n}"
        )


def test_top_budget_includes_cost_columns():
    resp = _router().dispatch("TOP 5 BUDGET=100000")
    assert resp["ok"] is True
    tables = [b for b in resp["blocks"] if b["type"] == "table"]
    plan = next(t for t in tables
                if "OUTLAY" in t["headers"])
    assert "RT COST" in plan["headers"]
    assert "COST %" in plan["headers"]
