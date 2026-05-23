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
    monkeypatch.chdir(tmp_path)
    resp = _router().dispatch("BT")
    assert resp["ok"] is True
    assert "CALIBRATION" in resp["title"]
    # the no-calibrator path always emits a note
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
    monkeypatch.chdir(tmp_path)
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


def test_top_budget_includes_cost_columns():
    resp = _router().dispatch("TOP 5 BUDGET=100000")
    assert resp["ok"] is True
    tables = [b for b in resp["blocks"] if b["type"] == "table"]
    plan = next(t for t in tables
                if "OUTLAY" in t["headers"])
    assert "RT COST" in plan["headers"]
    assert "COST %" in plan["headers"]
