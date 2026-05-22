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
