"""Tests for PositionManager."""

import pytest

from trading.position_manager import PositionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pm():
    return PositionManager()


# ---------------------------------------------------------------------------
# open_position
# ---------------------------------------------------------------------------


class TestOpenPosition:
    def test_opens_new_position(self):
        pm = _pm()
        pos = pm.open_position("AAPL", 10, 150.0, "2024-01-01")
        assert pos["ticker"] == "AAPL"
        assert pos["quantity"] == 10
        assert pos["avg_price"] == pytest.approx(150.0)

    def test_adds_to_existing_position_updates_avg_price(self):
        pm = _pm()
        pm.open_position("AAPL", 10, 100.0)
        pos = pm.open_position("AAPL", 10, 200.0)
        assert pos["quantity"] == 20
        assert pos["avg_price"] == pytest.approx(150.0)

    def test_invalid_quantity_raises(self):
        pm = _pm()
        with pytest.raises(ValueError, match="quantity"):
            pm.open_position("X", 0, 100.0)

    def test_invalid_price_raises(self):
        pm = _pm()
        with pytest.raises(ValueError, match="price"):
            pm.open_position("X", 5, -10.0)

    def test_multiple_tickers_independent(self):
        pm = _pm()
        pm.open_position("AAPL", 5, 100.0)
        pm.open_position("TSLA", 3, 200.0)
        assert pm.get_position("AAPL")["quantity"] == 5
        assert pm.get_position("TSLA")["quantity"] == 3


# ---------------------------------------------------------------------------
# close_position
# ---------------------------------------------------------------------------


class TestClosePosition:
    def test_full_close_removes_position(self):
        pm = _pm()
        pm.open_position("AAPL", 10, 100.0, "open_ts")
        trade = pm.close_position("AAPL", 10, 110.0, "close_ts")
        assert pm.get_position("AAPL") is None
        assert trade["pnl"] == pytest.approx(100.0)

    def test_partial_close_reduces_quantity(self):
        pm = _pm()
        pm.open_position("AAPL", 10, 100.0)
        pm.close_position("AAPL", 4, 110.0)
        pos = pm.get_position("AAPL")
        assert pos["quantity"] == 6

    def test_pnl_negative_on_loss(self):
        pm = _pm()
        pm.open_position("AAPL", 5, 200.0)
        trade = pm.close_position("AAPL", 5, 180.0)
        assert trade["pnl"] == pytest.approx(-100.0)

    def test_close_nonexistent_raises_key_error(self):
        pm = _pm()
        with pytest.raises(KeyError, match="AAPL"):
            pm.close_position("AAPL", 5, 100.0)

    def test_close_too_many_raises_value_error(self):
        pm = _pm()
        pm.open_position("AAPL", 5, 100.0)
        with pytest.raises(ValueError, match="Cannot close"):
            pm.close_position("AAPL", 10, 100.0)

    def test_trade_record_has_expected_keys(self):
        pm = _pm()
        pm.open_position("AAPL", 5, 100.0, "t0")
        trade = pm.close_position("AAPL", 5, 110.0, "t1")
        expected_keys = {
            "ticker", "quantity", "entry_price", "exit_price",
            "pnl", "open_timestamp", "close_timestamp",
        }
        assert expected_keys == set(trade.keys())

    def test_close_invalid_price_raises(self):
        pm = _pm()
        pm.open_position("AAPL", 5, 100.0)
        with pytest.raises(ValueError, match="price"):
            pm.close_position("AAPL", 5, 0.0)


# ---------------------------------------------------------------------------
# get_position / get_all_positions
# ---------------------------------------------------------------------------


class TestGetPosition:
    def test_returns_none_for_unknown_ticker(self):
        pm = _pm()
        assert pm.get_position("UNKNOWN") is None

    def test_all_positions_returns_all_open(self):
        pm = _pm()
        pm.open_position("AAPL", 5, 100.0)
        pm.open_position("TSLA", 3, 200.0)
        all_pos = pm.get_all_positions()
        assert set(all_pos.keys()) == {"AAPL", "TSLA"}

    def test_closed_position_not_in_all_positions(self):
        pm = _pm()
        pm.open_position("AAPL", 5, 100.0)
        pm.close_position("AAPL", 5, 110.0)
        assert pm.get_all_positions() == {}


# ---------------------------------------------------------------------------
# get_trades
# ---------------------------------------------------------------------------


class TestGetTrades:
    def test_empty_initially(self):
        pm = _pm()
        assert pm.get_trades() == []

    def test_records_closed_trades(self):
        pm = _pm()
        pm.open_position("AAPL", 10, 100.0)
        pm.close_position("AAPL", 5, 110.0)
        pm.close_position("AAPL", 5, 120.0)
        trades = pm.get_trades()
        assert len(trades) == 2

    def test_returns_copy(self):
        pm = _pm()
        pm.open_position("AAPL", 5, 100.0)
        pm.close_position("AAPL", 5, 110.0)
        trades = pm.get_trades()
        trades.clear()
        assert len(pm.get_trades()) == 1


# ---------------------------------------------------------------------------
# P&L
# ---------------------------------------------------------------------------


class TestPnL:
    def test_total_realized_pnl_zero_initially(self):
        pm = _pm()
        assert pm.get_total_realized_pnl() == pytest.approx(0.0)

    def test_total_realized_pnl_accumulates(self):
        pm = _pm()
        pm.open_position("AAPL", 10, 100.0)
        pm.close_position("AAPL", 5, 110.0)   # pnl = +50
        pm.close_position("AAPL", 5, 90.0)    # pnl = -50
        assert pm.get_total_realized_pnl() == pytest.approx(0.0)

    def test_unrealized_pnl_positive(self):
        pm = _pm()
        pm.open_position("AAPL", 10, 100.0)
        upnl = pm.get_unrealized_pnl({"AAPL": 110.0})
        assert upnl["AAPL"] == pytest.approx(100.0)

    def test_unrealized_pnl_negative(self):
        pm = _pm()
        pm.open_position("AAPL", 10, 100.0)
        upnl = pm.get_unrealized_pnl({"AAPL": 90.0})
        assert upnl["AAPL"] == pytest.approx(-100.0)

    def test_unrealized_pnl_skips_missing_tickers(self):
        pm = _pm()
        pm.open_position("AAPL", 10, 100.0)
        pm.open_position("TSLA", 5, 200.0)
        upnl = pm.get_unrealized_pnl({"AAPL": 110.0})
        assert "TSLA" not in upnl
        assert "AAPL" in upnl

    def test_unrealized_pnl_empty_when_no_positions(self):
        pm = _pm()
        assert pm.get_unrealized_pnl({"AAPL": 110.0}) == {}
