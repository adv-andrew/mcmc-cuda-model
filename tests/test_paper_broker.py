import pytest
from brokers.paper import PaperBroker


@pytest.fixture
def broker():
    return PaperBroker(initial_cash=10_000.0, slippage_pct=0.001, market_impact_pct=0.0005)


class TestPaperBrokerInit:
    def test_initial_cash(self, broker):
        assert broker.get_cash() == pytest.approx(10_000.0)

    def test_initial_equity_equals_cash(self, broker):
        assert broker.get_equity() == pytest.approx(10_000.0)

    def test_no_positions(self, broker):
        assert broker.get_positions() == {}

    def test_no_orders(self, broker):
        assert broker.get_orders() == []


class TestSetPrice:
    def test_set_price_updates_position_current_price(self, broker):
        broker.set_price("AAPL", 150.0)
        broker.buy("AAPL", 10, 150.0)
        broker.set_price("AAPL", 160.0)
        pos = broker.get_position("AAPL")
        assert pos.current_price == pytest.approx(160.0)


class TestBuy:
    def test_buy_reduces_cash(self, broker):
        broker.set_price("AAPL", 100.0)
        broker.buy("AAPL", 10, 100.0)
        assert broker.get_cash() < 10_000.0

    def test_buy_creates_position(self, broker):
        broker.buy("AAPL", 5, 100.0)
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 5

    def test_buy_slippage_increases_fill_price(self, broker):
        order = broker.buy("AAPL", 1, 100.0)
        assert order.filled_price > 100.0

    def test_buy_order_recorded(self, broker):
        broker.buy("AAPL", 1, 100.0)
        orders = broker.get_orders()
        assert len(orders) == 1
        assert orders[0].side == "buy"

    def test_buy_insufficient_cash_raises(self, broker):
        with pytest.raises(ValueError, match="Insufficient cash"):
            broker.buy("AAPL", 1000, 100.0)

    def test_buy_zero_quantity_raises(self, broker):
        with pytest.raises(ValueError):
            broker.buy("AAPL", 0, 100.0)

    def test_buy_accumulates_position(self, broker):
        broker.buy("AAPL", 5, 100.0)
        broker.buy("AAPL", 5, 110.0)
        pos = broker.get_position("AAPL")
        assert pos.quantity == 10
        assert 100.0 < pos.avg_price < 112.0


class TestSell:
    def test_sell_increases_cash(self, broker):
        broker.buy("AAPL", 10, 100.0)
        cash_after_buy = broker.get_cash()
        broker.sell("AAPL", 10, 100.0)
        assert broker.get_cash() > cash_after_buy

    def test_sell_reduces_position(self, broker):
        broker.buy("AAPL", 10, 100.0)
        broker.sell("AAPL", 5, 100.0)
        assert broker.get_position("AAPL").quantity == 5

    def test_sell_all_removes_position(self, broker):
        broker.buy("AAPL", 10, 100.0)
        broker.sell("AAPL", 10, 100.0)
        assert broker.get_position("AAPL") is None

    def test_sell_slippage_decreases_fill_price(self, broker):
        broker.buy("AAPL", 1, 100.0)
        order = broker.sell("AAPL", 1, 100.0)
        assert order.filled_price < 100.0

    def test_sell_insufficient_position_raises(self, broker):
        broker.buy("AAPL", 5, 100.0)
        with pytest.raises(ValueError, match="Insufficient position"):
            broker.sell("AAPL", 10, 100.0)

    def test_sell_no_position_raises(self, broker):
        with pytest.raises(ValueError):
            broker.sell("AAPL", 1, 100.0)


class TestEquity:
    def test_equity_includes_positions(self, broker):
        broker.set_price("AAPL", 100.0)
        broker.buy("AAPL", 10, 100.0)
        broker.set_price("AAPL", 200.0)
        equity = broker.get_equity()
        assert equity > 10_000.0  # profit


class TestReset:
    def test_reset_restores_cash(self, broker):
        broker.buy("AAPL", 10, 100.0)
        broker.reset()
        assert broker.get_cash() == pytest.approx(10_000.0)

    def test_reset_clears_positions(self, broker):
        broker.buy("AAPL", 10, 100.0)
        broker.reset()
        assert broker.get_positions() == {}

    def test_reset_clears_orders(self, broker):
        broker.buy("AAPL", 1, 100.0)
        broker.reset()
        assert broker.get_orders() == []
