from datetime import datetime
import pytest
from brokers.base import Order, Position


def make_order(**kwargs):
    defaults = dict(
        id="o1",
        ticker="AAPL",
        side="buy",
        quantity=10,
        price=150.0,
        status="filled",
        timestamp=datetime.utcnow(),
        filled_price=150.5,
    )
    defaults.update(kwargs)
    return Order(**defaults)


def make_position(**kwargs):
    defaults = dict(
        ticker="AAPL",
        quantity=10,
        avg_price=150.0,
        current_price=160.0,
    )
    defaults.update(kwargs)
    return Position(**defaults)


class TestOrder:
    def test_value_uses_filled_price(self):
        o = make_order(quantity=10, filled_price=151.0)
        assert o.value == pytest.approx(1510.0)

    def test_value_falls_back_to_price(self):
        o = make_order(quantity=5, price=200.0, filled_price=None)
        assert o.value == pytest.approx(1000.0)


class TestPosition:
    def test_market_value(self):
        p = make_position(quantity=10, current_price=160.0)
        assert p.market_value == pytest.approx(1600.0)

    def test_cost_basis(self):
        p = make_position(quantity=10, avg_price=150.0)
        assert p.cost_basis == pytest.approx(1500.0)

    def test_unrealized_pnl(self):
        p = make_position(quantity=10, avg_price=150.0, current_price=160.0)
        assert p.unrealized_pnl == pytest.approx(100.0)

    def test_unrealized_pnl_pct(self):
        p = make_position(quantity=10, avg_price=150.0, current_price=160.0)
        assert p.unrealized_pnl_pct == pytest.approx(100.0 / 1500.0)

    def test_unrealized_pnl_pct_zero_cost(self):
        p = make_position(quantity=0, avg_price=0.0, current_price=100.0)
        assert p.unrealized_pnl_pct == 0.0
