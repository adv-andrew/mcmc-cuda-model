import pytest
from brokers.paper import PaperBroker
from trading.risk_manager import RiskManager


@pytest.fixture
def broker():
    b = PaperBroker(initial_cash=100_000.0, slippage_pct=0.0, market_impact_pct=0.0)
    return b


@pytest.fixture
def rm(broker):
    r = RiskManager(
        broker,
        position_size_pct=0.10,
        max_position_pct=0.25,
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        daily_loss_limit_pct=0.05,
    )
    return r


class TestCalculatePositionSize:
    def test_basic_size(self, broker, rm):
        # 10% of 100_000 = 10_000 / 50 = 200 shares
        size = rm.calculate_position_size("AAPL", 50.0)
        assert size == 200

    def test_signal_strength_scales_size(self, broker, rm):
        size_full = rm.calculate_position_size("AAPL", 50.0, signal_strength=1.0)
        size_half = rm.calculate_position_size("AAPL", 50.0, signal_strength=0.5)
        assert size_half == size_full // 2

    def test_zero_price_returns_zero(self, broker, rm):
        assert rm.calculate_position_size("AAPL", 0.0) == 0

    def test_capped_by_max_position(self, broker, rm):
        # Buy up to max (25% = 25_000). With slippage=0, fills at exact price.
        broker.buy("AAPL", 250, 100.0)  # 25_000 cost -> at max
        broker.set_price("AAPL", 100.0)
        size = rm.calculate_position_size("AAPL", 100.0)
        assert size == 0

    def test_capped_by_cash(self, broker, rm):
        # Use up most cash first
        broker.buy("MSFT", 900, 100.0)  # 90_000 spent, 10_000 left
        # 10% of equity ~100k = 10_000, but only 10_000 cash left
        size = rm.calculate_position_size("AAPL", 100.0)
        assert size <= 100


class TestStopLoss:
    def test_no_position_returns_false(self, rm):
        assert rm.should_stop_loss("AAPL") is False

    def test_loss_below_threshold_triggers(self, broker, rm):
        broker.buy("AAPL", 100, 100.0)
        broker.set_price("AAPL", 97.0)  # -3% > 2% stop
        assert rm.should_stop_loss("AAPL") is True

    def test_small_loss_does_not_trigger(self, broker, rm):
        broker.buy("AAPL", 100, 100.0)
        broker.set_price("AAPL", 99.5)  # -0.5% < 2% stop
        assert rm.should_stop_loss("AAPL") is False


class TestTakeProfit:
    def test_no_position_returns_false(self, rm):
        assert rm.should_take_profit("AAPL") is False

    def test_gain_above_threshold_triggers(self, broker, rm):
        broker.buy("AAPL", 100, 100.0)
        broker.set_price("AAPL", 106.0)  # +6% > 5% take-profit
        assert rm.should_take_profit("AAPL") is True

    def test_small_gain_does_not_trigger(self, broker, rm):
        broker.buy("AAPL", 100, 100.0)
        broker.set_price("AAPL", 103.0)  # +3% < 5%
        assert rm.should_take_profit("AAPL") is False


class TestDailyLoss:
    def test_no_loss_not_exceeded(self, rm):
        assert rm.is_daily_loss_exceeded() is False

    def test_daily_loss_exceeded(self, broker, rm):
        # Simulate equity drop of 6% (> 5% limit)
        rm.reset_day_start(equity=100_000.0)
        broker.buy("AAPL", 1000, 100.0)  # spend 100k, 0 cash left
        broker.set_price("AAPL", 94.0)   # equity now ~94k
        assert rm.is_daily_loss_exceeded() is True

    def test_small_loss_not_exceeded(self, broker, rm):
        rm.reset_day_start(equity=100_000.0)
        broker.buy("AAPL", 1000, 100.0)
        broker.set_price("AAPL", 99.0)  # -1% < 5%
        assert rm.is_daily_loss_exceeded() is False


class TestCanTrade:
    def test_can_trade_normally(self, rm):
        assert rm.can_trade() is True

    def test_cannot_trade_after_daily_loss(self, broker, rm):
        rm.reset_day_start(equity=100_000.0)
        broker.buy("AAPL", 1000, 100.0)
        broker.set_price("AAPL", 94.0)
        assert rm.can_trade() is False


class TestGetExitSignals:
    def test_no_signals_when_flat(self, rm):
        assert rm.get_exit_signals() == []

    def test_stop_loss_signal(self, broker, rm):
        broker.buy("AAPL", 100, 100.0)
        broker.set_price("AAPL", 97.0)
        signals = rm.get_exit_signals()
        assert any(s["ticker"] == "AAPL" and s["reason"] == "stop_loss" for s in signals)

    def test_take_profit_signal(self, broker, rm):
        broker.buy("AAPL", 100, 100.0)
        broker.set_price("AAPL", 106.0)
        signals = rm.get_exit_signals()
        assert any(s["ticker"] == "AAPL" and s["reason"] == "take_profit" for s in signals)

    def test_multiple_signals(self, broker, rm):
        broker.buy("AAPL", 10, 100.0)
        broker.set_price("AAPL", 97.0)
        broker.buy("MSFT", 10, 200.0)
        broker.set_price("MSFT", 212.0)
        signals = rm.get_exit_signals()
        tickers = {s["ticker"] for s in signals}
        assert "AAPL" in tickers
        assert "MSFT" in tickers
