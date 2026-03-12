"""End-to-end integration tests for the MCMC Trading System."""

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import Backtester
from brokers.paper import PaperBroker
from trading.indicator import MCMCIndicator
from trading.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trending_data(n: int = 80, start_price: float = 100.0) -> dict:
    """Return a data dict with a clear upward trend to encourage BUY signals."""
    np.random.seed(42)
    # Upward drift of 0.5% per bar to produce reliable BUY signals
    prices = start_price * np.cumprod(1 + np.random.normal(0.005, 0.008, n))
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": prices * 0.999,
            "High": prices * 1.008,
            "Low": prices * 0.993,
            "Close": prices,
            "Volume": np.ones(n) * 2_000_000,
        },
        index=idx,
    )
    return {"AAPL": df}


def _make_multi_ticker_data(n: int = 80) -> dict:
    """Return a two-ticker data dict for integration tests."""
    np.random.seed(7)
    data = {}
    for ticker, drift in [("AAPL", 0.004), ("MSFT", 0.003)]:
        prices = 150.0 * np.cumprod(1 + np.random.normal(drift, 0.009, n))
        idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": prices * 0.999,
                "High": prices * 1.007,
                "Low": prices * 0.993,
                "Close": prices,
                "Volume": np.ones(n) * 1_500_000,
            },
            index=idx,
        )
        data[ticker] = df
    return data


# ---------------------------------------------------------------------------
# test_full_trading_loop
# ---------------------------------------------------------------------------

class TestFullTradingLoop:
    """Test the end-to-end signal -> trade -> position lifecycle."""

    def test_full_trading_loop(self):
        """Generate a signal, execute a trade, verify position is recorded."""
        broker = PaperBroker(initial_cash=100_000.0)
        indicator = MCMCIndicator(
            n_simulations=5000,
            n_regimes=3,
            slope_threshold=5.0,
        )
        risk = RiskManager(
            broker=broker,
            position_size_pct=0.10,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
        )

        data = _make_trending_data(n=60)
        ticker = "AAPL"
        df = data[ticker]
        current_price = float(df["Close"].iloc[-1])
        broker.set_price(ticker, current_price)

        # Generate signal
        signal = indicator.generate_signal(ticker, df, "1d")

        assert "suggested_action" in signal
        assert "signal_strength" in signal
        assert 0.0 <= signal["signal_strength"] <= 1.0

        # Execute a buy regardless of signal to verify position management
        initial_equity = broker.get_equity()
        qty = risk.calculate_position_size(ticker, current_price, 0.8)
        assert qty >= 0

        if qty > 0:
            broker.buy(ticker, qty, current_price)
            position = broker.get_position(ticker)
            assert position is not None, "Position should exist after buy"
            assert position.quantity == qty
            assert position.ticker == ticker

            # Equity should decrease by cost of position (approximately)
            assert broker.get_cash() < initial_equity

            # Sell to close
            broker.sell(ticker, qty, current_price)
            position_after_sell = broker.get_position(ticker)
            assert position_after_sell is None, "Position should be gone after full sell"

    def test_stop_loss_triggers_on_price_drop(self):
        """Verify stop-loss logic fires when price drops below threshold."""
        broker = PaperBroker(initial_cash=50_000.0)
        risk = RiskManager(
            broker=broker,
            position_size_pct=0.10,
            stop_loss_pct=0.02,
            take_profit_pct=0.10,
        )
        ticker = "MSFT"
        entry_price = 200.0
        broker.set_price(ticker, entry_price)
        broker.buy(ticker, 10, entry_price)

        # Simulate a 5% price drop (beyond the 2% stop)
        drop_price = entry_price * 0.94
        broker.set_price(ticker, drop_price)

        assert risk.should_stop_loss(ticker), "Stop loss should trigger after 5% drop"
        assert not risk.should_take_profit(ticker)

    def test_take_profit_triggers_on_price_rise(self):
        """Verify take-profit logic fires when price rises above threshold."""
        broker = PaperBroker(initial_cash=50_000.0)
        risk = RiskManager(
            broker=broker,
            position_size_pct=0.10,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
        )
        ticker = "NVDA"
        entry_price = 300.0
        broker.set_price(ticker, entry_price)
        broker.buy(ticker, 5, entry_price)

        # Simulate a 7% price rise (beyond the 5% take-profit)
        rise_price = entry_price * 1.07
        broker.set_price(ticker, rise_price)

        assert risk.should_take_profit(ticker), "Take profit should trigger after 7% rise"
        assert not risk.should_stop_loss(ticker)

    def test_broker_equity_tracks_position_value(self):
        """Equity should reflect market value of open positions."""
        broker = PaperBroker(initial_cash=10_000.0)
        ticker = "SPY"
        buy_price = 400.0
        broker.set_price(ticker, buy_price)
        broker.buy(ticker, 10, buy_price)

        # Update market price upward
        new_price = 420.0
        broker.set_price(ticker, new_price)

        equity = broker.get_equity()
        # Equity = cash + (10 shares * 420)
        expected_min = broker.get_cash() + 10 * new_price
        assert equity >= expected_min * 0.98  # allow slippage


# ---------------------------------------------------------------------------
# test_backtest_produces_valid_results
# ---------------------------------------------------------------------------

class TestBacktestResults:
    """Verify that Backtester returns well-formed result structures."""

    def test_backtest_produces_valid_results(self):
        """Single-window backtest returns valid metrics dict."""
        data = _make_trending_data(n=80)
        bt = Backtester(
            data=data,
            initial_cash=50_000.0,
            params={
                "n_simulations": 5000,
                "n_regimes": 3,
                "slope_threshold": 5.0,
                "signal_strength_min": 0.5,
                "position_size_pct": 0.10,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05,
            },
        )
        result = bt.run()

        assert "metrics" in result
        assert "equity_curve" in result
        assert "trades" in result
        assert "params" in result

        metrics = result["metrics"]
        assert "total_return_pct" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics
        assert "n_trades" in metrics

        equity_curve = result["equity_curve"]
        assert len(equity_curve) >= 2
        assert equity_curve[0] == 50_000.0

        # All equity values must be positive (cash never goes negative in PaperBroker)
        assert all(v > 0 for v in equity_curve)

    def test_walk_forward_returns_list_of_results(self):
        """Walk-forward backtest returns a list of per-window result dicts."""
        data = _make_multi_ticker_data(n=80)
        bt = Backtester(
            data=data,
            initial_cash=100_000.0,
            params={
                "n_simulations": 3000,
                "n_regimes": 2,
                "slope_threshold": 8.0,
                "signal_strength_min": 0.5,
                "position_size_pct": 0.10,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05,
            },
        )
        results = bt.run_walk_forward(train_days=40, test_days=20)

        assert isinstance(results, list)

        for r in results:
            assert "metrics" in r
            assert "equity_curve" in r
            assert "trades" in r
            assert "window_start" in r
            assert "window_end" in r
            assert isinstance(r["trades"], list)
            assert len(r["equity_curve"]) >= 1

    def test_backtest_max_drawdown_non_positive(self):
        """Max drawdown should be <= 0 (expressed as negative percentage or zero)."""
        data = _make_trending_data(n=80)
        bt = Backtester(data=data, initial_cash=50_000.0, params={"n_simulations": 3000})
        result = bt.run()
        dd = result["metrics"].get("max_drawdown_pct", 0)
        assert dd <= 0.0, f"Max drawdown should be <= 0, got {dd}"

    def test_insufficient_data_walk_forward_returns_empty(self):
        """Walk-forward with too few bars returns empty list gracefully."""
        np.random.seed(1)
        prices = 100.0 * np.cumprod(1 + np.random.normal(0.001, 0.01, 10))
        idx = pd.date_range("2023-01-02", periods=10, freq="B", tz="UTC")
        df = pd.DataFrame({"Close": prices, "Open": prices, "High": prices, "Low": prices, "Volume": np.ones(10)}, index=idx)
        data = {"TEST": df}

        bt = Backtester(data=data, initial_cash=10_000.0)
        results = bt.run_walk_forward(train_days=126, test_days=42)
        assert results == []
