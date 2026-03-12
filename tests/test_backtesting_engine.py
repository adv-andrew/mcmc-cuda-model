"""Minimal tests for backtesting.engine.Backtester."""

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import Backtester


def _make_data(n: int = 60, start_price: float = 100.0) -> dict:
    """Return a minimal single-ticker data dict with n daily bars."""
    np.random.seed(0)
    prices = start_price * np.cumprod(1 + np.random.normal(0.001, 0.01, n))
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": prices * 0.999,
            "High": prices * 1.005,
            "Low": prices * 0.995,
            "Close": prices,
            "Volume": np.ones(n) * 1_000_000,
        },
        index=idx,
    )
    return {"TEST": df}


class TestBacktesterInit:
    def test_default_params_populated(self):
        bt = Backtester(data=_make_data())
        assert "slope_threshold" in bt.params
        assert "stop_loss_pct" in bt.params

    def test_custom_params_override_defaults(self):
        bt = Backtester(data=_make_data(), params={"stop_loss_pct": 0.03})
        assert bt.params["stop_loss_pct"] == 0.03
        # Other defaults still present
        assert "take_profit_pct" in bt.params


class TestBacktesterRun:
    def test_run_returns_expected_keys(self):
        bt = Backtester(data=_make_data(), initial_cash=50_000)
        result = bt.run()
        assert set(result.keys()) == {"metrics", "equity_curve", "trades", "params"}

    def test_equity_curve_starts_at_initial_cash(self):
        initial_cash = 50_000.0
        bt = Backtester(data=_make_data(), initial_cash=initial_cash)
        result = bt.run()
        assert result["equity_curve"][0] == initial_cash

    def test_equity_curve_length_positive(self):
        bt = Backtester(data=_make_data())
        result = bt.run()
        assert len(result["equity_curve"]) > 1

    def test_metrics_contains_total_return(self):
        bt = Backtester(data=_make_data())
        result = bt.run()
        assert "total_return_pct" in result["metrics"]

    def test_date_range_filtering(self):
        bt = Backtester(data=_make_data(n=100))
        full = bt.run()
        partial = bt.run(start_date="2023-03-01", end_date="2023-04-30")
        # Partial run covers a shorter date range so equity curve should be shorter
        assert len(partial["equity_curve"]) <= len(full["equity_curve"])

    def test_trades_is_list(self):
        bt = Backtester(data=_make_data())
        result = bt.run()
        assert isinstance(result["trades"], list)


class TestWalkForward:
    def test_returns_list(self):
        bt = Backtester(data=_make_data(n=100))
        results = bt.run_walk_forward(train_days=40, test_days=20)
        assert isinstance(results, list)

    def test_each_result_has_window_keys(self):
        bt = Backtester(data=_make_data(n=100))
        results = bt.run_walk_forward(train_days=40, test_days=20)
        for r in results:
            assert "window_start" in r
            assert "window_end" in r

    def test_insufficient_data_returns_empty(self):
        bt = Backtester(data=_make_data(n=10))
        results = bt.run_walk_forward(train_days=50, test_days=20)
        assert results == []
