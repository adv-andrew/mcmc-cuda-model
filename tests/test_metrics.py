"""Tests for performance metrics calculation module."""

import pytest
import numpy as np
from backtesting.metrics import calculate_metrics


class TestCalculateMetrics:
    """Test suite for calculate_metrics function."""

    def test_empty_trades_and_equity(self):
        """Test handling of empty trades and equity curve."""
        result = calculate_metrics(np.array([]), [])

        assert result["total_return_pct"] == 0.0
        assert result["sharpe_ratio"] == 0.0
        assert result["max_drawdown_pct"] == 0.0
        assert result["n_trades"] == 0
        assert result["win_rate"] == 0.0
        assert result["profit_factor"] == 0.0
        assert result["avg_win"] == 0.0
        assert result["avg_loss"] == 0.0
        assert result["avg_duration_minutes"] == 0.0

    def test_single_data_point(self):
        """Test handling of single data point in equity curve."""
        equity_curve = np.array([10000.0])
        result = calculate_metrics(equity_curve, [])

        assert result["total_return_pct"] == 0.0
        assert result["sharpe_ratio"] == 0.0
        assert result["max_drawdown_pct"] == 0.0
        assert result["n_trades"] == 0

    def test_basic_metrics_calculation(self):
        """Test basic metrics calculation with simple data."""
        equity_curve = np.array([10000.0, 11000.0, 11500.0, 11000.0, 12000.0])
        trades = [
            {"pnl": 500.0, "duration_minutes": 60.0},
            {"pnl": -200.0, "duration_minutes": 30.0},
            {"pnl": 700.0, "duration_minutes": 90.0},
        ]

        result = calculate_metrics(equity_curve, trades)

        assert result["n_trades"] == 3
        assert result["total_return_pct"] == pytest.approx(20.0, rel=0.01)
        assert result["win_rate"] == pytest.approx(66.67, rel=0.01)
        assert result["avg_win"] == pytest.approx(600.0, rel=0.01)
        assert result["avg_loss"] == pytest.approx(-200.0, rel=0.01)
        assert result["avg_duration_minutes"] == pytest.approx(60.0, rel=0.01)

    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        equity_curve = np.array([10000.0, 11000.0])
        trades = [
            {"pnl": 1000.0, "duration_minutes": 60.0},
            {"pnl": -500.0, "duration_minutes": 30.0},
            {"pnl": 500.0, "duration_minutes": 45.0},
        ]

        result = calculate_metrics(equity_curve, trades)

        # Gross profit: 1000 + 500 = 1500
        # Gross loss: 500
        # Profit factor: 1500 / 500 = 3.0
        assert result["profit_factor"] == pytest.approx(3.0, rel=0.01)

    def test_all_winning_trades(self):
        """Test metrics when all trades are profitable."""
        equity_curve = np.array([10000.0, 11000.0, 12000.0, 13000.0])
        trades = [
            {"pnl": 500.0, "duration_minutes": 60.0},
            {"pnl": 1000.0, "duration_minutes": 120.0},
            {"pnl": 500.0, "duration_minutes": 90.0},
        ]

        result = calculate_metrics(equity_curve, trades)

        assert result["win_rate"] == 100.0
        assert result["profit_factor"] == pytest.approx(1.0, rel=0.01)
        assert result["avg_loss"] == 0.0

    def test_all_losing_trades(self):
        """Test metrics when all trades are losers."""
        equity_curve = np.array([10000.0, 9500.0, 9000.0, 8500.0])
        trades = [
            {"pnl": -500.0, "duration_minutes": 60.0},
            {"pnl": -500.0, "duration_minutes": 120.0},
            {"pnl": -500.0, "duration_minutes": 90.0},
        ]

        result = calculate_metrics(equity_curve, trades)

        assert result["win_rate"] == 0.0
        assert result["profit_factor"] == 0.0
        assert result["avg_win"] == 0.0
        assert result["avg_loss"] == pytest.approx(-500.0, rel=0.01)

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Equity curve: 10000 -> 12000 -> 8000 -> 9000
        # Running max: 10000 -> 12000 -> 12000 -> 12000
        # Drawdown: 0 -> 0 -> -33.33% -> -25%
        # Max DD: -33.33%
        equity_curve = np.array([10000.0, 12000.0, 8000.0, 9000.0])
        result = calculate_metrics(equity_curve, [])

        assert result["max_drawdown_pct"] == pytest.approx(-33.33, rel=0.1)

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        # Create equity curve with consistent returns
        equity_curve = np.array([10000.0, 10100.0, 10201.0, 10303.01, 10406.04])
        result = calculate_metrics(equity_curve, [], risk_free_rate=0.0)

        # Sharpe ratio should be positive for consistent positive returns
        assert result["sharpe_ratio"] > 0

    def test_total_return_calculation(self):
        """Test total return percentage calculation."""
        equity_curve = np.array([10000.0, 12500.0])
        result = calculate_metrics(equity_curve, [])

        assert result["total_return_pct"] == pytest.approx(25.0, rel=0.01)

    def test_zero_initial_equity(self):
        """Test handling of zero initial equity."""
        equity_curve = np.array([0.0, 100.0])
        result = calculate_metrics(equity_curve, [])

        assert result["total_return_pct"] == 0.0

    def test_none_equity_curve(self):
        """Test handling of None equity curve."""
        result = calculate_metrics(None, [])

        assert result["total_return_pct"] == 0.0
        assert result["n_trades"] == 0

    def test_missing_duration_in_trades(self):
        """Test handling of trades missing duration_minutes key."""
        equity_curve = np.array([10000.0, 11000.0])
        trades = [
            {"pnl": 500.0},  # Missing duration_minutes
            {"pnl": 500.0, "duration_minutes": 60.0},
        ]

        result = calculate_metrics(equity_curve, trades)

        assert result["n_trades"] == 2
        assert result["avg_duration_minutes"] == pytest.approx(30.0, rel=0.01)
