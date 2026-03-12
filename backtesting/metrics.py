"""Performance metrics calculation for MCMC trading system."""

import numpy as np
from typing import Optional


def calculate_metrics(
    equity_curve: np.ndarray,
    trades: list[dict],
    risk_free_rate: float = 0.0
) -> dict:
    """
    Calculate comprehensive performance metrics for a trading strategy.

    Args:
        equity_curve: Array of portfolio values over time
        trades: List of trade dictionaries with keys 'pnl' and 'duration_minutes'
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        Dictionary containing:
        - total_return_pct: Total return percentage
        - sharpe_ratio: Annualized Sharpe ratio (252 trading days)
        - max_drawdown_pct: Maximum drawdown percentage
        - win_rate: Percentage of winning trades
        - profit_factor: Ratio of gross profit to gross loss
        - n_trades: Number of trades
        - avg_win: Average winning trade PnL
        - avg_loss: Average losing trade PnL
        - avg_duration_minutes: Average trade duration in minutes
    """

    metrics = {}

    # Handle edge case: empty equity curve
    if equity_curve is None or len(equity_curve) == 0:
        return {
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "n_trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_duration_minutes": 0.0,
        }

    equity_curve = np.asarray(equity_curve, dtype=np.float64)

    # Total Return
    if len(equity_curve) > 0 and equity_curve[0] != 0:
        total_return_pct = ((equity_curve[-1] - equity_curve[0]) / equity_curve[0]) * 100
    else:
        total_return_pct = 0.0

    metrics["total_return_pct"] = total_return_pct

    # Sharpe Ratio (annualized)
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / equity_curve[:-1]

        if len(returns) > 0:
            excess_returns = returns - (risk_free_rate / 252)
            daily_sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
            sharpe_ratio = daily_sharpe * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    metrics["sharpe_ratio"] = sharpe_ratio

    # Maximum Drawdown
    max_drawdown_pct = _calculate_max_drawdown(equity_curve)
    metrics["max_drawdown_pct"] = max_drawdown_pct

    # Trade Statistics
    n_trades = len(trades)
    metrics["n_trades"] = n_trades

    if n_trades > 0:
        pnls = np.array([trade["pnl"] for trade in trades])
        winning_trades = pnls[pnls > 0]
        losing_trades = pnls[pnls < 0]

        # Win Rate
        win_rate = (len(winning_trades) / n_trades) * 100 if n_trades > 0 else 0.0
        metrics["win_rate"] = win_rate

        # Profit Factor
        gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
        gross_loss = np.sum(np.abs(losing_trades)) if len(losing_trades) > 0 else 0.0
        profit_factor = gross_profit / (gross_loss + 1e-8) if gross_loss > 0 else (
            1.0 if gross_profit > 0 else 0.0
        )
        metrics["profit_factor"] = profit_factor

        # Average Win/Loss
        metrics["avg_win"] = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
        metrics["avg_loss"] = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0

        # Average Duration
        durations = np.array([trade.get("duration_minutes", 0.0) for trade in trades])
        metrics["avg_duration_minutes"] = np.mean(durations) if len(durations) > 0 else 0.0
    else:
        metrics["win_rate"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["avg_win"] = 0.0
        metrics["avg_loss"] = 0.0
        metrics["avg_duration_minutes"] = 0.0

    return metrics


def _calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown as a percentage.

    Args:
        equity_curve: Array of portfolio values over time

    Returns:
        Maximum drawdown as a percentage (negative value)
    """
    if len(equity_curve) == 0:
        return 0.0

    # Compute the running maximum
    running_max = np.maximum.accumulate(equity_curve)

    # Compute drawdown
    drawdown = (equity_curve - running_max) / (running_max + 1e-8)

    # Return maximum drawdown as percentage
    max_drawdown = np.min(drawdown) * 100

    return max_drawdown
