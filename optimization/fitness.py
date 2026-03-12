"""Fitness function for MCMC strategy parameter optimization."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def calculate_fitness(
    data: dict,
    params: dict,
    initial_cash: float = 100_000.0,
    min_trades: int = 100,
) -> float:
    """
    Evaluate a parameter set and return a scalar fitness score.

    The score is the total return percentage with penalties applied for:
    - Too few trades (encourages a minimally active strategy).
    - Extreme drawdown (discourages reckless risk-taking).

    Parameters
    ----------
    data:
        Mapping of ticker -> DataFrame passed directly to ``Backtester``.
    params:
        Strategy parameter dict (see ``Backtester`` / ``_DEFAULT_PARAMS``).
    initial_cash:
        Starting capital for the simulated run.
    min_trades:
        Minimum number of completed trades expected. Fewer trades incur a
        linearly scaled penalty up to -50 fitness points.

    Returns
    -------
    float
        Fitness value (higher is better). Returns ``-1000.0`` on error.
    """
    # Import here to avoid circular dependency at module load time
    from backtesting.engine import Backtester

    try:
        bt = Backtester(data=data, initial_cash=initial_cash, params=params)
        result = bt.run()
    except Exception as exc:
        logger.warning("Backtester raised during fitness evaluation: %s", exc)
        return -1000.0

    metrics = result.get("metrics", {})
    total_return = float(metrics.get("total_return_pct", 0.0))
    max_drawdown = float(metrics.get("max_drawdown_pct", 0.0))  # negative value
    n_trades = int(metrics.get("n_trades", 0))

    fitness = total_return

    # Penalty: too few trades
    if n_trades < min_trades:
        trade_shortfall = (min_trades - n_trades) / max(min_trades, 1)
        fitness -= 50.0 * trade_shortfall

    # Penalty: extreme drawdown (beyond -20 %)
    if max_drawdown < -20.0:
        excess_dd = abs(max_drawdown) - 20.0
        fitness -= excess_dd * 2.0

    return fitness
