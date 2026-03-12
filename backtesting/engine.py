"""Backtesting engine with walk-forward validation for MCMC Trading System."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from brokers.paper import PaperBroker
from trading.indicator import MCMCIndicator
from trading.risk_manager import RiskManager
from backtesting.metrics import calculate_metrics

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS = {
    "slope_threshold": 10.0,
    "n_regimes": 3,
    "n_simulations": 25000,
    "signal_strength_min": 0.6,
    "position_size_pct": 0.10,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.05,
}


class Backtester:
    """
    Event-driven backtester that iterates through daily bars, checks exits
    first (stop loss / take profit), then evaluates entry signals via MCMC.

    Parameters
    ----------
    data:
        Mapping of ticker -> DataFrame with at least a ``Close`` column and a
        DatetimeIndex sorted in ascending order.
    initial_cash:
        Starting portfolio cash.
    params:
        Override dict for strategy parameters (merged with defaults).
    slippage_pct:
        Fractional slippage applied per fill in PaperBroker.
    market_impact_pct:
        Fractional market-impact cost per fill in PaperBroker.
    """

    def __init__(
        self,
        data: dict,
        initial_cash: float = 100_000.0,
        params: Optional[dict] = None,
        slippage_pct: float = 0.0002,
        market_impact_pct: float = 0.0001,
    ) -> None:
        self.data = data
        self.initial_cash = initial_cash
        self.slippage_pct = slippage_pct
        self.market_impact_pct = market_impact_pct

        # Merge provided params over defaults
        self.params = {**_DEFAULT_PARAMS, **(params or {})}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """
        Run a single backtest over the supplied date range.

        Parameters
        ----------
        start_date:
            ISO date string for backtest start (inclusive). ``None`` uses
            the earliest date available in ``data``.
        end_date:
            ISO date string for backtest end (inclusive). ``None`` uses the
            latest date available in ``data``.

        Returns
        -------
        dict with keys: metrics, equity_curve, trades, params
        """
        broker = PaperBroker(
            initial_cash=self.initial_cash,
            slippage_pct=self.slippage_pct,
            market_impact_pct=self.market_impact_pct,
        )
        indicator = MCMCIndicator(
            n_simulations=int(self.params["n_simulations"]),
            n_regimes=int(self.params["n_regimes"]),
            slope_threshold=float(self.params["slope_threshold"]),
        )
        risk = RiskManager(
            broker=broker,
            position_size_pct=float(self.params["position_size_pct"]),
            stop_loss_pct=float(self.params["stop_loss_pct"]),
            take_profit_pct=float(self.params["take_profit_pct"]),
        )

        # Build unified sorted date index
        dates = self._build_date_index(start_date, end_date)

        equity_curve: list[float] = [self.initial_cash]
        completed_trades: list[dict] = []
        # Track entry price per ticker for PnL calculation
        entry_prices: dict[str, float] = {}

        for current_date in dates:
            for ticker, df in self.data.items():
                # Slice history available up to and including current_date
                history = self._history_up_to(df, current_date)
                if history.empty or len(history) < 5:
                    continue

                current_price = float(history["Close"].iloc[-1])
                broker.set_price(ticker, current_price)

                # --- Check exits first ---
                position = broker.get_position(ticker)
                if position is not None and position.quantity > 0:
                    if risk.should_stop_loss(ticker) or risk.should_take_profit(ticker):
                        reason = (
                            "stop_loss"
                            if risk.should_stop_loss(ticker)
                            else "take_profit"
                        )
                        try:
                            broker.sell(ticker, position.quantity, current_price)
                            entry_px = entry_prices.pop(ticker, position.avg_price)
                            pnl = (current_price - entry_px) * position.quantity
                            completed_trades.append(
                                {
                                    "ticker": ticker,
                                    "date": str(current_date),
                                    "side": "sell",
                                    "quantity": position.quantity,
                                    "price": current_price,
                                    "pnl": pnl,
                                    "reason": reason,
                                    "duration_minutes": 390,  # approx 1 day
                                }
                            )
                        except ValueError as exc:
                            logger.debug("Exit failed for %s: %s", ticker, exc)
                        continue  # skip entry check this bar

                # --- Entry check ---
                if not risk.can_trade():
                    continue

                try:
                    signal = indicator.generate_signal(ticker, history, "1d")
                except Exception as exc:
                    logger.debug("Signal failed for %s on %s: %s", ticker, current_date, exc)
                    continue

                strength_ok = signal["signal_strength"] >= float(
                    self.params["signal_strength_min"]
                )
                if signal["suggested_action"] == "BUY" and strength_ok:
                    qty = risk.calculate_position_size(ticker, current_price, signal["signal_strength"])
                    if qty > 0:
                        try:
                            broker.buy(ticker, qty, current_price)
                            entry_prices[ticker] = current_price
                        except ValueError as exc:
                            logger.debug("Buy failed for %s: %s", ticker, exc)

            equity_curve.append(broker.get_equity())

        metrics = calculate_metrics(np.array(equity_curve), completed_trades)
        return {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "trades": completed_trades,
            "params": self.params,
        }

    def run_walk_forward(
        self,
        train_days: int = 126,
        test_days: int = 42,
    ) -> list[dict]:
        """
        Run a walk-forward backtest, returning one result dict per test window.

        Parameters
        ----------
        train_days:
            Number of trading days in each training window (currently used
            to determine the minimum lookback before testing begins).
        test_days:
            Number of trading days in each out-of-sample test window.

        Returns
        -------
        list[dict]
            Each element has the same shape as the dict returned by ``run()``,
            plus ``window_start`` and ``window_end`` keys.
        """
        all_dates = self._build_date_index()
        if len(all_dates) < train_days + test_days:
            logger.warning(
                "Insufficient data for walk-forward: need %d days, have %d.",
                train_days + test_days,
                len(all_dates),
            )
            return []

        results: list[dict] = []
        # Slide the test window forward by test_days each iteration
        start_idx = train_days
        while start_idx + test_days <= len(all_dates):
            test_start = all_dates[start_idx]
            test_end = all_dates[min(start_idx + test_days - 1, len(all_dates) - 1)]

            result = self.run(
                start_date=str(test_start.date()),
                end_date=str(test_end.date()),
            )
            result["window_start"] = str(test_start.date())
            result["window_end"] = str(test_end.date())
            results.append(result)

            start_idx += test_days

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_date_index(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DatetimeIndex:
        """Return a sorted, deduplicated union of all dates across tickers."""
        all_idx: list = []
        for df in self.data.values():
            if not df.empty:
                all_idx.extend(df.index.normalize().unique().tolist())

        if not all_idx:
            return pd.DatetimeIndex([])

        combined = pd.DatetimeIndex(all_idx).unique().sort_values()

        if start_date:
            start_ts = pd.Timestamp(start_date)
            # Make tz-aware if index is tz-aware
            if combined.tz is not None and start_ts.tz is None:
                start_ts = start_ts.tz_localize("UTC")
            combined = combined[combined >= start_ts]

        if end_date:
            end_ts = pd.Timestamp(end_date)
            if combined.tz is not None and end_ts.tz is None:
                end_ts = end_ts.tz_localize("UTC")
            combined = combined[combined <= end_ts]

        return combined

    @staticmethod
    def _history_up_to(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        """Return all rows in ``df`` with index <= ``date``."""
        idx = df.index
        if idx.tz is not None and date.tz is None:
            date = date.tz_localize("UTC")
        elif idx.tz is None and date.tz is not None:
            date = date.tz_localize(None)
        # normalize intraday index to date comparison via end-of-day boundary
        return df.loc[idx.normalize() <= date.normalize()]
