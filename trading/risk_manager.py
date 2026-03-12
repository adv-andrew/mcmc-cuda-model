from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brokers.base import BaseBroker


class RiskManager:

    def __init__(
        self,
        broker: "BaseBroker",
        position_size_pct: float = 0.10,
        max_position_pct: float = 0.25,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.05,
        daily_loss_limit_pct: float = 0.05,
    ):
        self._broker = broker
        self._position_size_pct = position_size_pct
        self._max_position_pct = max_position_pct
        self._stop_loss_pct = stop_loss_pct
        self._take_profit_pct = take_profit_pct
        self._daily_loss_limit_pct = daily_loss_limit_pct

        # Track equity at start of each trading day
        self._day_start_equity: float = broker.get_equity()
        self._tracked_date: date = date.today()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_day_start(self) -> None:
        today = date.today()
        if today != self._tracked_date:
            self._day_start_equity = self._broker.get_equity()
            self._tracked_date = today

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_position_size(
        self, ticker: str, price: float, signal_strength: float = 1.0
    ) -> int:
        """Return share quantity to trade, capped by max_position_pct."""
        if price <= 0:
            return 0
        equity = self._broker.get_equity()
        target_value = equity * self._position_size_pct * signal_strength

        # Cap so total position does not exceed max_position_pct of equity
        max_value = equity * self._max_position_pct
        existing = self._broker.get_position(ticker)
        existing_value = existing.market_value if existing else 0.0
        allowable = max(0.0, max_value - existing_value)
        trade_value = min(target_value, allowable)

        # Also limit by available cash
        cash = self._broker.get_cash()
        trade_value = min(trade_value, cash)

        return int(trade_value // price)

    def should_stop_loss(self, ticker: str) -> bool:
        """Return True if position has fallen past stop-loss threshold."""
        position = self._broker.get_position(ticker)
        if position is None or position.avg_price == 0:
            return False
        return position.unrealized_pnl_pct <= -self._stop_loss_pct

    def should_take_profit(self, ticker: str) -> bool:
        """Return True if position has risen past take-profit threshold."""
        position = self._broker.get_position(ticker)
        if position is None or position.avg_price == 0:
            return False
        return position.unrealized_pnl_pct >= self._take_profit_pct

    def is_daily_loss_exceeded(self) -> bool:
        """Return True if daily loss limit has been breached."""
        self._refresh_day_start()
        current_equity = self._broker.get_equity()
        if self._day_start_equity == 0:
            return False
        daily_return = (current_equity - self._day_start_equity) / self._day_start_equity
        return daily_return <= -self._daily_loss_limit_pct

    def can_trade(self) -> bool:
        """Return True if new trades are permitted."""
        return not self.is_daily_loss_exceeded()

    def get_exit_signals(self) -> list[dict]:
        """Return list of exit signal dicts for all open positions."""
        signals = []
        for ticker in self._broker.get_positions():
            if self.should_stop_loss(ticker):
                signals.append({"ticker": ticker, "reason": "stop_loss"})
            elif self.should_take_profit(ticker):
                signals.append({"ticker": ticker, "reason": "take_profit"})
        return signals

    def reset_day_start(self, equity: float | None = None) -> None:
        """Manually reset the day-start equity (useful for testing)."""
        self._day_start_equity = equity if equity is not None else self._broker.get_equity()
        self._tracked_date = date.today()
