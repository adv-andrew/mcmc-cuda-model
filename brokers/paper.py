import uuid
from datetime import datetime
from typing import Optional

from brokers.base import BaseBroker, Order, Position


class PaperBroker(BaseBroker):

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        slippage_pct: float = 0.0002,
        market_impact_pct: float = 0.0001,
    ):
        self._initial_cash = initial_cash
        self._slippage_pct = slippage_pct
        self._market_impact_pct = market_impact_pct
        self._cash = initial_cash
        self._positions: dict[str, Position] = {}
        self._orders: list[Order] = []
        self._prices: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Price management
    # ------------------------------------------------------------------

    def set_price(self, ticker: str, price: float) -> None:
        self._prices[ticker] = price
        if ticker in self._positions:
            self._positions[ticker].current_price = price

    # ------------------------------------------------------------------
    # Slippage helpers
    # ------------------------------------------------------------------

    def _fill_price_buy(self, price: float) -> float:
        """Buys fill slightly higher (slippage + market impact)."""
        return price * (1 + self._slippage_pct + self._market_impact_pct)

    def _fill_price_sell(self, price: float) -> float:
        """Sells fill slightly lower (slippage + market impact)."""
        return price * (1 - self._slippage_pct - self._market_impact_pct)

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def buy(self, ticker: str, quantity: int, price: float) -> Order:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        filled_price = self._fill_price_buy(price)
        cost = filled_price * quantity
        if cost > self._cash:
            raise ValueError(
                f"Insufficient cash: need {cost:.2f}, have {self._cash:.2f}"
            )
        order = Order(
            id=str(uuid.uuid4()),
            ticker=ticker,
            side="buy",
            quantity=quantity,
            price=price,
            status="filled",
            timestamp=datetime.utcnow(),
            filled_price=filled_price,
        )
        self._cash -= cost
        self._update_position_buy(ticker, quantity, filled_price)
        self._orders.append(order)
        return order

    def sell(self, ticker: str, quantity: int, price: float) -> Order:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        position = self._positions.get(ticker)
        if position is None or position.quantity < quantity:
            held = position.quantity if position else 0
            raise ValueError(
                f"Insufficient position: need {quantity}, have {held}"
            )
        filled_price = self._fill_price_sell(price)
        proceeds = filled_price * quantity
        order = Order(
            id=str(uuid.uuid4()),
            ticker=ticker,
            side="sell",
            quantity=quantity,
            price=price,
            status="filled",
            timestamp=datetime.utcnow(),
            filled_price=filled_price,
        )
        self._cash += proceeds
        self._update_position_sell(ticker, quantity)
        self._orders.append(order)
        return order

    # ------------------------------------------------------------------
    # Position bookkeeping
    # ------------------------------------------------------------------

    def _update_position_buy(self, ticker: str, quantity: int, filled_price: float) -> None:
        if ticker in self._positions:
            pos = self._positions[ticker]
            total_cost = pos.avg_price * pos.quantity + filled_price * quantity
            pos.quantity += quantity
            pos.avg_price = total_cost / pos.quantity
            pos.current_price = self._prices.get(ticker, filled_price)
        else:
            self._positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity,
                avg_price=filled_price,
                current_price=self._prices.get(ticker, filled_price),
                opened_at=datetime.utcnow(),
            )

    def _update_position_sell(self, ticker: str, quantity: int) -> None:
        pos = self._positions[ticker]
        pos.quantity -= quantity
        if pos.quantity == 0:
            del self._positions[ticker]

    # ------------------------------------------------------------------
    # BaseBroker interface
    # ------------------------------------------------------------------

    def get_position(self, ticker: str) -> Optional[Position]:
        return self._positions.get(ticker)

    def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def get_equity(self) -> float:
        return self._cash + sum(p.market_value for p in self._positions.values())

    def get_cash(self) -> float:
        return self._cash

    def get_orders(self) -> list[Order]:
        return list(self._orders)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._cash = self._initial_cash
        self._positions.clear()
        self._orders.clear()
        self._prices.clear()
