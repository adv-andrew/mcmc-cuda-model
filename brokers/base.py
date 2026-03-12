from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Order:
    id: str
    ticker: str
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    status: str  # 'pending', 'filled', 'cancelled'
    timestamp: datetime
    filled_price: Optional[float] = None

    @property
    def value(self) -> float:
        p = self.filled_price if self.filled_price is not None else self.price
        return self.quantity * p


@dataclass
class Position:
    ticker: str
    quantity: int
    avg_price: float
    current_price: float
    opened_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis


class BaseBroker(ABC):

    @abstractmethod
    def buy(self, ticker: str, quantity: int, price: float) -> Order:
        pass

    @abstractmethod
    def sell(self, ticker: str, quantity: int, price: float) -> Order:
        pass

    @abstractmethod
    def get_position(self, ticker: str) -> Optional[Position]:
        pass

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        pass

    @abstractmethod
    def get_equity(self) -> float:
        pass

    @abstractmethod
    def get_cash(self) -> float:
        pass

    @abstractmethod
    def get_orders(self) -> list[Order]:
        pass
