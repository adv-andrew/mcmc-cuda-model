"""PositionManager — tracks open positions and realised/unrealised P&L."""

from __future__ import annotations

from typing import Any


class PositionManager:
    """In-memory position book with P&L tracking.

    Each position is stored as a dict::

        {
            "ticker": str,
            "quantity": float,   # total open quantity (signed: + long, - short)
            "avg_price": float,  # average entry price
        }

    Closed trades are appended to an internal list as::

        {
            "ticker": str,
            "quantity": float,   # quantity closed (positive)
            "entry_price": float,
            "exit_price": float,
            "pnl": float,
            "open_timestamp": Any,
            "close_timestamp": Any,
        }
    """

    def __init__(self) -> None:
        # ticker -> position dict
        self._positions: dict[str, dict] = {}
        # list of closed trade dicts
        self._trades: list[dict] = {}
        self._trades = []

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def open_position(
        self,
        ticker: str,
        quantity: float,
        price: float,
        timestamp: Any = None,
    ) -> dict:
        """Open or add to a position.

        If a position already exists for ``ticker`` the average entry price is
        recalculated using a weighted average.

        Returns the updated position dict.
        """
        if quantity <= 0:
            raise ValueError(f"quantity must be positive; got {quantity}")
        if price <= 0:
            raise ValueError(f"price must be positive; got {price}")

        if ticker in self._positions:
            pos = self._positions[ticker]
            total_qty = pos["quantity"] + quantity
            pos["avg_price"] = (
                pos["avg_price"] * pos["quantity"] + price * quantity
            ) / total_qty
            pos["quantity"] = total_qty
            pos["open_timestamp"] = pos.get("open_timestamp")
        else:
            self._positions[ticker] = {
                "ticker": ticker,
                "quantity": quantity,
                "avg_price": price,
                "open_timestamp": timestamp,
            }

        return dict(self._positions[ticker])

    def close_position(
        self,
        ticker: str,
        quantity: float,
        price: float,
        timestamp: Any = None,
    ) -> dict:
        """Close (reduce) an open position and record the trade.

        Parameters
        ----------
        ticker:
            Instrument to close.
        quantity:
            Number of units to close.  Must be <= current open quantity.
        price:
            Exit price.
        timestamp:
            Close timestamp (any type accepted).

        Returns
        -------
        dict — the closed trade record including realised ``pnl``.

        Raises
        ------
        KeyError if no open position exists for ``ticker``.
        ValueError if ``quantity`` exceeds the open position size.
        """
        if ticker not in self._positions:
            raise KeyError(f"No open position for ticker '{ticker}'.")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive; got {quantity}")
        if price <= 0:
            raise ValueError(f"price must be positive; got {price}")

        pos = self._positions[ticker]
        if quantity > pos["quantity"]:
            raise ValueError(
                f"Cannot close {quantity} units; only {pos['quantity']} open."
            )

        pnl = (price - pos["avg_price"]) * quantity

        trade = {
            "ticker": ticker,
            "quantity": quantity,
            "entry_price": pos["avg_price"],
            "exit_price": price,
            "pnl": round(pnl, 8),
            "open_timestamp": pos.get("open_timestamp"),
            "close_timestamp": timestamp,
        }
        self._trades.append(trade)

        remaining = pos["quantity"] - quantity
        if remaining == 0.0:
            del self._positions[ticker]
        else:
            pos["quantity"] = remaining
            # avg_price does not change on a partial close

        return trade

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_position(self, ticker: str) -> dict | None:
        """Return the open position for ``ticker``, or ``None``."""
        pos = self._positions.get(ticker)
        return dict(pos) if pos is not None else None

    def get_all_positions(self) -> dict[str, dict]:
        """Return a copy of all open positions keyed by ticker."""
        return {t: dict(p) for t, p in self._positions.items()}

    def get_trades(self) -> list[dict]:
        """Return all closed trade records."""
        return list(self._trades)

    def get_total_realized_pnl(self) -> float:
        """Sum of P&L across all closed trades."""
        return sum(t["pnl"] for t in self._trades)

    def get_unrealized_pnl(self, current_prices: dict[str, float]) -> dict[str, float]:
        """Calculate unrealised P&L for each open position.

        Parameters
        ----------
        current_prices:
            Mapping of ticker -> current market price.

        Returns
        -------
        dict mapping ticker -> unrealised P&L float.
        Missing tickers (no current price provided) are skipped.
        """
        result: dict[str, float] = {}
        for ticker, pos in self._positions.items():
            if ticker in current_prices:
                upnl = (current_prices[ticker] - pos["avg_price"]) * pos["quantity"]
                result[ticker] = round(upnl, 8)
        return result
