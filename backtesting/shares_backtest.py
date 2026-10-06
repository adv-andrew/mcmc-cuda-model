"""
Shares (ETF) swing backtester for the dip-buying signal.

The research showed the timing edge is strong on the underlying itself
across 16 ETFs and three eras, while option bid/ask and implied-vol premium
consume most of it outside SPY/QQQ/IWM. This backtester trades the same
entries and exits in shares: it buys a fixed fraction of equity at the
signal-day close, exits on the same rules, charges a per-side cost in basis
points, and marks positions to market every day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.options_backtest import BacktestResult, ExitRules, Trade


@dataclass
class _Pos:
    symbol: str
    entry_i: int
    entry_date: pd.Timestamp
    entry_px: float
    dollars: float  # capital committed at entry


class SharesSwingBacktester:
    """Long-only shares backtest with a fixed number of equal-sized slots."""

    def __init__(
        self,
        exits: ExitRules,
        position_frac: float = 0.25,
        max_concurrent: int = 4,
        cost_bps: float = 3.0,
        initial_equity: float = 100_000.0,
    ) -> None:
        self.exits = exits
        self.position_frac = position_frac
        self.max_concurrent = max_concurrent
        self.cost = cost_bps / 10_000.0
        self.initial_equity = initial_equity

    def run(
        self,
        close: Dict[str, pd.Series],
        entries: Dict[str, pd.Series],
        exit_signals: Dict[str, pd.Series],
        priority: Optional[Dict[str, pd.Series]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> BacktestResult:
        symbols = list(close)
        dates = pd.DatetimeIndex(sorted(set().union(*[close[s].index for s in symbols])))
        if start:
            dates = dates[dates >= pd.Timestamp(start)]
        if end:
            dates = dates[dates < pd.Timestamp(end)]
        px = {s: close[s].reindex(dates).ffill().to_numpy(dtype=float) for s in symbols}
        live = {s: close[s].reindex(dates).notna().to_numpy() for s in symbols}
        ent = {s: entries[s].reindex(dates).fillna(False).to_numpy(dtype=bool) for s in symbols}
        ext = {s: exit_signals[s].reindex(dates).fillna(False).to_numpy(dtype=bool)
               for s in symbols}
        prio = ({s: priority[s].reindex(dates).fillna(0.0).to_numpy() for s in symbols}
                if priority else None)

        cash = self.initial_equity
        positions: List[_Pos] = []
        trades: List[Trade] = []
        equity = np.empty(len(dates))
        for i, date in enumerate(dates):
            keep = []
            for p in positions:
                held = i - p.entry_i
                if not live[p.symbol][i]:
                    keep.append(p)
                    continue
                reason = None
                if self.exits.signal_exit and held >= self.exits.min_hold and ext[p.symbol][i]:
                    reason = "signal"
                elif held >= self.exits.max_hold:
                    reason = "time"
                if reason is None:
                    keep.append(p)
                    continue
                gross = p.dollars * px[p.symbol][i] / p.entry_px
                proceeds = gross * (1 - self.cost)
                cash += proceeds
                ret = proceeds / p.dollars - 1.0
                trades.append(Trade(
                    symbol=p.symbol, entry_date=p.entry_date, exit_date=date, direction=1,
                    structure="shares", entry_spot=p.entry_px, exit_spot=px[p.symbol][i],
                    atm_iv=float("nan"), days_held=held, ret_on_risk=ret,
                    risk_dollars=p.dollars, pnl=proceeds - p.dollars, exit_reason=reason,
                ))
            positions = keep
            mtm = sum(p.dollars * px[p.symbol][i] / p.entry_px for p in positions)
            equity[i] = cash + mtm

            if i >= len(dates) - 1:
                continue
            held_syms = {p.symbol for p in positions}
            cands = [s for s in symbols if ent[s][i] and live[s][i] and s not in held_syms]
            if prio:
                cands.sort(key=lambda s: prio[s][i], reverse=True)
            for s in cands:
                if len(positions) >= self.max_concurrent:
                    break
                dollars = self.position_frac * equity[i]
                cost = dollars * self.cost
                if dollars + cost > cash:
                    break
                cash -= dollars + cost
                positions.append(_Pos(s, i, date, px[s][i], dollars))

        eq = pd.Series(equity, index=dates, name="equity")
        return BacktestResult(trades=trades, equity=eq, config={
            "structure": "shares", "position_frac": self.position_frac,
            "max_concurrent": self.max_concurrent, "cost_bps": self.cost * 10_000,
        })
