"""
Step-by-step engine for portfolio mode (trend core + options dip overlay).

One engine drives both the historical simulation
(``backtesting.portfolio.simulate_core_overlay``) and the live paper-trading
ledger (``trading.paper_ledger``), so what runs live is exactly what was
backtested. All state lives in :class:`EngineState`, which round-trips
through JSON.

Per trading day, in order:
  1. accrue T-bill interest on cash (calendar days since the last step)
  2. age and value open option positions; close those whose exit rule fires
  3. rebalance the core on the first trading day of a month or when the
     core signal flips (core holds ``core_weight`` of equity, or nothing)
  4. record equity
  5. open new option positions (premium = ``risk_per_trade`` x equity)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from backtesting.options_backtest import ExitRules, StructureSpec, build_legs
from backtesting.options_pricing import (
    CostModel,
    Leg,
    OptionPosition,
    SkewModel,
    close_value,
    open_position,
    position_return,
    profit_fraction,
    risk_free_rate,
)


@dataclass
class EngineState:
    cash: float
    core_units: float = 0.0
    last_date: Optional[str] = None
    last_core_in: Optional[bool] = None
    positions: List[dict] = field(default_factory=list)
    trades: List[dict] = field(default_factory=list)
    equity: List[list] = field(default_factory=list)  # [date, value]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EngineState":
        return cls(**d)


def _pos_to_dict(pos: OptionPosition) -> dict:
    return {
        "legs": [[leg.is_call, float(leg.strike), int(leg.qty)] for leg in pos.legs],
        "expiry": str(pd.Timestamp(pos.expiry).date()),
        "entry_value": float(pos.entry_value),
        "entry_cost": float(pos.entry_cost),
        "max_loss": float(pos.max_loss),
        "max_profit": None if math.isinf(pos.max_profit) else float(pos.max_profit),
    }


def _pos_from_dict(d: dict) -> OptionPosition:
    return OptionPosition(
        legs=[Leg(bool(c), float(k), int(q)) for c, k, q in d["legs"]],
        expiry=pd.Timestamp(d["expiry"]),
        entry_value=d["entry_value"],
        entry_cost=d["entry_cost"],
        max_loss=d["max_loss"],
        max_profit=float("inf") if d["max_profit"] is None else d["max_profit"],
    )


class CoreOverlayEngine:
    def __init__(
        self,
        structure: StructureSpec,
        exits: ExitRules,
        core_weight: float = 0.85,
        risk_per_trade: float = 0.06,
        max_concurrent: int = 2,
        switch_cost_bps: float = 5.0,
        skew: Optional[SkewModel] = None,
        cost_mult: float = 1.0,
        whole_contracts: bool = False,
        max_overshoot: float = 1.5,
    ) -> None:
        """``whole_contracts`` buys an integer number of option contracts:
        as many as fit in the target premium, or one if a single contract
        costs at most ``max_overshoot`` x the target (otherwise the trade is
        skipped). Fractional contracts (the default) match the research."""
        self.whole_contracts = whole_contracts
        self.max_overshoot = max_overshoot
        self.structure = structure
        self.exits = exits
        self.core_weight = core_weight
        self.risk_per_trade = risk_per_trade
        self.max_concurrent = max_concurrent
        self.switch_cost = switch_cost_bps / 10_000.0
        self.skew = skew or SkewModel()
        self.cost_mult = cost_mult

    def _costs(self, scale: float) -> CostModel:
        c = CostModel.etf().scaled(self.cost_mult)
        return CostModel(c.min_half_spread / scale, c.pct_half_spread,
                         c.commission_per_contract / scale)

    def step(
        self,
        state: EngineState,
        date: pd.Timestamp,
        core_px: float,
        core_in: bool,
        closes: Dict[str, float],
        ivs: Dict[str, float],
        entries: Dict[str, bool],
        exit_flags: Dict[str, bool],
        priority: Optional[Dict[str, float]] = None,
        scales: Optional[Dict[str, float]] = None,
        allow_entries: bool = True,
    ) -> EngineState:
        """Advance ``state`` by one trading day (mutates and returns it)."""
        date = pd.Timestamp(date)
        if core_px is None or math.isnan(core_px) or core_px <= 0:
            raise ValueError(f"invalid core price {core_px!r} on {date.date()}")
        scales = scales or {}
        last = pd.Timestamp(state.last_date) if state.last_date else None

        # 1. interest on cash
        if last is not None:
            state.cash *= 1.0 + risk_free_rate(date) * (date - last).days / 365.0

        # 2. option exits
        still, opt_value = [], 0.0
        for p in state.positions:
            p["days_held"] += 1
            s = p["symbol"]
            px = closes.get(s, float("nan"))
            if px is None or math.isnan(px):  # no quote today: keep the last mark
                still.append(p)
                opt_value += p.get("last_value", p["risk"])
                continue
            pos = _pos_from_dict(p["pos"])
            v = close_value(pos, px, date, ivs[s], self.skew, self._costs(scales.get(s, 1.0)))
            r = position_return(pos, v)
            held = p["days_held"]
            reason = None
            if (pos.expiry - date).days <= 0:
                reason = "expiry"
            elif self.exits.stop_loss is not None and r <= self.exits.stop_loss:
                reason = "stop"
            elif (self.exits.profit_target is not None
                  and profit_fraction(pos, v) >= self.exits.profit_target):
                reason = "target"
            elif self.exits.signal_exit and held >= self.exits.min_hold and exit_flags.get(s):
                reason = "signal"
            elif held >= self.exits.max_hold:
                reason = "time"
            if reason:
                state.cash += p["risk"] * (1 + r)
                state.trades.append({
                    "id": p["id"], "symbol": s, "entry_date": p["entry_date"],
                    "exit_date": str(date.date()), "days_held": held,
                    "entry_spot": p["entry_spot"], "exit_spot": float(px),
                    "model_debit": p["model_debit"], "risk": float(p["risk"]),
                    "ret_on_risk": float(r),
                    "pnl": float(p["risk"] * r), "exit_reason": reason,
                    "actual_debit": p.get("actual_debit"),
                })
            else:
                p["last_value"] = float(p["risk"] * (1 + r))
                still.append(p)
                opt_value += p["risk"] * (1 + r)
        state.positions = still
        total = state.cash + state.core_units * core_px + opt_value

        # 3. core rebalance
        month_start = last is None or date.month != last.month or date.year != last.year
        flip = state.last_core_in is not None and core_in != state.last_core_in
        if month_start or flip:
            target = (self.core_weight * total / core_px) if core_in else 0.0
            trade_val = (target - state.core_units) * core_px
            state.cash -= trade_val + abs(trade_val) * self.switch_cost
            state.core_units = target
        total = state.cash + state.core_units * core_px + opt_value
        state.equity.append([str(date.date()), float(total)])

        # 5. entries
        if allow_entries:
            held_syms = {p["symbol"] for p in state.positions}
            cands = [s for s, on in entries.items() if on and s not in held_syms
                     and not math.isnan(closes.get(s, float("nan")))
                     and not math.isnan(ivs.get(s, float("nan")))]
            if priority:
                cands.sort(key=lambda s: priority.get(s, 0.0), reverse=True)
            for s in cands:
                if len(state.positions) >= self.max_concurrent:
                    break
                risk = self.risk_per_trade * total
                if risk > state.cash:
                    break
                spot, sc = closes[s], scales.get(s, 1.0)
                legs = build_legs(self.structure, 1, spot, self.structure.dte / 365.0, ivs[s],
                                  risk_free_rate(date), self.skew, True, sc)
                pos = open_position(legs, spot, date, self.structure.dte, ivs[s], self.skew,
                                    self._costs(sc))
                per_contract = pos.max_loss * sc * 100.0  # $ at the traded price
                if self.whole_contracts:
                    n = math.floor(risk / per_contract)
                    if n == 0 and per_contract <= self.max_overshoot * risk:
                        n = 1
                    if n == 0 or n * per_contract > state.cash:
                        continue
                    risk = n * per_contract
                state.cash -= risk
                state.positions.append({
                    "id": f"{s}-{date.date()}", "symbol": s, "entry_date": str(date.date()),
                    "entry_spot": float(spot), "entry_iv": float(ivs[s]),
                    "model_debit": float(pos.max_loss), "risk": float(risk),
                    "contracts": float(risk / per_contract),
                    "days_held": 0, "pos": _pos_to_dict(pos), "last_value": float(risk),
                })

        state.last_date = str(date.date())
        state.last_core_in = bool(core_in)
        return state
