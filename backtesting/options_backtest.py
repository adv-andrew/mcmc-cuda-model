"""
Event-driven options swing backtester.

Walks forward day by day over one or more underlyings. When an entry signal
fires at a close, it opens a defined-risk option structure priced with the
Black-Scholes / VIX-implied-vol / skew / spread model from
``backtesting.options_pricing``. Every open position is marked to its
liquidation value (mid minus closing costs) each day, so the equity curve
and drawdowns reflect what the account would actually have shown.

Exits are evaluated at each close, in order: expiry, stop loss, profit
target, signal-based exit (after ``min_hold`` days), and ``max_hold``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.options_pricing import (
    CostModel,
    Leg,
    OptionPosition,
    SkewModel,
    close_value,
    open_position,
    daily_risk_free,
    position_return,
    profit_fraction,
    risk_free_rate,
    round_strike,
    strike_for_delta,
    strike_increment,
)

CALENDAR_DAYS = 365.0


@dataclass(frozen=True)
class StructureSpec:
    """Which option structure to open on a bullish (+1) or bearish (-1) signal.

    kinds
    -----
    ``long_call``           buy one call at ``long_delta``
    ``call_debit_spread``   buy ``long_delta`` call, sell ``short_delta`` call
    ``put_credit_spread``   sell ``short_delta`` put, buy put ``width_pct`` lower
    ``long_put``, ``put_debit_spread``, ``call_credit_spread``: bearish mirrors
    """

    kind: str = "put_credit_spread"
    dte: int = 14
    long_delta: float = 0.50
    short_delta: float = 0.30
    width_pct: float = 0.02

    def label(self) -> str:
        if self.kind in ("put_credit_spread", "call_credit_spread"):
            return f"{self.kind}(d{self.short_delta:.2f},w{self.width_pct:.1%},{self.dte}d)"
        if self.kind in ("call_debit_spread", "put_debit_spread"):
            return f"{self.kind}(d{self.long_delta:.2f}/{self.short_delta:.2f},{self.dte}d)"
        return f"{self.kind}(d{self.long_delta:.2f},{self.dte}d)"


@dataclass(frozen=True)
class ExitRules:
    min_hold: int = 1                 # trading days before a signal exit may fire
    max_hold: int = 7                 # trading days; forced exit
    profit_target: Optional[float] = 0.50  # fraction of max profit (ROR for single long options)
    stop_loss: Optional[float] = None      # return on capital at risk, e.g. -0.60
    signal_exit: bool = True          # exit when the underlying recovers (see backtester)


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int
    structure: str
    entry_spot: float
    exit_spot: float
    atm_iv: float
    days_held: int
    ret_on_risk: float
    risk_dollars: float
    pnl: float
    exit_reason: str
    signal: str = ""


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity: pd.Series
    config: dict = field(default_factory=dict)

    # ---- summary statistics -------------------------------------------
    def trade_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def stats(self, start: Optional[str] = None, end: Optional[str] = None) -> dict:
        tf = self.trade_frame()
        eq = self.equity
        if start is not None:
            eq = eq.loc[start:]
            tf = tf[tf.entry_date >= pd.Timestamp(start)] if len(tf) else tf
        if end is not None:
            eq = eq.loc[:end]
            tf = tf[tf.entry_date < pd.Timestamp(end)] if len(tf) else tf
        return summarize(tf, eq)


def summarize(tf: pd.DataFrame, eq: pd.Series) -> dict:
    out: dict = {"trades": int(len(tf))}
    if len(tf):
        r = tf.ret_on_risk
        wins, losses = r[r > 0], r[r <= 0]
        out.update(
            win_rate=float((r > 0).mean()),
            avg_ret=float(r.mean()),
            avg_win=float(wins.mean()) if len(wins) else 0.0,
            avg_loss=float(losses.mean()) if len(losses) else 0.0,
            profit_factor=float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
            avg_days=float(tf.days_held.mean()),
            worst=float(r.min()),
        )
    if len(eq) > 2:
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        total = eq.iloc[-1] / eq.iloc[0] - 1.0
        daily = eq.pct_change().dropna()
        excess = daily - daily_risk_free(eq.index).reindex(daily.index)
        dd = eq / eq.cummax() - 1.0
        sd = daily.std()
        out.update(
            total_return=float(total),
            cagr=float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else -1.0,
            max_drawdown=float(dd.min()),
            sharpe=float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
            # Sharpe on returns above T-bills: the right measure once idle
            # cash earns interest (otherwise the cash yield inflates Sharpe).
            sharpe_excess=float(excess.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        )
    return out


def build_legs(
    spec: StructureSpec,
    direction: int,
    spot: float,
    t_years: float,
    atm_iv: float,
    rate: float,
    skew: SkewModel,
    is_etf: bool,
    price_scale: float = 1.0,
) -> List[Leg]:
    """Strikes for ``spec``. ``price_scale`` converts ``spot`` to the actually
    traded price (split-unadjusted) so the listed strike grid is realistic."""
    inc = strike_increment(spot * price_scale, is_etf) / price_scale
    kind = spec.kind
    if direction < 0:  # map bullish kinds to their bearish mirrors
        kind = {
            "long_call": "long_put",
            "call_debit_spread": "put_debit_spread",
            "put_credit_spread": "call_credit_spread",
        }.get(kind, kind)

    def k_for(delta: float, is_call: bool) -> float:
        # Solve for the strike using the skew-adjusted IV at a first-pass strike
        k0 = strike_for_delta(spot, delta, t_years, atm_iv, rate, is_call, inc)
        iv = skew.iv(atm_iv, spot, k0, t_years)
        return strike_for_delta(spot, delta, t_years, iv, rate, is_call, inc)

    if kind == "long_call":
        return [Leg(True, k_for(spec.long_delta, True), +1)]
    if kind == "long_put":
        return [Leg(False, k_for(spec.long_delta, False), +1)]
    if kind == "call_debit_spread":
        kl = k_for(spec.long_delta, True)
        ks = max(k_for(spec.short_delta, True), kl + inc)
        return [Leg(True, kl, +1), Leg(True, ks, -1)]
    if kind == "put_debit_spread":
        kl = max(k_for(spec.long_delta, False), 2 * inc)
        ks = min(k_for(spec.short_delta, False), kl - inc)
        return [Leg(False, kl, +1), Leg(False, ks, -1)]
    if kind == "put_credit_spread":
        ks = max(k_for(spec.short_delta, False), 2 * inc)
        kl = min(round_strike(ks - spec.width_pct * spot, inc), ks - inc)
        return [Leg(False, ks, -1), Leg(False, kl, +1)]
    if kind == "call_credit_spread":
        ks = k_for(spec.short_delta, True)
        kl = max(round_strike(ks + spec.width_pct * spot, inc), ks + inc)
        return [Leg(True, ks, -1), Leg(True, kl, +1)]
    raise ValueError(f"Unknown structure kind {spec.kind}")


@dataclass
class _Open:
    symbol: str
    pos: OptionPosition
    entry_i: int
    entry_date: pd.Timestamp
    direction: int
    risk_dollars: float
    units: float  # number of "per-share" units held (contracts x 100)
    atm_iv: float
    entry_spot: float
    signal: str
    last_mark: float = float("nan")  # last liquidation value (for days without a quote)


class OptionsSwingBacktester:
    """Portfolio-level options swing backtest over aligned daily data."""

    def __init__(
        self,
        structure: StructureSpec,
        exits: ExitRules,
        risk_per_trade: float = 0.05,
        max_concurrent: int = 3,
        initial_equity: float = 100_000.0,
        skew: Optional[SkewModel] = None,
        cost_multiplier: float = 1.0,
        etfs: tuple = ("SPY", "QQQ", "IWM", "DIA"),
        tight_cost_symbols: Optional[tuple] = None,
        cash_yield: bool = False,
    ) -> None:
        """``etfs`` get the $1 ETF strike grid. ``tight_cost_symbols`` (default:
        same as ``etfs``) get the tight ETF bid/ask tier; everything else pays
        the wider stock tier. ``cash_yield`` accrues T-bill interest on cash
        not committed to option premium."""
        self.structure = structure
        self.exits = exits
        self.risk_per_trade = risk_per_trade
        self.max_concurrent = max_concurrent
        self.initial_equity = initial_equity
        self.skew = skew or SkewModel()
        self.cost_multiplier = cost_multiplier
        self.etfs = set(etfs)
        self.tight = set(etfs if tight_cost_symbols is None else tight_cost_symbols)
        self.cash_yield = cash_yield

    def _costs(self, symbol: str, price_scale: float = 1.0) -> CostModel:
        """Costs in the (possibly split-adjusted) price units of the backtest."""
        base = CostModel.etf() if symbol in self.tight else CostModel.stock()
        c = base.scaled(self.cost_multiplier)
        if price_scale != 1.0:  # $ minimums apply at the traded price
            c = CostModel(c.min_half_spread / price_scale, c.pct_half_spread,
                          c.commission_per_contract / price_scale)
        return c

    def run(
        self,
        features: Dict[str, pd.DataFrame],
        atm_iv: Dict[str, pd.Series],
        entries: Dict[str, pd.Series],
        exit_signals: Optional[Dict[str, pd.Series]] = None,
        direction: Dict[str, pd.Series] | int = 1,
        priority: Optional[Dict[str, pd.Series]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        price_scale: Optional[Dict[str, pd.Series]] = None,
    ) -> BacktestResult:
        """Run the backtest.

        Parameters
        ----------
        features: per-symbol frames with at least ``close``.
        atm_iv: per-symbol ATM implied vol series.
        entries: per-symbol boolean entry series (signal at that close).
        exit_signals: per-symbol boolean "underlying has recovered" series.
        direction: +1/-1 constant or per-symbol series.
        priority: per-symbol score; higher entries are taken first when
            several symbols fire on the same day and slots are limited.
        price_scale: per-symbol factor converting backtest prices to the
            actually traded price (see ``market_data.split_factor``).
        """
        symbols = list(features)
        dates = sorted(set().union(*[features[s].index for s in symbols]))
        dates = pd.DatetimeIndex(dates)
        if start:
            dates = dates[dates >= pd.Timestamp(start)]
        if end:
            dates = dates[dates < pd.Timestamp(end)]

        close = {s: features[s]["close"].reindex(dates) for s in symbols}
        iv = {s: atm_iv[s].reindex(dates).ffill() for s in symbols}
        ent = {s: entries[s].reindex(dates).fillna(False).astype(bool) for s in symbols}
        ext = (
            {s: exit_signals[s].reindex(dates).fillna(False).astype(bool) for s in symbols}
            if exit_signals else None
        )
        prio = (
            {s: priority[s].reindex(dates).fillna(0.0) for s in symbols}
            if priority else None
        )
        scale = {
            s: (price_scale[s].reindex(dates).ffill().bfill().to_numpy()
                if price_scale and s in price_scale else np.ones(len(dates)))
            for s in symbols
        }

        cash = self.initial_equity
        open_pos: List[_Open] = []
        trades: List[Trade] = []
        equity = np.empty(len(dates))
        rf = daily_risk_free(dates).to_numpy() if self.cash_yield else np.zeros(len(dates))

        for i, date in enumerate(dates):
            cash *= 1.0 + rf[i]
            # ---- 1. manage open positions --------------------------------
            still_open: List[_Open] = []
            for op in open_pos:
                spot = close[op.symbol].iloc[i]
                if np.isnan(spot):
                    still_open.append(op)
                    continue
                costs = self._costs(op.symbol, scale[op.symbol][i])
                value = close_value(op.pos, spot, date, iv[op.symbol].iloc[i], self.skew, costs)
                r = position_return(op.pos, value)
                held = i - op.entry_i
                reason = None
                if (op.pos.expiry - date).days <= 0:
                    reason = "expiry"
                elif self.exits.stop_loss is not None and r <= self.exits.stop_loss:
                    reason = "stop"
                elif (self.exits.profit_target is not None
                      and profit_fraction(op.pos, value) >= self.exits.profit_target):
                    reason = "target"
                elif (
                    self.exits.signal_exit and ext is not None and held >= self.exits.min_hold
                    and ext[op.symbol].iloc[i]
                ):
                    reason = "signal"
                elif held >= self.exits.max_hold:
                    reason = "time"
                if reason:
                    pnl = r * op.risk_dollars
                    cash += op.risk_dollars + pnl  # release reserved risk capital
                    trades.append(Trade(
                        symbol=op.symbol, entry_date=op.entry_date, exit_date=date,
                        direction=op.direction, structure=self.structure.label(),
                        entry_spot=op.entry_spot, exit_spot=float(spot), atm_iv=op.atm_iv,
                        days_held=held, ret_on_risk=float(r), risk_dollars=op.risk_dollars,
                        pnl=float(pnl), exit_reason=reason, signal=op.signal,
                    ))
                else:
                    still_open.append(op)
            open_pos = still_open

            # ---- 2. mark to market --------------------------------------
            mtm = 0.0
            for op in open_pos:
                spot = close[op.symbol].iloc[i]
                if np.isnan(spot):
                    mtm += op.risk_dollars if np.isnan(op.last_mark) else op.last_mark
                    continue
                v = close_value(op.pos, spot, date, iv[op.symbol].iloc[i], self.skew,
                                self._costs(op.symbol, scale[op.symbol][i]))
                op.last_mark = op.risk_dollars * (1.0 + position_return(op.pos, v))
                mtm += op.last_mark
            equity_now = cash + mtm
            equity[i] = equity_now

            # ---- 3. new entries -------------------------------------------
            if i >= len(dates) - 1:
                continue
            held_syms = {op.symbol for op in open_pos}
            cands = [s for s in symbols if ent[s].iloc[i] and s not in held_syms
                     and not np.isnan(close[s].iloc[i]) and not np.isnan(iv[s].iloc[i])]
            if prio:
                cands.sort(key=lambda s: prio[s].iloc[i], reverse=True)
            for s in cands:
                if len(open_pos) >= self.max_concurrent:
                    break
                d = direction if isinstance(direction, int) else int(direction[s].reindex(dates).iloc[i])
                spot = float(close[s].iloc[i])
                a_iv = float(iv[s].iloc[i])
                t = self.structure.dte / CALENDAR_DAYS
                legs = build_legs(self.structure, d, spot, t, a_iv, risk_free_rate(date),
                                  self.skew, s in self.etfs, scale[s][i])
                pos = open_position(legs, spot, date, self.structure.dte, a_iv, self.skew,
                                    self._costs(s, scale[s][i]))
                risk = self.risk_per_trade * equity_now
                if risk > cash or pos.max_loss <= 0:
                    continue
                cash -= risk
                open_pos.append(_Open(
                    symbol=s, pos=pos, entry_i=i, entry_date=date, direction=d,
                    risk_dollars=risk, units=risk / pos.max_loss, atm_iv=a_iv,
                    entry_spot=spot, signal="",
                ))

        eq = pd.Series(equity, index=dates, name="equity")
        return BacktestResult(trades=trades, equity=eq, config={
            "structure": self.structure.label(),
            "exits": self.exits.__dict__,
            "risk_per_trade": self.risk_per_trade,
            "max_concurrent": self.max_concurrent,
            "cost_multiplier": self.cost_multiplier,
        })


def monte_carlo_equity(
    trade_returns: np.ndarray,
    trades_per_year: float,
    risk_per_trade: float,
    years: int = 1,
    n_sims: int = 10_000,
    seed: int = 0,
) -> dict:
    """Bootstrap the trade sequence to estimate the spread of outcomes.

    Resamples per-trade returns on risk (with replacement), compounds them at
    ``risk_per_trade`` of equity, and reports the distribution of the
    ``years``-year return and of the worst peak-to-trough drawdown. This is a
    sizing tool: it answers "how bad can a year plausibly get at this size?",
    which a single historical equity curve cannot.
    """
    rng = np.random.default_rng(seed)
    n = max(int(round(trades_per_year * years)), 1)
    draws = rng.choice(np.asarray(trade_returns, dtype=float), size=(n_sims, n))
    growth = np.cumprod(1.0 + risk_per_trade * draws, axis=1)
    final = growth[:, -1] - 1.0
    peaks = np.maximum.accumulate(np.concatenate([np.ones((n_sims, 1)), growth], axis=1), axis=1)
    dd = (np.concatenate([np.ones((n_sims, 1)), growth], axis=1) / peaks - 1.0).min(axis=1)
    return {
        "median_return": float(np.median(final)),
        "p05_return": float(np.percentile(final, 5)),
        "p95_return": float(np.percentile(final, 95)),
        "prob_loss": float(np.mean(final < 0)),
        "median_max_dd": float(np.median(dd)),
        "p95_max_dd": float(np.percentile(dd, 5)),
    }


# ----------------------------------------------------------------------
# Per-trade simulation (research / robustness testing)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PricingScenario:
    """Pricing / execution assumptions for stress testing.

    ``entry_iv_mult`` scales implied vol *only* when the position is opened,
    modelling paying up for short-dated vol on dip days. Delays shift the
    fill to that many trading days after the signal / exit trigger.
    """

    skew: SkewModel = field(default_factory=SkewModel)
    entry_iv_mult: float = 1.0
    cost_mult: float = 1.0
    entry_delay: int = 0
    exit_delay: int = 0


def simulate_trades(
    close: pd.Series,
    atm_iv: pd.Series,
    exit_sig: pd.Series,
    spec: StructureSpec,
    exits: ExitRules,
    scenario: Optional[PricingScenario] = None,
    base_costs: Optional[CostModel] = None,
    entry_dates: Optional[pd.DatetimeIndex] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    is_etf: bool = True,
    price_scale: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Simulate one independent bullish trade per entry date (no portfolio).

    With ``entry_dates=None`` a trade is simulated for *every* day in
    ``[start, end)``, which is what placebo tests need. Returns, indexed by
    signal date: ``ret`` (return on capital at risk), ``held`` (trading days),
    ``und_ret`` (underlying return over the same window) and ``reason``.
    """
    sc = scenario or PricingScenario()
    costs = (base_costs or CostModel.etf()).scaled(sc.cost_mult)
    dates = close.index
    px = close.to_numpy(dtype=float)
    ivs = atm_iv.reindex(dates).ffill().to_numpy(dtype=float)
    ext = exit_sig.reindex(dates).fillna(False).to_numpy(dtype=bool)
    scale = (price_scale.reindex(dates).ffill().bfill().to_numpy(dtype=float)
             if price_scale is not None else np.ones(len(dates)))
    lo = dates.searchsorted(pd.Timestamp(start)) if start else 0
    hi = dates.searchsorted(pd.Timestamp(end)) if end else len(dates)
    last_ok = len(dates) - exits.max_hold - sc.exit_delay - sc.entry_delay - 1
    if entry_dates is None:
        idx = range(lo, min(hi, last_ok))
    else:
        pos_idx = dates.get_indexer(pd.DatetimeIndex(entry_dates))
        idx = [i for i in pos_idx if i >= lo and i < min(hi, last_ok) and i >= 0]

    rows = []
    for i in idx:
        e = i + sc.entry_delay
        if np.isnan(ivs[e]) or np.isnan(px[e]):
            continue
        spot, date = px[e], dates[e]
        c_in = CostModel(costs.min_half_spread / scale[e], costs.pct_half_spread,
                         costs.commission_per_contract / scale[e])
        legs = build_legs(spec, +1, spot, spec.dte / CALENDAR_DAYS, ivs[e] * sc.entry_iv_mult,
                          risk_free_rate(date), sc.skew, is_etf, scale[e])
        pos = open_position(legs, spot, date, spec.dte, ivs[e] * sc.entry_iv_mult, sc.skew, c_in)
        j, reason = e, "time"
        while True:
            j += 1
            held = j - e
            c_j = CostModel(costs.min_half_spread / scale[j], costs.pct_half_spread,
                            costs.commission_per_contract / scale[j])
            value = close_value(pos, px[j], dates[j], ivs[j], sc.skew, c_j)
            if (pos.expiry - dates[j]).days <= 0:
                reason = "expiry"
                break
            if (exits.stop_loss is not None
                    and position_return(pos, value) <= exits.stop_loss):
                reason = "stop"
                break
            if (exits.profit_target is not None
                    and profit_fraction(pos, value) >= exits.profit_target):
                reason = "target"
                break
            if exits.signal_exit and held >= exits.min_hold and ext[j]:
                reason = "signal"
            if reason == "signal" or held >= exits.max_hold:
                if sc.exit_delay:
                    j += sc.exit_delay
                    c_j = CostModel(costs.min_half_spread / scale[j], costs.pct_half_spread,
                                    costs.commission_per_contract / scale[j])
                    value = close_value(pos, px[j], dates[j], ivs[j], sc.skew, c_j)
                break
        rows.append((dates[i], position_return(pos, value), j - e, px[j] / spot - 1.0, reason))
    return pd.DataFrame(rows, columns=["date", "ret", "held", "und_ret", "reason"]).set_index("date")


def non_overlapping_entries(signal: pd.Series, held: pd.Series) -> pd.DatetimeIndex:
    """Entry dates a single-position-per-symbol strategy would actually take.

    ``held`` maps each candidate date to its holding period (trading days);
    a new entry is allowed only after the previous trade has exited.
    """
    sig_dates = signal.index[signal.fillna(False).to_numpy(dtype=bool)]
    taken, busy_until = [], -1
    pos = held.index
    for d in sig_dates:
        k = pos.get_indexer([d])[0]
        if k < 0 or k <= busy_until:
            continue
        taken.append(d)
        busy_until = k + int(held.iloc[k])
    return pd.DatetimeIndex(taken)
