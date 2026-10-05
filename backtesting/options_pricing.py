"""
Option pricing and cost model used by the options backtests.

Why this exists
---------------
The earlier backtests priced an option as ``intrinsic + 0.4 * realized_vol *
sqrt(T)``. That ignores three things that dominate short-hold option P&L:

1. Options trade on *implied* volatility, which usually sits above the
   volatility that subsequently realizes (the volatility risk premium).
   Buyers pay it; sellers collect it.
2. Implied volatility is skewed: OTM puts are priced at a higher IV than ATM.
3. Every leg crosses a bid/ask spread and pays commission, twice.

This module prices with Black-Scholes, takes ATM implied vol from the VIX
(scaled per underlying by its realized-vol ratio to SPY), applies a simple
skew, and charges spread + commission on every leg at entry and exit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

_SQRT2 = math.sqrt(2.0)


def _ncdf(x: float) -> float:
    """Standard normal CDF via erf (much faster than scipy for scalars)."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


TRADING_DAYS = 252
CALENDAR_DAYS = 365.0

# Approximate 1-3 month T-bill yields by year; option values over a few days
# are barely sensitive to this, but it keeps carry roughly right.
_RATE_BY_YEAR = {
    2010: 0.001, 2011: 0.001, 2012: 0.001, 2013: 0.001, 2014: 0.001,
    2015: 0.002, 2016: 0.004, 2017: 0.010, 2018: 0.019, 2019: 0.021,
    2020: 0.004, 2021: 0.001, 2022: 0.020, 2023: 0.052, 2024: 0.051,
    2025: 0.043, 2026: 0.038,
}


def risk_free_rate(date: pd.Timestamp) -> float:
    """Approximate short-term risk-free rate for ``date``."""
    return _RATE_BY_YEAR.get(pd.Timestamp(date).year, 0.03)


# ----------------------------------------------------------------------
# Black-Scholes
# ----------------------------------------------------------------------

def bs_price(
    spot: float, strike: float, t_years: float, vol: float, rate: float, is_call: bool
) -> float:
    """Black-Scholes price of a European option (no dividends).

    At or past expiry (``t_years <= 0``) returns intrinsic value.
    """
    if t_years <= 0 or vol <= 0:
        intrinsic = spot - strike if is_call else strike - spot
        return max(intrinsic, 0.0)
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc = math.exp(-rate * t_years)
    if is_call:
        return spot * _ncdf(d1) - strike * disc * _ncdf(d2)
    return strike * disc * _ncdf(-d2) - spot * _ncdf(-d1)


def bs_delta(
    spot: float, strike: float, t_years: float, vol: float, rate: float, is_call: bool
) -> float:
    """Black-Scholes delta."""
    if t_years <= 0 or vol <= 0:
        if is_call:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t_years) / (
        vol * math.sqrt(t_years)
    )
    return _ncdf(d1) if is_call else _ncdf(d1) - 1.0


def strike_for_delta(
    spot: float,
    target_delta: float,
    t_years: float,
    vol: float,
    rate: float,
    is_call: bool,
    increment: float = 1.0,
) -> float:
    """Strike (rounded to ``increment``) whose BS delta is closest to ``target_delta``.

    ``target_delta`` is given as a positive number for both calls and puts
    (e.g. 0.30 means a 30-delta call or a -30-delta put).
    """
    sqrt_t = math.sqrt(max(t_years, 1e-8))
    # Call delta = N(d1); put delta = N(d1) - 1  =>  N(d1) = 1 - |put delta|
    n_d1 = target_delta if is_call else 1.0 - target_delta
    d1 = norm.ppf(min(max(n_d1, 1e-4), 1 - 1e-4))
    raw = spot * math.exp(-(d1 * vol * sqrt_t) + (rate + 0.5 * vol * vol) * t_years)
    return round_strike(raw, increment)


def round_strike(strike: float, increment: float) -> float:
    return max(increment, round(strike / increment) * increment)


def strike_increment(spot: float, is_etf: bool) -> float:
    """Typical listed strike spacing near the money."""
    if is_etf:
        return 1.0
    if spot < 25:
        return 0.5
    if spot < 200:
        return 1.0 if spot < 100 else 2.5
    return 5.0


# ----------------------------------------------------------------------
# Implied-vol model
# ----------------------------------------------------------------------

@dataclass
class SkewModel:
    """Linear skew in standardized moneyness z = ln(K/S) / (atm_iv * sqrt(T)).

    ``put_slope`` lifts IV for strikes below spot (z < 0); ``call_slope``
    lowers IV for strikes above spot. Defaults roughly match SPX-style skew
    (a 25-delta put trades ~20% rich to ATM, a 25-delta call ~7% cheap).
    """

    put_slope: float = 0.30
    call_slope: float = 0.10
    floor_mult: float = 0.60
    cap_mult: float = 2.50

    def iv(self, atm_iv: float, spot: float, strike: float, t_years: float) -> float:
        z = math.log(strike / spot) / (atm_iv * math.sqrt(max(t_years, 1.0 / CALENDAR_DAYS)))
        mult = 1.0 - (self.put_slope * z if z < 0 else self.call_slope * z)
        return atm_iv * min(max(mult, self.floor_mult), self.cap_mult)


def atm_iv_series(
    df: pd.DataFrame,
    vix: pd.DataFrame,
    spy: pd.DataFrame,
    vix_to_atm: float = 0.90,
    rv_floor_mult: float = 1.00,
) -> pd.Series:
    """Estimate a daily 30-day ATM implied-vol series for an underlying.

    ``ATM_IV = vix_to_atm * VIX/100 * beta_vol`` where ``beta_vol`` is the
    trailing-year median ratio of the underlying's 20-day realized vol to
    SPY's. VIX overstates SPX ATM vol slightly because it includes the skew,
    hence ``vix_to_atm`` < 1. The result is floored at ``rv_floor_mult`` x
    the underlying's 10-day realized vol so that IV never prices a stock
    calmer than it is currently moving.
    """
    close = df["Close"]
    rv20 = np.log(close).diff().rolling(20).std() * math.sqrt(TRADING_DAYS)
    rv10 = np.log(close).diff().rolling(10).std() * math.sqrt(TRADING_DAYS)
    spy_rv20 = (
        np.log(spy["Close"]).diff().rolling(20).std() * math.sqrt(TRADING_DAYS)
    ).reindex(df.index).ffill()
    ratio = (rv20 / spy_rv20).rolling(TRADING_DAYS, min_periods=60).median()
    ratio = ratio.clip(0.8, 4.0).fillna(1.0)

    vix_close = vix["Close"].reindex(df.index).ffill(limit=3) / 100.0
    iv = vix_to_atm * vix_close * ratio
    # When VIX is missing (e.g. the mirror lags by more than a few days) fall
    # back to realized vol plus a typical implied-over-realized premium.
    iv = iv.fillna(rv20 * 1.2)
    return np.maximum(iv, rv_floor_mult * rv10.fillna(0.0)).rename("atm_iv")


# ----------------------------------------------------------------------
# Transaction costs
# ----------------------------------------------------------------------

@dataclass
class CostModel:
    """Per-leg, per-transaction costs, all in $ per share of underlying.

    half_spread = max(min_half_spread, pct_half_spread * mid)
    commission  = commission_per_contract / 100
    """

    min_half_spread: float = 0.01
    pct_half_spread: float = 0.01
    commission_per_contract: float = 0.65

    def leg_cost(self, mid: float) -> float:
        return max(self.min_half_spread, self.pct_half_spread * mid) + (
            self.commission_per_contract / 100.0
        )

    @classmethod
    def etf(cls) -> "CostModel":
        return cls(min_half_spread=0.01, pct_half_spread=0.01)

    @classmethod
    def stock(cls) -> "CostModel":
        return cls(min_half_spread=0.02, pct_half_spread=0.02)

    def scaled(self, factor: float) -> "CostModel":
        return CostModel(
            self.min_half_spread * factor,
            self.pct_half_spread * factor,
            self.commission_per_contract,
        )


# ----------------------------------------------------------------------
# Multi-leg positions
# ----------------------------------------------------------------------

@dataclass
class Leg:
    is_call: bool
    strike: float
    qty: int  # +1 long, -1 short (per share; x100 for contracts)


@dataclass
class OptionPosition:
    """A multi-leg option position opened at ``entry_date`` with expiry in calendar days."""

    legs: List[Leg]
    expiry: pd.Timestamp
    entry_value: float = 0.0  # signed mid value at entry (+ debit / - credit)
    entry_cost: float = 0.0   # spread + commission paid at entry
    max_loss: float = 0.0     # capital at risk per share
    max_profit: float = float("inf")  # best-case P&L per share (inf for single long options)
    meta: dict = field(default_factory=dict)

    def mid_value(
        self, spot: float, date: pd.Timestamp, atm_iv: float, skew: SkewModel
    ) -> float:
        t = max((self.expiry - date).days, 0) / CALENDAR_DAYS
        rate = risk_free_rate(date)
        total = 0.0
        for leg in self.legs:
            vol = skew.iv(atm_iv, spot, leg.strike, max(t, 1.0 / CALENDAR_DAYS))
            total += leg.qty * bs_price(spot, leg.strike, t, vol, rate, leg.is_call)
        return total

    def leg_mids(
        self, spot: float, date: pd.Timestamp, atm_iv: float, skew: SkewModel
    ) -> List[float]:
        t = max((self.expiry - date).days, 0) / CALENDAR_DAYS
        rate = risk_free_rate(date)
        return [
            bs_price(
                spot, leg.strike, t,
                skew.iv(atm_iv, spot, leg.strike, max(t, 1.0 / CALENDAR_DAYS)),
                rate, leg.is_call,
            )
            for leg in self.legs
        ]

    def transaction_cost(self, mids: Sequence[float], costs: CostModel) -> float:
        return sum(costs.leg_cost(m) * abs(leg.qty) for leg, m in zip(self.legs, mids))


def width_of(legs: Sequence[Leg]) -> float:
    strikes = [leg.strike for leg in legs]
    return max(strikes) - min(strikes)


def open_position(
    legs: List[Leg],
    spot: float,
    date: pd.Timestamp,
    dte: int,
    atm_iv: float,
    skew: SkewModel,
    costs: CostModel,
) -> OptionPosition:
    """Price a new position at mid, charge entry costs, and compute capital at risk.

    Capital at risk is the debit paid for debit structures, and
    ``width - credit`` for vertical credit spreads.
    """
    pos = OptionPosition(legs=legs, expiry=pd.Timestamp(date) + pd.Timedelta(days=dte))
    mids = pos.leg_mids(spot, date, atm_iv, skew)
    pos.entry_value = sum(leg.qty * m for leg, m in zip(legs, mids))
    pos.entry_cost = pos.transaction_cost(mids, costs)
    if pos.entry_value > 0:  # debit
        pos.max_loss = pos.entry_value + pos.entry_cost
        if len(legs) == 2:
            pos.max_profit = max(width_of(legs) - pos.max_loss, 1e-6)
    else:  # credit vertical: worst case is the width minus net credit received
        net_credit = -pos.entry_value - pos.entry_cost
        pos.max_loss = max(width_of(legs) - net_credit, 1e-6)
        pos.max_profit = max(net_credit, 1e-6)
    return pos


def profit_fraction(pos: OptionPosition, exit_value: float) -> float:
    """P&L as a fraction of the structure's max profit.

    For single long options (unbounded upside) this falls back to the
    return on capital at risk.
    """
    pnl = exit_value - pos.entry_value - pos.entry_cost
    if math.isinf(pos.max_profit):
        return pnl / pos.max_loss
    return pnl / pos.max_profit


def close_value(
    pos: OptionPosition,
    spot: float,
    date: pd.Timestamp,
    atm_iv: float,
    skew: SkewModel,
    costs: CostModel,
) -> float:
    """Net $/share received when closing ``pos`` (negative if paying to close).

    At expiry legs settle at intrinsic value with no spread charged.
    """
    mids = pos.leg_mids(spot, date, atm_iv, skew)
    value = sum(leg.qty * m for leg, m in zip(pos.legs, mids))
    if (pos.expiry - pd.Timestamp(date)).days <= 0:
        return value
    return value - pos.transaction_cost(mids, costs)


def position_return(pos: OptionPosition, exit_value: float) -> float:
    """Return on capital at risk for a closed position."""
    pnl = exit_value - pos.entry_value - pos.entry_cost
    return pnl / pos.max_loss
