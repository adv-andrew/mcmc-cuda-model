"""
Catalog of swing-trading entry signals drawn from well-known trader research.

Each signal is a function of the feature frame from ``trading.features``
returning a boolean Series (True = enter at that day's close). ``direction``
is +1 for bullish setups and -1 for bearish ones.

Families
--------
- Mean reversion in an uptrend (Connors RSI(2), IBS, N-day lows,
  consecutive down closes, Bollinger stretch)
- Volatility / fear (VIX spikes vs its own average, VIX at 1-year highs)
- Trend / breakout (Donchian, 52-week highs, NR7 breakouts, 12-1 momentum)
- Calendar (turn-of-the-month)
- Multi-timeframe confluence (weekly/monthly trend + daily trigger)
- Bearish mirrors (overbought in a downtrend)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import pandas as pd

Rule = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class SignalSpec:
    name: str
    direction: int
    family: str
    rule: Rule
    note: str = ""


def _b(x) -> pd.Series:
    return x.fillna(False).astype(bool)


SIGNALS: Dict[str, SignalSpec] = {}


def register(name: str, direction: int, family: str, note: str = ""):
    def deco(fn: Rule) -> Rule:
        SIGNALS[name] = SignalSpec(name, direction, family, lambda f: _b(fn(f)), note)
        return fn
    return deco


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------

@register("always_long", +1, "baseline", "Every day; the unconditional drift")
def _always(f):
    return pd.Series(True, index=f.index)


@register("above200", +1, "baseline", "Close above 200-day SMA")
def _above200(f):
    return f.above200


# ----------------------------------------------------------------------
# Mean reversion in an uptrend
# ----------------------------------------------------------------------

@register("rsi2_lt10_up", +1, "mean_rev", "Connors: RSI(2)<10, close>SMA200")
def _rsi2_10(f):
    return (f.rsi2 < 10) & f.above200


@register("rsi2_lt5_up", +1, "mean_rev", "Connors: RSI(2)<5, close>SMA200")
def _rsi2_5(f):
    return (f.rsi2 < 5) & f.above200


@register("crsi2_lt35_up", +1, "mean_rev", "Cumulative 2-day RSI(2) < 35, close>SMA200")
def _crsi(f):
    return (f.crsi2_2d < 35) & f.above200


@register("ibs_lt02_up", +1, "mean_rev", "IBS<0.2 (close near low of day), close>SMA200")
def _ibs(f):
    return (f.ibs < 0.2) & f.above200


@register("ibs_rsi2_up", +1, "mean_rev", "IBS<0.25 and RSI(2)<15, close>SMA200")
def _ibs_rsi(f):
    return (f.ibs < 0.25) & (f.rsi2 < 15) & f.above200


@register("down3_up", +1, "mean_rev", "3+ consecutive lower closes, close>SMA200")
def _down3(f):
    return (f.streak <= -3) & f.above200


@register("double7_up", +1, "mean_rev", "Connors Double-7: 7-day closing low, close>SMA200")
def _d7(f):
    return f.low_n7 & f.above200


@register("low10_up", +1, "mean_rev", "10-day closing low, close>SMA200")
def _low10(f):
    return f.low_n10 & f.above200


@register("bb_lt_m2_up", +1, "mean_rev", "Close < lower Bollinger(20,2), close>SMA200")
def _bb(f):
    return (f.bb_z < -2) & f.above200


@register("pullback_2atr_up", +1, "mean_rev", "Close >= 2 ATR below 5-day high, close>SMA200")
def _pb(f):
    return (f.pullback_atr >= 2.0) & f.above200


@register("rsi2_lt10_any", +1, "mean_rev", "RSI(2)<10 with no trend filter")
def _rsi2_any(f):
    return f.rsi2 < 10


# ----------------------------------------------------------------------
# Volatility / fear
# ----------------------------------------------------------------------

@register("vix_spike", +1, "volatility", "VIX >= 1.15x its 10-day SMA (fear spike)")
def _vix_spike(f):
    return f.vix_ratio10 >= 1.15


@register("vix_spike_up", +1, "volatility", "VIX spike while close>SMA200")
def _vix_spike_up(f):
    return (f.vix_ratio10 >= 1.10) & f.above200


@register("vix_high_rank", +1, "volatility", "VIX in top 10% of its 1-year range")
def _vix_rank(f):
    return f.vix_pct252 >= 0.90


@register("vix_rev", +1, "volatility", "VIX >= 1.10x SMA10 yesterday and VIX falling today")
def _vix_rev(f):
    return (f.vix_ratio10.shift(1) >= 1.10) & (f.vix_chg1 < 0)


# ----------------------------------------------------------------------
# Trend / breakout
# ----------------------------------------------------------------------

@register("donchian20_high", +1, "trend", "20-day closing high (turtle-style breakout)")
def _don(f):
    return f.high_n20


@register("high252", +1, "trend", "New 52-week closing high")
def _h252(f):
    return f.high_n252


@register("nr7_up", +1, "trend", "NR7 day in an uptrend (vol contraction before expansion)")
def _nr7(f):
    return f.nr7 & f.above50 & f.above200


@register("mom_12_1_pos", +1, "trend", "12-1 month momentum positive")
def _mom(f):
    return f.mom_12_1 > 0


@register("ema_stack", +1, "trend", "EMA8>EMA21>SMA50>SMA200 (full trend stack)")
def _stack(f):
    return (f.ema8 > f.ema21) & (f.ema21 > f.sma50) & (f.sma50 > f.sma200)


# ----------------------------------------------------------------------
# Calendar
# ----------------------------------------------------------------------

@register("turn_of_month", +1, "calendar", "Enter on 2nd-to-last trading day of the month")
def _tom(f):
    return f.tdom_rev == 2


@register("monday_after_down_week", +1, "calendar", "Friday close after a down week")
def _mon(f):
    return (f.dow == 4) & (f.mom5 < 0)


# ----------------------------------------------------------------------
# Multi-timeframe confluence
# ----------------------------------------------------------------------

@register("mtf_weekly_up_rsi2", +1, "mtf",
          "Elder triple-screen: weekly close>10wk SMA, daily RSI(2)<10")
def _mtf1(f):
    return f.weekly_up & (f.rsi2 < 10)


@register("mtf_all_up_dip", +1, "mtf",
          "Weekly & monthly uptrend, daily 3-day pullback (mom3<0) and IBS<0.3")
def _mtf2(f):
    return f.weekly_up & f.monthly_up & (f.mom3 < 0) & (f.ibs < 0.3)


@register("mtf_score4", +1, "mtf", "All four horizons (5d/21d/weekly/monthly) up")
def _mtf3(f):
    return f.mtf_score >= 4


@register("mtf_htf_up_ltf_down", +1, "mtf",
          "Monthly+weekly up but 5-day momentum negative (buy the dip)")
def _mtf4(f):
    return f.weekly_up & f.monthly_up & (f.mom5 < 0)


@register("mtf_weekly_up_vix_spike", +1, "mtf",
          "Weekly uptrend + VIX >= 1.10x 10-day SMA")
def _mtf5(f):
    return f.weekly_up & (f.vix_ratio10 >= 1.10)


# ----------------------------------------------------------------------
# Bearish mirrors
# ----------------------------------------------------------------------

@register("rsi2_gt90_down", -1, "bear", "RSI(2)>90, close<SMA200 (short the rip)")
def _bear1(f):
    return (f.rsi2 > 90) & ~f.above200


@register("ibs_gt08_down", -1, "bear", "IBS>0.8, close<SMA200")
def _bear2(f):
    return (f.ibs > 0.8) & ~f.above200


@register("donchian20_low", -1, "bear", "20-day closing low (breakdown)")
def _bear3(f):
    return f.low_n20


@register("mtf_all_down_rip", -1, "bear", "Weekly & monthly downtrend, 3-day bounce")
def _bear4(f):
    return ~f.weekly_up & ~f.monthly_up & (f.mom3 > 0)


@register("rsi2_gt90_up", -1, "bear", "RSI(2)>90 in an uptrend (fade strength)")
def _bear5(f):
    return (f.rsi2 > 90) & f.above200
