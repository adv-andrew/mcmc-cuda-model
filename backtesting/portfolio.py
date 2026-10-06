"""
Portfolio building blocks: a trend-following core sleeve and sleeve blending.

The dip strategy is invested only ~14% of the time. A *core* sleeve puts
the idle capital to work with a published, well-tested rule (Faber, 2007):
hold the index while its last month-end close is above its 10-month moving
average, otherwise hold T-bills. The dip strategy then runs as an overlay
sleeve, and the two are combined with fixed weights.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from backtesting.options_pricing import daily_risk_free
from trading.features import higher_tf_close, higher_tf_series


def faber_signal(close: pd.Series, months: int = 10) -> pd.Series:
    """True on days whose last *completed* month-end close is above the
    ``months``-month SMA of month-end closes (no lookahead)."""
    m_close = higher_tf_close(close, "ME")
    m_sma = higher_tf_series(close, "ME", lambda s: s.rolling(months).mean())
    return (m_close > m_sma).fillna(False)


def trend_core_returns(
    total_return_px: pd.Series,
    signal: pd.Series,
    switch_cost_bps: float = 5.0,
    cash_yield: bool = True,
) -> pd.Series:
    """Daily returns of a sleeve that holds the asset when ``signal`` (known
    at the prior close) is True and T-bills otherwise.

    ``total_return_px`` should include dividends (adjusted close). A cost of
    ``switch_cost_bps`` is charged on each change of position.
    """
    px = total_return_px.dropna()
    asset_ret = px.pct_change().fillna(0.0)
    held = signal.reindex(px.index).fillna(False).astype(bool).shift(1, fill_value=False)
    rf = daily_risk_free(px.index) if cash_yield else pd.Series(0.0, index=px.index)
    ret = pd.Series(np.where(held, asset_ret, rf), index=px.index)
    switches = held.astype(int).diff().abs().fillna(0.0)
    return ret - switches * switch_cost_bps / 10_000.0


def blend(returns: Dict[str, pd.Series], weights: Dict[str, float],
          initial: float = 100_000.0) -> pd.Series:
    """Equity curve of sleeves combined at fixed weights, rebalanced daily."""
    frame = pd.DataFrame(returns).dropna()
    w = pd.Series(weights)[frame.columns]
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"weights must sum to 1, got {w.sum():.4f}")
    combined = (frame * w).sum(axis=1)
    return initial * (1 + combined).cumprod()


def equity_to_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


# ----------------------------------------------------------------------
# Single-account simulation (verification of the blended approximation)
# ----------------------------------------------------------------------

def simulate_core_overlay(
    core_px: pd.Series,
    core_signal: pd.Series,
    features: Dict[str, pd.DataFrame],
    atm_iv: Dict[str, pd.Series],
    entries: Dict[str, pd.Series],
    exit_signals: Dict[str, pd.Series],
    structure,
    exits,
    core_weight: float = 0.85,
    risk_per_trade: float = 0.06,
    max_concurrent: int = 2,
    priority: Dict[str, pd.Series] | None = None,
    price_scale: Dict[str, pd.Series] | None = None,
    switch_cost_bps: float = 5.0,
    start: str | None = None,
    end: str | None = None,
    initial: float = 100_000.0,
    skew=None,
    cost_mult: float = 1.0,
    whole_contracts: bool = False,
):
    """One brokerage account: core ETF shares + option premiums paid from cash.

    - The core holds ``core_weight`` of total equity in ``core_px`` (a total-
      return price) while ``core_signal`` (known at the prior close) is True,
      rebalanced on the first trading day of each month and whenever the
      signal flips; otherwise that capital sits in T-bills.
    - Each option entry spends ``risk_per_trade`` x total equity in premium,
      only if enough uninvested cash is available.
    - All uninvested cash earns the T-bill rate.
    - ``skew`` / ``cost_mult`` allow harsher option-pricing assumptions.

    Returns ``(equity, trades)`` where trades is a DataFrame.
    """
    from backtesting.core_overlay_engine import CoreOverlayEngine, EngineState

    symbols = list(features)
    dates = pd.DatetimeIndex(core_px.dropna().index)
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates < pd.Timestamp(end)]
    cpx = core_px.reindex(dates).ffill().to_numpy(float)
    csig = core_signal.reindex(dates).fillna(False).astype(bool).shift(
        1, fill_value=False).to_numpy()
    close = {s: features[s]["close"].reindex(dates).to_numpy(float) for s in symbols}
    iv = {s: atm_iv[s].reindex(dates).ffill().to_numpy(float) for s in symbols}
    ent = {s: entries[s].reindex(dates).fillna(False).to_numpy(bool) for s in symbols}
    ext = {s: exit_signals[s].reindex(dates).fillna(False).to_numpy(bool) for s in symbols}
    prio = ({s: priority[s].reindex(dates).fillna(0.0).to_numpy() for s in symbols}
            if priority else None)
    scale = {s: (price_scale[s].reindex(dates).ffill().bfill().to_numpy(float)
                 if price_scale and s in price_scale else np.ones(len(dates)))
             for s in symbols}

    engine = CoreOverlayEngine(structure, exits, core_weight, risk_per_trade, max_concurrent,
                               switch_cost_bps, skew, cost_mult, whole_contracts)
    state = EngineState(cash=initial)
    last_i = len(dates) - 1
    for i, date in enumerate(dates):
        engine.step(
            state, date, cpx[i], bool(csig[i]),
            {s: close[s][i] for s in symbols}, {s: iv[s][i] for s in symbols},
            {s: bool(ent[s][i]) for s in symbols}, {s: bool(ext[s][i]) for s in symbols},
            {s: prio[s][i] for s in symbols} if prio else None,
            {s: scale[s][i] for s in symbols}, allow_entries=i < last_i)
    eq = pd.Series([v for _, v in state.equity], index=dates, name="equity")
    return eq, pd.DataFrame(state.trades)
