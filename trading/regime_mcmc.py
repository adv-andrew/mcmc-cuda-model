"""
Regime-switching Markov-chain Monte Carlo forecaster.

The original ``MCMCIndicator`` simulates geometric Brownian motion with the
sample-mean drift, so its median path is a deterministic extrapolation of
past drift and the simulation adds nothing beyond noise. This model instead
treats the market as a Markov chain over *observable states* and simulates
forward by resampling what historically happened next from each state:

State (10 values) = short-term stretch bucket (5) x volatility regime (2)

- stretch: 5-day return divided by the 5-day 1-sigma move implied by
  20-day realized volatility, bucketed at (-1.0, -0.33, 0.33, 1.0)
- volatility regime: 20-day realized vol above / below its 1-year median

Simulation: for each path and each step, draw a historical day ``j`` that
was in the current state, take its next-day return ``r[j+1]``, and move to
the state day ``j+1`` was in. Only days whose next-day outcome was known at
forecast time are eligible, so there is no lookahead.

The resulting distribution gives P(up), expected return and quantiles over
the holding horizon, conditioned on how stretched and how volatile the
market currently is - the two things short-horizon option trades depend on.

Validation status (scripts/research_mcmc_audit.py): as a *stand-alone
direction forecaster* this model is NOT reliable. Its rank IC against 3-5
day forward returns was about +0.03 to +0.04 on SPY/QQQ in 2011-2019
(already negative on IWM) and between -0.005 and -0.11 on all three in
2020-2026. It is kept as a research tool; the production
strategy (trading/options_swing.py) uses explicit, validated rules instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

STRETCH_EDGES = (-1.0, -0.33, 0.33, 1.0)
N_STRETCH = len(STRETCH_EDGES) + 1
N_STATES = N_STRETCH * 2


def compute_states(close: pd.Series) -> pd.Series:
    """Causal state label in ``[0, N_STATES)`` for each day (-1 while warming up)."""
    logret = np.log(close).diff()
    vol20 = logret.rolling(20).std()
    ret5 = np.log(close).diff(5)
    stretch = ret5 / (vol20 * math.sqrt(5))
    bucket = np.digitize(stretch.fillna(0.0).to_numpy(), STRETCH_EDGES)
    vol_med = vol20.rolling(252, min_periods=60).median()
    high_vol = (vol20 > vol_med).to_numpy().astype(int)
    states = bucket + N_STRETCH * high_vol
    valid = stretch.notna() & vol_med.notna()
    return pd.Series(np.where(valid, states, -1), index=close.index, dtype=int)


@dataclass
class RegimeForecast:
    state: int
    p_up: float
    exp_return: float
    median_return: float
    q05: float
    q95: float
    n_analogs: int


class RegimeMCMC:
    """Monte Carlo over an empirical Markov chain of market states."""

    def __init__(
        self,
        horizon: int = 3,
        n_paths: int = 4000,
        lookback: int = 1500,
        min_analogs: int = 30,
        seed: Optional[int] = 7,
    ) -> None:
        self.horizon = horizon
        self.n_paths = n_paths
        self.lookback = lookback
        self.min_analogs = min_analogs
        self.rng = np.random.default_rng(seed)

    def forecast_at(
        self, logret: np.ndarray, states: np.ndarray, t: int
    ) -> Optional[RegimeForecast]:
        """Forecast the ``horizon``-day log return from the close of day ``t``.

        ``logret[i]`` is the return from close i-1 to close i and ``states[i]``
        the state at close i. Analogs are days ``j`` in ``[t - lookback, t - 1]``
        so that ``logret[j+1]`` and ``states[j+1]`` are both known at ``t``.
        """
        s0 = states[t]
        if s0 < 0:
            return None
        lo = max(1, t - self.lookback)
        js = np.arange(lo, t)
        js = js[states[js] >= 0]
        if len(js) < 50:
            return None
        hist_states = states[js]
        # Pools of candidate days per state
        pools = [js[hist_states == s] for s in range(N_STATES)]
        if len(pools[s0]) < self.min_analogs:
            return None

        total = np.zeros(self.n_paths)
        cur = np.full(self.n_paths, s0)
        for _ in range(self.horizon):
            nxt_state = np.empty_like(cur)
            for s in np.unique(cur):
                mask = cur == s
                pool = pools[s] if len(pools[s]) >= 5 else js  # sparse state: unconditional
                draws = pool[self.rng.integers(0, len(pool), mask.sum())]
                total[mask] += logret[draws + 1]
                nxt_state[mask] = states[draws + 1]
            cur = np.where(nxt_state >= 0, nxt_state, s0)

        simple = np.expm1(total)
        return RegimeForecast(
            state=int(s0),
            p_up=float(np.mean(simple > 0)),
            exp_return=float(np.mean(simple)),
            median_return=float(np.median(simple)),
            q05=float(np.quantile(simple, 0.05)),
            q95=float(np.quantile(simple, 0.95)),
            n_analogs=int(len(pools[s0])),
        )

    def forecast_series(self, close: pd.Series, start: Optional[str] = None) -> pd.DataFrame:
        """Walk forward through ``close`` producing one causal forecast per day."""
        logret = np.log(close).diff().fillna(0.0).to_numpy()
        states = compute_states(close).to_numpy()
        idx = close.index
        t0 = 0 if start is None else int(idx.searchsorted(pd.Timestamp(start)))
        rows = []
        for t in range(max(t0, 1), len(close)):
            fc = self.forecast_at(logret, states, t)
            if fc is not None:
                rows.append((idx[t], fc.state, fc.p_up, fc.exp_return,
                             fc.median_return, fc.q05, fc.q95, fc.n_analogs))
        return pd.DataFrame(
            rows,
            columns=["date", "state", "p_up", "exp_return", "median_return",
                     "q05", "q95", "n_analogs"],
        ).set_index("date")

    def forecast_latest(self, close: pd.Series) -> Optional[RegimeForecast]:
        """Forecast from the most recent close."""
        logret = np.log(close).diff().fillna(0.0).to_numpy()
        states = compute_states(close).to_numpy()
        return self.forecast_at(logret, states, len(close) - 1)
