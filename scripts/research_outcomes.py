"""What range of outcomes should you expect from portfolio mode?

Stationary block bootstrap of daily returns (1996-2026, mean block length
~1 month so crashes and recoveries stay intact) from the single-account
backtest of portfolio mode, sampled *jointly* with SPY buy & hold on the
same days. Reports 1-year and 5-year return percentiles, drawdown odds,
and how often portfolio mode beats buy & hold.

Usage:
    python scripts/research_outcomes.py
"""

import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from backtesting.market_data import load_long_history, load_universe, split_factor
from backtesting.options_pricing import atm_iv_series
from backtesting.portfolio import faber_signal, simulate_core_overlay
from trading.features import build_features
from trading.options_swing import SwingConfig, confidence_score, entry_signal, exit_signal

CORE = ["SPY", "QQQ", "IWM"]
END = "2026-10-03"
START = "1996-01-01"
N_SIMS = 10_000
MEAN_BLOCK = 21


def stationary_bootstrap(n_obs: int, length: int, rng) -> np.ndarray:
    """Indices for one stationary-bootstrap path (Politis & Romano)."""
    idx = np.empty(length, dtype=int)
    idx[0] = rng.integers(n_obs)
    restart = rng.random(length) < 1.0 / MEAN_BLOCK
    starts = rng.integers(n_obs, size=length)
    for t in range(1, length):
        idx[t] = starts[t] if restart[t] else (idx[t - 1] + 1) % n_obs
    return idx


def main() -> None:
    cfg = SwingConfig.from_yaml()
    vix = load_universe(["VIX"])["VIX"]
    data = {s: load_long_history(s) for s in CORE}
    data = {s: d[d.index < END] for s, d in data.items()}
    feats = {s: build_features(data[s], vix) for s in CORE}
    ivs = {s: atm_iv_series(data[s], vix, data["SPY"]) for s in CORE}
    eq, _ = simulate_core_overlay(
        data["SPY"]["AdjClose"], faber_signal(data["SPY"]["Close"]), feats, ivs,
        {s: entry_signal(feats[s], cfg) for s in CORE}, {s: exit_signal(feats[s]) for s in CORE},
        cfg.structure(), cfg.exits(), cfg.core_weight, cfg.risk_per_trade, cfg.max_concurrent,
        {s: confidence_score(feats[s]) + feats[s].pullback_atr for s in CORE},
        {s: split_factor(data[s].index, s) for s in CORE}, start=START)
    spy = data["SPY"]["AdjClose"].reindex(eq.index)
    rets = pd.DataFrame({"pm": eq.pct_change(), "spy": spy.pct_change()}).dropna().to_numpy()
    rng = np.random.default_rng(11)

    for years in (1, 5):
        length = 252 * years
        out = {"pm": [], "spy": [], "pm_dd": [], "spy_dd": []}
        for _ in range(N_SIMS):
            path = rets[stationary_bootstrap(len(rets), length, rng)]
            for k, col in (("pm", 0), ("spy", 1)):
                curve = np.cumprod(1 + path[:, col])
                out[k].append(curve[-1] - 1)
                out[k + "_dd"].append((curve / np.maximum.accumulate(curve) - 1).min())
        pm, sp = np.array(out["pm"]), np.array(out["spy"])
        pmd, spd = np.array(out["pm_dd"]), np.array(out["spy_dd"])
        ann = (lambda x: (1 + x) ** (1 / years) - 1)
        print(f"\n=== {years}-year horizon ({N_SIMS:,} block-bootstrap paths) ===")
        print(f"{'':<22}{'portfolio mode':>16}{'SPY buy & hold':>16}")
        for q, lab in ((5, "bad (5th pct)"), (25, "weak (25th)"), (50, "median"),
                       (75, "good (75th)"), (95, "great (95th)")):
            print(f"  {lab:<20}{np.percentile(pm, q):>+15.1%} {np.percentile(sp, q):>+15.1%}")
        if years > 1:
            print(f"  {'median annualized':<20}{ann(np.median(pm)):>+15.1%} "
                  f"{ann(np.median(sp)):>+15.1%}")
        print(f"  {'P(loss)':<20}{np.mean(pm < 0):>15.0%} {np.mean(sp < 0):>15.0%}")
        print(f"  {'median max DD':<20}{np.median(pmd):>+15.1%} {np.median(spd):>+15.1%}")
        print(f"  {'P(drawdown < -20%)':<20}{np.mean(pmd < -0.2):>15.0%} "
              f"{np.mean(spd < -0.2):>15.0%}")
        print(f"  {'P(drawdown < -30%)':<20}{np.mean(pmd < -0.3):>15.0%} "
              f"{np.mean(spd < -0.3):>15.0%}")
        print(f"  P(portfolio mode beats SPY over the period): {np.mean(pm > sp):.0%}")


if __name__ == "__main__":
    main()
