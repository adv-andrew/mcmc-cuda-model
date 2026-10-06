"""Walk-forward re-selection of the entry thresholds.

If the chosen parameters (pullback >= 1.5 ATR, RSI(2) < 10) were a lucky
fit, a process that re-picks them every year from trailing data only would
do noticeably worse out of sample than the fixed values do in hindsight,
and the yearly picks would jump around.

Each year Y in 2003-2026: among pullback {1.0, 1.25, 1.5, 1.75, 2.0} ATR x
RSI(2) cap {5, 10, 15}, pick the combination with the best per-trade Sharpe
of the 0.80-delta call on SPY/QQQ/IWM trades entered in [Y-8, Y) (at least
30 trades), then trade year Y with it. Pre-registered decision: switch to
adaptive parameters only if they beat the fixed ones out of sample by t > 2.

Usage:
    python scripts/research_walk_forward.py
"""

import sys

sys.path.insert(0, ".")

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from backtesting.market_data import load_long_history, load_universe, split_factor
from backtesting.options_backtest import non_overlapping_entries, simulate_trades
from backtesting.options_pricing import atm_iv_series
from trading.features import build_features
from trading.options_swing import SwingConfig, entry_signal, exit_signal

CORE = ["SPY", "QQQ", "IWM"]
GRID = list(product((1.0, 1.25, 1.5, 1.75, 2.0), (5.0, 10.0, 15.0)))
FIXED = (1.5, 10.0)
TRAIL_YEARS = 8
CFG = SwingConfig.from_yaml()
U: dict = {}


def run(args):
    sym, atr, rsi = args
    u = U[sym]
    cfg = replace(CFG, min_pullback_atr=atr, rsi2_max=rsi)
    sig = entry_signal(u["f"], cfg)
    tr = simulate_trades(u["f"]["close"], u["iv"], exit_signal(u["f"]), CFG.structure(),
                         CFG.exits(), entry_dates=sig.index[sig.to_numpy()],
                         start="1996-01-01", price_scale=u["scale"])
    if tr.empty:
        return tr
    tr = tr.loc[non_overlapping_entries(pd.Series(True, index=tr.index), tr.held)]
    return tr.assign(symbol=sym, atr=atr, rsi=rsi)


def per_trade_sharpe(r: pd.Series) -> float:
    return r.mean() / r.std(ddof=1) if len(r) > 2 and r.std() > 0 else -np.inf


def main() -> None:
    vix = load_universe(["VIX"])["VIX"]
    data = {s: load_long_history(s) for s in CORE}
    for s in CORE:
        f = build_features(data[s], vix)
        U[s] = {"f": f, "iv": atm_iv_series(data[s], vix, data["SPY"]),
                "scale": split_factor(data[s].index, s)}
    with ProcessPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(run, [(s, a, r) for (a, r) in GRID for s in CORE]))
    allt = pd.concat([p for p in parts if not p.empty])
    allt["year"] = allt.index.year

    picks, wf, fixed = [], [], []
    for y in range(2003, 2027):
        hist = allt[(allt.year >= y - TRAIL_YEARS) & (allt.year < y)]
        scores = {}
        for a, r in GRID:
            h = hist[(hist.atr == a) & (hist.rsi == r)].ret
            if len(h) >= 30:
                scores[(a, r)] = per_trade_sharpe(h)
        if not scores:
            continue
        best = max(scores, key=scores.get)
        picks.append({"year": y, "atr": best[0], "rsi2<": best[1],
                      "trailing Sharpe": scores[best],
                      "fixed rank": sorted(scores.values(), reverse=True).index(scores[FIXED]) + 1
                      if FIXED in scores else None})
        yr = allt[allt.year == y]
        wf.append(yr[(yr.atr == best[0]) & (yr.rsi == best[1])])
        fixed.append(yr[(yr.atr == FIXED[0]) & (yr.rsi == FIXED[1])])
    wf, fixed = pd.concat(wf), pd.concat(fixed)
    pk = pd.DataFrame(picks).set_index("year")
    pd.set_option("display.width", 200)
    print("Yearly picks (trailing 8 years) and where the fixed 1.5 ATR / RSI<10 ranked of 15:")
    print(pk.round(3).to_string())

    def line(name, r):
        t = r.mean() / r.std(ddof=1) * math.sqrt(len(r))
        print(f"  {name:<34} n={len(r):>4} win={(r > 0).mean():.0%} avg={r.mean():+.2%} "
              f"trade-Sharpe={r.mean() / r.std(ddof=1):+.3f} t={t:+.2f}")

    print("\nOut-of-sample trades, 2003-2026:")
    line("walk-forward (re-picked yearly)", wf.ret)
    line("fixed 1.5 ATR / RSI(2)<10", fixed.ret)
    # paired difference by year (same years, different parameter sets)
    a = wf.groupby(wf.year).ret.mean()
    b = fixed.groupby(fixed.year).ret.mean()
    d = (a - b).dropna()
    t_diff = d.mean() / d.std(ddof=1) * math.sqrt(len(d))
    print(f"  yearly mean difference (walk-forward - fixed): {d.mean():+.2%} per trade, "
          f"t = {t_diff:+.2f} over {len(d)} years")
    print("  decision rule: adopt adaptive parameters only if t > 2 ->",
          "ADOPT" if t_diff > 2 else "keep fixed parameters")
    for half, (lo, hi) in {"2003-2014": (2003, 2014), "2015-2026": (2015, 2026)}.items():
        line(f"walk-forward {half}", wf[(wf.year >= lo) & (wf.year <= hi)].ret)
        line(f"fixed {half}", fixed[(fixed.year >= lo) & (fixed.year <= hi)].ret)
    pk.to_csv("data/results/walk_forward_picks.csv")


if __name__ == "__main__":
    main()
