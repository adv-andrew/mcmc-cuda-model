"""Pre-registered exit-rule study for the Options Swing Strategy.

Exits decide how much of each bounce is kept. Candidates (fixed before
running), all with max hold 10 trading days and a 3-day minimum unless
stated:

    E0  close > SMA(5)                         (current, Connors)
    E1  close > SMA(10)                        (fuller recovery)
    E2  RSI(2) > 70                            (short-term overbought)
    E3  close > prior day's high               (strength exit)
    E4  close at a new 5-day closing high      (pullback fully recovered)
    E5  first down close after a close > SMA(5) (let the bounce run)
    E6  fixed 5 trading days                   (time-only baseline)
    E7  close > SMA(5), minimum hold 2 days

Selection: SPY/QQQ/IWM 2011-2019 only, highest *worst-case* per-trade
Sharpe of the default 0.80-delta call across base pricing, +10% entry IV,
steep call skew and 1.5x bid/ask. Validation: SPY/QQQ/IWM 2020-2026 and
pre-2011, plus the underlying on 13 never-used ETFs (1999-2026).

Usage:
    python scripts/research_exits.py
"""

import sys

sys.path.insert(0, ".")

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import pandas as pd

from backtesting.market_data import load_long_history, load_universe, split_factor
from backtesting.options_backtest import (
    PricingScenario,
    non_overlapping_entries,
    simulate_trades,
)
from backtesting.options_pricing import CostModel, SkewModel, atm_iv_series
from trading.features import build_features
from trading.options_swing import SwingConfig, entry_signal

CORE = ["SPY", "QQQ", "IWM"]
FRESH = ["DIA", "MDY", "IJR", "RSP", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV",
         "XLY", "EFA", "EEM"]
FRESH = [s for s in FRESH if s not in ("MDY", "IJR", "RSP")] + ["MDY", "IJR", "RSP"]
SCENARIOS = {
    "base": PricingScenario(),
    "entry IV +10%": PricingScenario(entry_iv_mult=1.10),
    "call skew 0.25": PricingScenario(skew=SkewModel(call_slope=0.25)),
    "1.5x bid/ask": PricingScenario(cost_mult=1.5),
}
CFG = SwingConfig.from_yaml()


def exit_rules(f: pd.DataFrame) -> dict:
    """Exit-signal series for every candidate: name -> (series, min_hold, signal_exit)."""
    c = f["close"]
    above5 = c > f["sma5"]
    return {
        "E0 close>SMA5": (above5, 3, True),
        "E1 close>SMA10": (c > f["sma10"], 3, True),
        "E2 RSI2>70": (f["rsi2"] > 70, 3, True),
        "E3 close>prior high": (c > f["high"].shift(1), 3, True),
        "E4 new 5d closing high": (c > c.shift(1).rolling(5).max(), 3, True),
        "E5 down close after >SMA5": ((c < c.shift(1)) & above5.shift(1, fill_value=False),
                                       3, True),
        "E6 fixed 5 days": (pd.Series(False, index=f.index), 5, False),
        "E7 close>SMA5 min2": (above5, 2, True),
    }


UNIVERSE: dict = {}


def prepare(symbols, data, vix, spy):
    for s in symbols:
        f = build_features(data[s], vix)
        UNIVERSE[s] = {"f": f, "iv": atm_iv_series(data[s], vix, spy),
                       "sig": entry_signal(f, CFG), "exits": exit_rules(f),
                       "scale": split_factor(data[s].index, s)}


def run(args):
    sym, ename, sc_name, start, end, tight = args
    u = UNIVERSE[sym]
    ex, min_hold, sig_exit = u["exits"][ename]
    max_hold = 5 if ename.startswith("E6") else CFG.max_hold
    exits = replace(CFG.exits(), min_hold=min_hold, max_hold=max_hold, signal_exit=sig_exit)
    sig = u["sig"].loc[start:end]
    tr = simulate_trades(u["f"]["close"], u["iv"], ex, CFG.structure(), exits,
                         SCENARIOS[sc_name],
                         base_costs=CostModel.etf() if tight else CostModel.stock(),
                         entry_dates=sig.index[sig.to_numpy()], start=start, end=end,
                         price_scale=u["scale"])
    if tr.empty:
        return tr
    return tr.loc[non_overlapping_entries(pd.Series(True, index=tr.index), tr.held)]


def pooled(pool, syms, ename, sc_name, start, end, tight):
    parts = [p for p in pool.map(run, [(s, ename, sc_name, start, end, s in tight)
                                       for s in syms]) if not p.empty]
    return pd.concat(parts) if parts else pd.DataFrame(columns=["ret", "und_ret", "held"])


def st(r: pd.Series) -> dict:
    r = r.dropna()
    sd = r.std(ddof=1)
    return {"n": len(r), "win": (r > 0).mean(), "avg": r.mean(), "sharpe": r.mean() / sd,
            "t": r.mean() / sd * math.sqrt(len(r))}


def main() -> None:
    pd.set_option("display.width", 220)
    vix = load_universe(["VIX"])["VIX"]
    data = {s: load_long_history(s) for s in CORE + FRESH}
    prepare(CORE + FRESH, data, vix, data["SPY"])
    names = list(exit_rules(UNIVERSE["SPY"]["f"]))
    tight = frozenset(CORE)

    with ProcessPoolExecutor(max_workers=4) as pool:
        print("=" * 110)
        print("SELECTION: SPY/QQQ/IWM 2011-2019, 0.80-delta call, worst-case per-trade Sharpe")
        print("=" * 110)
        rows = []
        for e in names:
            row = {"exit": e}
            for sc in SCENARIOS:
                tr = pooled(pool, CORE, e, sc, "2011-01-01", "2019-12-31", tight)
                s = st(tr.ret)
                row[sc] = s["sharpe"]
                if sc == "base":
                    row.update(n=s["n"], win=s["win"], avg=s["avg"], days=tr.held.mean(),
                               und=tr.und_ret.mean())
            row["worst"] = min(row[k] for k in SCENARIOS)
            rows.append(row)
        sel = pd.DataFrame(rows).set_index("exit").sort_values("worst", ascending=False)
        print(sel[["n", "win", "avg", "days", "und", *SCENARIOS, "worst"]].round(3).to_string())
        winner = sel.index[0]
        print(f"\nWINNER (pre-registered rule): {winner}")

        print("\n" + "=" * 110)
        print("VALIDATION (winner vs current E0)")
        print("=" * 110)
        windows = {
            "core 2020-2026": (CORE, "2020-01-01", "2026-12-31"),
            "core pre-2011": (CORE, "1996-01-01", "2010-12-31"),
            "13 fresh ETFs 1999-2026 (underlying only)": (FRESH, "1999-06-01", "2026-12-31"),
        }
        for wname, (syms, a, b) in windows.items():
            print(f"\n{wname}")
            for e in dict.fromkeys([winner, "E0 close>SMA5"]):
                tr = pooled(pool, syms, e, "base", a, b, tight)
                u = st(tr.und_ret)
                line = (f"  {e:<26} underlying: n={u['n']:>4} win={u['win']:.0%} "
                        f"avg={u['avg']:+.2%} t={u['t']:+.2f}  days={tr.held.mean():.1f}")
                if syms is CORE:
                    o = st(tr.ret)
                    s2 = st(pooled(pool, syms, e, "entry IV +10%", a, b, tight).ret)
                    line += (f" | option: avg={o['avg']:+.2%} t={o['t']:+.2f}"
                             f" | +10% IV: avg={s2['avg']:+.2%} t={s2['t']:+.2f}")
                print(line)


if __name__ == "__main__":
    main()
