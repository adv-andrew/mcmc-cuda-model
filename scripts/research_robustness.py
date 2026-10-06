"""Robustness round: choose the most *reliable* structure, then widen the universe.

Pre-registered protocol (fixed before any of this was run)
----------------------------------------------------------
Signal and exits are unchanged (trading/options_swing.py). Only the option
structure, max hold and universe are up for change.

Phase 1 - choose (SPY/QQQ/IWM, 2011-2019 only)
    Candidates: single calls with delta {0.60, 0.70, 0.80, 0.90} x DTE
    {30, 45, 60, 90} x max hold {7, 10}, plus the old 0.55/0.30 debit spread.
    Score = the *worst* per-trade Sharpe (mean / stdev of return on risk)
    across four pricing worlds: base, +10% IV paid at entry, steep call
    skew (0.25), and 1.5x bid/ask. Highest worst-case wins.

Phase 2 - validate the winner against the current default
    A  SPY/QQQ/IWM 2020-2026 (held out)
    B  SPY/QQQ/IWM before 2011 (SPY 1996-, QQQ 2000-, IWM 2001-)
    C  13 ETFs never used before (DIA, MDY, IJR, RSP, 9 sector SPDRs,
       EFA, EEM), 1999-2026, charged the wider stock-tier bid/ask. Includes a
       placebo test against random uptrend days.

Phase 3 - portfolio (all data from 2000, same total risk budget)
    P1  core 3 ETFs, 2 positions max
    P2  core + 12 liquid-option ETFs (MDY/IJR/RSP excluded: thin options),
        4 positions max at half the size each
    Position size targets the same per-trade P&L volatility as the current
    default (5% premium on a 0.70-delta call), so structures are compared at
    equal risk rather than equal premium.

Usage:
    python scripts/research_robustness.py
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
from backtesting.options_backtest import (
    OptionsSwingBacktester,
    PricingScenario,
    StructureSpec,
    non_overlapping_entries,
    simulate_trades,
)
from backtesting.options_pricing import CostModel, SkewModel, atm_iv_series
from trading.features import build_features
from trading.options_swing import SwingConfig, confidence_score, entry_signal, exit_signal

CORE = ["SPY", "QQQ", "IWM"]
FRESH = {
    "US broad (DIA MDY IJR RSP)": ["DIA", "MDY", "IJR", "RSP"],
    "Sectors (9 SPDRs)": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
    "International (EFA EEM)": ["EFA", "EEM"],
}
TRADABLE_FRESH = ["DIA", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
                  "EFA", "EEM"]

SCENARIOS = {
    "base": PricingScenario(),
    "entry IV +10%": PricingScenario(entry_iv_mult=1.10),
    "call skew 0.25": PricingScenario(skew=SkewModel(call_slope=0.25)),
    "1.5x bid/ask": PricingScenario(cost_mult=1.5),
}
ALL_HARSH = PricingScenario(skew=SkewModel(call_slope=0.25), entry_iv_mult=1.10, cost_mult=1.5,
                            entry_delay=1, exit_delay=1)

CFG = SwingConfig.from_yaml()
CANDIDATES = {
    f"call d{d:.2f} {dte}d hold<={mh}": (StructureSpec("long_call", dte=dte, long_delta=d), mh)
    for d, dte, mh in product((0.60, 0.70, 0.80, 0.90), (30, 45, 60, 90), (7, 10))
}
CANDIDATES["debit spread 0.55/0.30 21d hold<=7"] = (
    StructureSpec("call_debit_spread", dte=21, long_delta=0.55, short_delta=0.30), 7)
DEFAULT_NAME = "call d0.70 30d hold<=7"

# Populated in main() before worker processes fork.
UNIVERSE: dict = {}


def prepare(symbols, data, vix, spy):
    out = {}
    for s in symbols:
        df = data[s]
        f = build_features(df, vix)
        out[s] = {
            "f": f,
            "iv": atm_iv_series(df, vix, spy),
            "sig": entry_signal(f, CFG),
            "exit": exit_signal(f),
            "scale": split_factor(df.index, s),
        }
    return out


def run_symbol(args):
    """Strategy trades for one symbol / structure / scenario / window."""
    sym, name, sc_name, start, end, tight = args
    u = UNIVERSE[sym]
    spec, max_hold = CANDIDATES[name]
    sc = ALL_HARSH if sc_name == "ALL HARSH" else SCENARIOS[sc_name]
    exits = replace(CFG.exits(), max_hold=max_hold)
    sig = u["sig"].loc[start:end]
    tr = simulate_trades(u["f"]["close"], u["iv"], u["exit"], spec, exits, sc,
                         base_costs=CostModel.etf() if tight else CostModel.stock(),
                         entry_dates=sig.index[sig.to_numpy()], start=start, end=end,
                         is_etf=True, price_scale=u["scale"])
    if tr.empty:
        return tr
    taken = non_overlapping_entries(pd.Series(True, index=tr.index), tr.held)
    return tr.loc[taken].assign(symbol=sym)


def run_placebo_symbol(args):
    """Every uptrend day (for placebo sampling)."""
    sym, name, start, end, tight = args
    u = UNIVERSE[sym]
    spec, max_hold = CANDIDATES[name]
    exits = replace(CFG.exits(), max_hold=max_hold)
    tr = simulate_trades(u["f"]["close"], u["iv"], u["exit"], spec, exits, PricingScenario(),
                         base_costs=CostModel.etf() if tight else CostModel.stock(),
                         start=start, end=end, is_etf=True, price_scale=u["scale"])
    up = u["f"]["above200"].reindex(tr.index, fill_value=False)
    return tr[up.to_numpy()].assign(symbol=sym)


def pooled(pool, symbols, name, sc_name, start, end, tight_set=frozenset(CORE)):
    jobs = [(s, name, sc_name, start, end, s in tight_set) for s in symbols]
    parts = [p for p in pool.map(run_symbol, jobs) if not p.empty]
    return pd.concat(parts).sort_index() if parts else pd.DataFrame(columns=["ret"])


def tstats(r: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 3:
        return {"n": n, "win": np.nan, "avg": np.nan, "sd": np.nan, "sharpe": np.nan, "t": np.nan}
    sd = r.std(ddof=1)
    return {"n": n, "win": (r > 0).mean(), "avg": r.mean(), "sd": sd,
            "sharpe": r.mean() / sd, "t": r.mean() / sd * math.sqrt(n)}


def fmt(st: dict) -> str:
    if not st["n"] or np.isnan(st["avg"]):
        return f"n={st['n']:>4}  (too few)"
    return (f"n={st['n']:>4} win={st['win']:.0%} avg={st['avg']:+.2%} "
            f"trade-Sharpe={st['sharpe']:+.3f} t={st['t']:+.2f}")


def main() -> None:
    pd.set_option("display.width", 220)
    vix = load_universe(["VIX"])["VIX"]
    fresh_all = sorted({s for g in FRESH.values() for s in g})
    data = {s: load_long_history(s) for s in CORE + fresh_all}
    spy = data["SPY"]
    UNIVERSE.update(prepare(CORE + fresh_all, data, vix, spy))

    with ProcessPoolExecutor(max_workers=4) as pool:
        # ------------------------------------------------------------ Phase 1
        print("=" * 100)
        print("PHASE 1 - choose on SPY/QQQ/IWM 2011-2019 by WORST-CASE per-trade Sharpe")
        print("=" * 100)
        rows = []
        for name in CANDIDATES:
            row = {"candidate": name}
            for sc_name in SCENARIOS:
                st = tstats(pooled(pool, CORE, name, sc_name, "2011-01-01", "2019-12-31").ret)
                row[sc_name] = st["sharpe"]
                if sc_name == "base":
                    row.update(n=st["n"], win=st["win"], avg=st["avg"], sd=st["sd"])
            row["worst"] = min(row[k] for k in SCENARIOS)
            rows.append(row)
        p1 = pd.DataFrame(rows).set_index("candidate").sort_values("worst", ascending=False)
        print(p1[["n", "win", "avg", "sd", *SCENARIOS, "worst"]].round(3).head(15).to_string())
        print("...")
        print(p1.loc[[DEFAULT_NAME, "debit spread 0.55/0.30 21d hold<=7"],
                     ["n", "win", "avg", "sd", *SCENARIOS, "worst"]].round(3).to_string())
        winner = p1.index[0]
        print(f"\nWINNER (pre-registered rule): {winner}")

        # ------------------------------------------------------------ Phase 2
        print("\n" + "=" * 100)
        print("PHASE 2 - validation (winner vs current default)")
        print("=" * 100)
        windows = {
            "A  core 2020-2026 (held out)": (CORE, "2020-01-01", "2026-12-31", frozenset(CORE)),
            "B  core pre-2011": (CORE, "1996-01-01", "2010-12-31", frozenset(CORE)),
        }
        for gname, syms in FRESH.items():
            windows[f"C  {gname}, 1999-2026"] = (syms, "1999-06-01", "2026-12-31", frozenset())
        for wname, (syms, a, b, tight) in windows.items():
            print(f"\n{wname}")
            for name in dict.fromkeys([winner, DEFAULT_NAME]):
                for sc_name in ("base", "entry IV +10%", "ALL HARSH"):
                    tr = pooled(pool, syms, name, sc_name, a, b, tight)
                    print(f"  {name:<26} {sc_name:<14} options: {fmt(tstats(tr.ret))}")
                    if sc_name == "base":
                        print(f"  {'':<26} {'underlying':<14}          "
                              f"{fmt(tstats(tr.und_ret))}")

        # Placebo on the fresh universe for the winner
        print("\nPlacebo, fresh ETFs 1999-2026 (winner, base pricing): signal vs random uptrend days")
        fresh_trades = pooled(pool, fresh_all, winner, "base", "1999-06-01", "2026-12-31",
                              frozenset())
        pool_parts = list(pool.map(run_placebo_symbol,
                                   [(s, winner, "1999-06-01", "2026-12-31", False)
                                    for s in fresh_all]))
        placebo = pd.concat(pool_parts)
        rng = np.random.default_rng(7)
        counts = fresh_trades.symbol.value_counts()
        by_sym = {s: placebo[placebo.symbol == s] for s in counts.index}
        sims_opt, sims_und = [], []
        for _ in range(5000):
            draw = pd.concat([by_sym[s].iloc[rng.integers(0, len(by_sym[s]), n)]
                              for s, n in counts.items()])
            sims_opt.append(draw.ret.mean())
            sims_und.append(draw.und_ret.mean())
        sims_opt, sims_und = np.array(sims_opt), np.array(sims_und)
        print(f"  options: signal {fresh_trades.ret.mean():+.2%} vs random "
              f"{np.median(sims_opt):+.2%}  p={np.mean(sims_opt >= fresh_trades.ret.mean()):.4f}")
        print(f"  underlying: signal {fresh_trades.und_ret.mean():+.2%} vs random "
              f"{np.median(sims_und):+.2%}  p={np.mean(sims_und >= fresh_trades.und_ret.mean()):.4f}")

        # Sizing: equal per-trade P&L volatility to the current default
        base_sd = p1.loc[DEFAULT_NAME, "sd"]
        win_sd = p1.loc[winner, "sd"]
        target = CFG.risk_per_trade * base_sd
        risk_winner = target / win_sd
        print(f"\nSizing for equal per-trade volatility: default {CFG.risk_per_trade:.1%} premium "
              f"(trade sd {base_sd:.2f}) -> winner {risk_winner:.1%} premium (trade sd {win_sd:.2f})")

    # ---------------------------------------------------------------- Phase 3
    print("\n" + "=" * 100)
    print("PHASE 3 - portfolio, same total risk budget (2000-2026, all data from one source)")
    print("=" * 100)
    feats = {s: UNIVERSE[s]["f"] for s in UNIVERSE}
    ivs = {s: UNIVERSE[s]["iv"] for s in UNIVERSE}
    sigs = {s: UNIVERSE[s]["sig"] for s in UNIVERSE}
    exs = {s: UNIVERSE[s]["exit"] for s in UNIVERSE}
    scales = {s: UNIVERSE[s]["scale"] for s in UNIVERSE}
    prio = {s: confidence_score(feats[s]) + feats[s].pullback_atr for s in UNIVERSE}
    etf_grid = tuple(UNIVERSE)

    def portfolio(name, syms, conc, risk, scenario_cost=1.0, skew=None):
        spec, mh = CANDIDATES[name]
        bt = OptionsSwingBacktester(spec, replace(CFG.exits(), max_hold=mh), risk_per_trade=risk,
                                    max_concurrent=conc, etfs=etf_grid, tight_cost_symbols=CORE,
                                    cost_multiplier=scenario_cost, skew=skew)
        sub = lambda d: {s: d[s] for s in syms}  # noqa: E731
        return bt.run(sub(feats), sub(ivs), sub(sigs), sub(exs), 1, sub(prio),
                      start="2000-01-01", price_scale=sub(scales))

    configs = {
        f"P1 core, default ({DEFAULT_NAME}), 2 x {CFG.risk_per_trade:.1%}":
            (DEFAULT_NAME, CORE, 2, CFG.risk_per_trade),
        f"P1 core, winner, 2 x {risk_winner:.1%}": (winner, CORE, 2, risk_winner),
        f"P2 core+12 ETFs, default, 4 x {CFG.risk_per_trade / 2:.1%}":
            (DEFAULT_NAME, CORE + TRADABLE_FRESH, 4, CFG.risk_per_trade / 2),
        f"P2 core+12 ETFs, winner, 4 x {risk_winner / 2:.1%}":
            (winner, CORE + TRADABLE_FRESH, 4, risk_winner / 2),
    }
    periods = {"2000-2010": ("2000-01-01", "2011-01-01"), "2011-2019": ("2011-01-01", "2020-01-01"),
               "2020-2026": ("2020-01-01", None), "ALL 2000-2026": ("2000-01-01", None)}
    for label, (name, syms, conc, risk) in configs.items():
        for stress, kw in {"base": {}, "harsh (skew .25, 1.5x cost)":
                           {"scenario_cost": 1.5, "skew": SkewModel(call_slope=0.25)}}.items():
            res = portfolio(name, syms, conc, risk, **kw)
            print(f"\n{label}  [{stress}]")
            for pname, (a, b) in periods.items():
                st = res.stats(a, b)
                eq = res.equity.loc[a:b] if b else res.equity.loc[a:]
                yr = pd.concat([eq.iloc[:1], eq.resample("YE").last()]).pct_change().iloc[1:]
                yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
                print(f"   {pname:<14} trades/yr={st['trades'] / yrs:5.1f}  win={st.get('win_rate', 0):.0%}"
                      f"  CAGR={st['cagr']:+.1%}  maxDD={st['max_drawdown']:+.1%}  "
                      f"Sharpe={st['sharpe']:.2f}  losing years={int((yr < 0).sum())}/{len(yr)}")


if __name__ == "__main__":
    main()
