"""Single pre-registered test: Treasuries instead of T-bills when the core is out.

When SPY is below its 10-month SMA the core normally sits in T-bills.
Intermediate Treasuries (IEF, 2002+) often rally in equity bear markets
(2008, 2020) but fell with stocks in 2022. Rule fixed before running:
adopt IEF only if it beats T-bills on excess Sharpe in ALL three eras
(2003-2010, 2011-2019, 2020-2026), for the full portfolio (core + options).

The core sleeve is modelled as one synthetic total-return index that is
SPY when the signal (known at the prior close) is on and the risk-off asset
when it is off, with 5 bps per switch; the single-account engine then holds
85% in that index, rebalanced monthly, with the options overlay on top.

Usage:
    python scripts/research_risk_off.py
"""

import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from backtesting.market_data import load_long_history, load_universe, split_factor
from backtesting.options_backtest import summarize
from backtesting.options_pricing import atm_iv_series, daily_risk_free
from backtesting.portfolio import faber_signal, simulate_core_overlay
from trading.features import build_features
from trading.options_swing import SwingConfig, confidence_score, entry_signal, exit_signal

CORE = ["SPY", "QQQ", "IWM"]
END = "2026-10-03"
START = "2003-01-02"
ERAS = {"2003-2010": ("2003-01-01", "2011-01-01"), "2011-2019": ("2011-01-01", "2020-01-01"),
        "2020-2026": ("2020-01-01", END), "ALL 2003-2026": ("2003-01-01", END)}


def synthetic_core(spy_tr: pd.Series, off_ret: pd.Series, signal: pd.Series,
                   switch_bps: float = 5.0) -> pd.Series:
    held = signal.reindex(spy_tr.index).fillna(False).astype(bool).shift(1, fill_value=False)
    ret = pd.Series(np.where(held, spy_tr.pct_change().fillna(0.0),
                             off_ret.reindex(spy_tr.index).fillna(0.0)), index=spy_tr.index)
    ret -= held.astype(int).diff().abs().fillna(0.0) * switch_bps / 10_000.0
    return (1 + ret).cumprod()


def eras(eq: pd.Series) -> dict:
    out = {}
    for name, (a, b) in ERAS.items():
        e = eq[(eq.index >= a) & (eq.index < b)]
        st = summarize(pd.DataFrame(), e)
        yr = pd.concat([e.iloc[:1], e.resample("YE").last()]).pct_change().iloc[1:]
        out[name] = (st["cagr"], st["max_drawdown"], st["sharpe_excess"], yr.min())
    return out


def main() -> None:
    cfg = SwingConfig.from_yaml()
    vix = load_universe(["VIX"])["VIX"]
    data = {s: load_long_history(s) for s in CORE + ["IEF"]}
    data = {s: d[(d.index < END)] for s, d in data.items()}
    spy = data["SPY"].loc["1995":]
    spy_tr = spy["AdjClose"].loc[START:]
    sig = faber_signal(spy["Close"])
    ief_ret = data["IEF"]["AdjClose"].pct_change()
    tbill_ret = daily_risk_free(spy_tr.index)

    feats = {s: build_features(data[s], vix) for s in CORE}
    ivs = {s: atm_iv_series(data[s], vix, data["SPY"]) for s in CORE}
    ent = {s: entry_signal(feats[s], cfg) for s in CORE}
    ext = {s: exit_signal(feats[s]) for s in CORE}
    prio = {s: confidence_score(feats[s]) + feats[s].pullback_atr for s in CORE}
    scale = {s: split_factor(data[s].index, s) for s in CORE}
    always = pd.Series(True, index=spy_tr.index)
    none = {s: ent[s] & False for s in CORE}

    results = {}
    for name, off in {"T-bills when out": tbill_ret, "IEF when out": ief_ret}.items():
        core_idx = synthetic_core(spy_tr, off, sig)
        for label, entries in {"core only (85%)": none, "core + options": ent}.items():
            eq, _ = simulate_core_overlay(core_idx, always, feats, ivs, entries, ext,
                                          cfg.structure(), cfg.exits(), cfg.core_weight,
                                          cfg.risk_per_trade, cfg.max_concurrent, prio, scale,
                                          start=START, switch_cost_bps=0.0)
            results[(name, label)] = eras(eq)
            print(f"\n{name} | {label}")
            for era, (c, dd, sh, wy) in results[(name, label)].items():
                print(f"   {era:<14} CAGR={c:+6.1%}  maxDD={dd:+6.1%}  Sharpe(ex)={sh:5.2f}  "
                      f"worst year={wy:+6.1%}")

    t = results[("T-bills when out", "core + options")]
    i = results[("IEF when out", "core + options")]
    wins = [i[e][2] > t[e][2] for e in list(ERAS)[:3]]
    print("\nIEF beats T-bills on excess Sharpe by era:",
          dict(zip(list(ERAS)[:3], wins)))
    print("Pre-registered decision:", "ADOPT IEF" if all(wins) else "keep T-bills")


if __name__ == "__main__":
    main()
