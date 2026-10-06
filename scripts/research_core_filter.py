"""Single pre-registered test: a whipsaw-resistant core filter.

Portfolio mode's worst year (2022) came from the 10-month SMA flipping in
and out on bear-market rallies. Alternative (fixed before running): the core
goes to T-bills only when BOTH SPY's month-end close is below its 10-month
SMA AND its trailing 12-month total return is below the T-bill return
(absolute momentum, Antonacci 2012). Requiring two independent confirmations
should cut false exits. Adopt only if excess Sharpe improves in ALL three
eras (1996-2010, 2011-2019, 2020-2026), options overlay included.

Usage:
    python scripts/research_core_filter.py
"""

import sys

sys.path.insert(0, ".")

import pandas as pd

from backtesting.market_data import load_long_history, load_universe, split_factor
from backtesting.options_backtest import summarize
from backtesting.options_pricing import atm_iv_series, daily_risk_free
from backtesting.portfolio import faber_signal, simulate_core_overlay
from trading.features import build_features, higher_tf_series
from trading.options_swing import SwingConfig, confidence_score, entry_signal, exit_signal

CORE = ["SPY", "QQQ", "IWM"]
END = "2026-10-03"
START = "1996-01-01"
ERAS = {"1996-2010": ("1996-01-01", "2011-01-01"), "2011-2019": ("2011-01-01", "2020-01-01"),
        "2020-2026": ("2020-01-01", END), "ALL 1996-2026": ("1996-01-01", END)}


def absolute_momentum(total_return: pd.Series) -> pd.Series:
    """True when the last completed month's 12-month total return beats T-bills."""
    rf_index = (1 + daily_risk_free(total_return.index)).cumprod()
    excess = total_return / rf_index
    return (higher_tf_series(excess, "ME", lambda s: s.pct_change(12)) > 0).fillna(False)


def main() -> None:
    cfg = SwingConfig.from_yaml()
    vix = load_universe(["VIX"])["VIX"]
    data = {s: load_long_history(s) for s in CORE}
    data = {s: d[d.index < END] for s, d in data.items()}
    feats = {s: build_features(data[s], vix) for s in CORE}
    ivs = {s: atm_iv_series(data[s], vix, data["SPY"]) for s in CORE}
    ent = {s: entry_signal(feats[s], cfg) for s in CORE}
    ext = {s: exit_signal(feats[s]) for s in CORE}
    prio = {s: confidence_score(feats[s]) + feats[s].pullback_atr for s in CORE}
    scale = {s: split_factor(data[s].index, s) for s in CORE}
    spy = data["SPY"]
    sma_on = faber_signal(spy["Close"])
    dual_on = sma_on | absolute_momentum(spy["AdjClose"])  # out only if BOTH say out

    res = {}
    for name, sig in {"10-month SMA (current)": sma_on,
                      "SMA OR 12-month momentum (out only if both negative)": dual_on}.items():
        eq, _ = simulate_core_overlay(spy["AdjClose"], sig, feats, ivs, ent, ext,
                                      cfg.structure(), cfg.exits(), cfg.core_weight,
                                      cfg.risk_per_trade, cfg.max_concurrent, prio, scale,
                                      start=START)
        flips = int(sig.loc[START:].astype(int).diff().abs().sum())
        print(f"\n{name}  (signal changes: {flips})")
        res[name] = {}
        for era, (a, b) in ERAS.items():
            e = eq[(eq.index >= a) & (eq.index < b)]
            st = summarize(pd.DataFrame(), e)
            yr = pd.concat([e.iloc[:1], e.resample("YE").last()]).pct_change().iloc[1:]
            res[name][era] = st["sharpe_excess"]
            print(f"   {era:<14} CAGR={st['cagr']:+6.1%}  maxDD={st['max_drawdown']:+6.1%}  "
                  f"Sharpe(ex)={st['sharpe_excess']:5.2f}  worst year={yr.min():+6.1%}")
        y22 = eq.loc["2022"]
        print(f"   2022: {y22.iloc[-1] / eq.loc[:'2021'].iloc[-1] - 1:+.1%}")
    a, b = list(res)
    wins = {e: res[b][e] > res[a][e] for e in list(ERAS)[:3]}
    print("\nDual filter beats current on excess Sharpe:", wins)
    print("Pre-registered decision:", "ADOPT dual filter" if all(wins.values())
          else "keep the 10-month SMA")


if __name__ == "__main__":
    main()
