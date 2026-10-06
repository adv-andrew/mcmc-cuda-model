"""Single pre-registered test: SPY-only core vs a three-index core.

Alternative core: one third each in SPY, QQQ and IWM, each held while its
own month-end close is above its 10-month SMA (else T-bills), modelled as
one synthetic total-return index rebalanced to equal thirds daily. Options
overlay unchanged. Rule fixed before running: adopt only if it beats the
SPY core on excess Sharpe in ALL three eras (2001-2010, 2011-2019,
2020-2026). QQQ's strong 2010s is hindsight-tinted, which is why every era
(including the 2000-2002 tech crash) must pass.

Usage:
    python scripts/research_core_universe.py
"""

import sys

sys.path.insert(0, ".")

import pandas as pd

from backtesting.market_data import load_long_history, load_universe, split_factor
from backtesting.options_backtest import summarize
from backtesting.options_pricing import atm_iv_series
from backtesting.portfolio import faber_signal, simulate_core_overlay, trend_core_returns
from trading.features import build_features
from trading.options_swing import SwingConfig, confidence_score, entry_signal, exit_signal

CORE = ["SPY", "QQQ", "IWM"]
END = "2026-10-03"
START = "2001-06-01"  # IWM (May 2000) + 10 months of history for its trend filter
ERAS = {"2001-2010": ("2001-06-01", "2011-01-01"), "2011-2019": ("2011-01-01", "2020-01-01"),
        "2020-2026": ("2020-01-01", END), "ALL 2001-2026": ("2001-06-01", END)}


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

    cal = data["SPY"].loc[START:].index
    legs = {s: trend_core_returns(data[s]["AdjClose"], faber_signal(data[s]["Close"]))
            .reindex(cal).fillna(0.0) for s in CORE}
    spy_core = (1 + legs["SPY"]).cumprod()
    multi_core = (1 + sum(legs[s] for s in CORE) / 3.0).cumprod()
    always = pd.Series(True, index=cal)

    res = {}
    for name, idx in {"SPY core": spy_core, "SPY/QQQ/IWM core": multi_core}.items():
        eq, _ = simulate_core_overlay(idx, always, feats, ivs, ent, ext, cfg.structure(),
                                      cfg.exits(), cfg.core_weight, cfg.risk_per_trade,
                                      cfg.max_concurrent, prio, scale, start=START,
                                      switch_cost_bps=0.0)
        print(f"\n{name} + options overlay")
        res[name] = {}
        for era, (a, b) in ERAS.items():
            e = eq[(eq.index >= a) & (eq.index < b)]
            st = summarize(pd.DataFrame(), e)
            yr = pd.concat([e.iloc[:1], e.resample("YE").last()]).pct_change().iloc[1:]
            res[name][era] = st["sharpe_excess"]
            print(f"   {era:<14} CAGR={st['cagr']:+6.1%}  maxDD={st['max_drawdown']:+6.1%}  "
                  f"Sharpe(ex)={st['sharpe_excess']:5.2f}  worst year={yr.min():+6.1%}")
    wins = {e: res["SPY/QQQ/IWM core"][e] > res["SPY core"][e] for e in list(ERAS)[:3]}
    print("\nThree-index core beats SPY core on excess Sharpe:", wins)
    print("Pre-registered decision:", "ADOPT three-index core" if all(wins.values())
          else "keep SPY core")


if __name__ == "__main__":
    main()
