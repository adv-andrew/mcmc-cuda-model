"""Does portfolio mode still work with whole option contracts?

The research sizes options fractionally (e.g. "1.53 contracts"). Real
options trade in whole contracts of 100 shares, and a 0.80-delta, 30-day
SPY call costs ~$4,000-5,000 in 2026. This replays portfolio mode with
integer contracts for several starting account sizes, starting fresh in
2011 and in 2020, and compares with fractional sizing.

Usage:
    python scripts/research_account_size.py
"""

import sys

sys.path.insert(0, ".")

import pandas as pd

from backtesting.market_data import load_long_history, load_universe, split_factor
from backtesting.options_backtest import summarize
from backtesting.options_pricing import atm_iv_series
from backtesting.portfolio import faber_signal, simulate_core_overlay
from trading.features import build_features
from trading.options_swing import SwingConfig, confidence_score, entry_signal, exit_signal

CORE = ["SPY", "QQQ", "IWM"]
END = "2026-10-03"


def main() -> None:
    cfg = SwingConfig.from_yaml()
    vix = load_universe(["VIX"])["VIX"]
    data = {s: load_long_history(s) for s in CORE}
    data = {s: d[d.index < END] for s, d in data.items()}
    feats = {s: build_features(data[s], vix) for s in CORE}
    ivs = {s: atm_iv_series(data[s], vix, data["SPY"]) for s in CORE}
    sig = {s: entry_signal(feats[s], cfg) for s in CORE}
    ext = {s: exit_signal(feats[s]) for s in CORE}
    prio = {s: confidence_score(feats[s]) + feats[s].pullback_atr for s in CORE}
    scale = {s: split_factor(data[s].index, s) for s in CORE}
    fs = faber_signal(data["SPY"]["Close"])

    rows = []
    for start in ("2011-01-03", "2020-01-02"):
        for initial in (10_000, 25_000, 50_000, 100_000, 250_000, 1_000_000):
            for whole in (False, True):
                eq, tr = simulate_core_overlay(
                    data["SPY"]["AdjClose"], fs, feats, ivs, sig, ext, cfg.structure(),
                    cfg.exits(), cfg.core_weight, cfg.risk_per_trade, cfg.max_concurrent,
                    prio, scale, start=start, initial=initial, whole_contracts=whole)
                st = summarize(pd.DataFrame(), eq)
                by_sym = tr.symbol.value_counts().to_dict() if len(tr) else {}
                rows.append({"start": start[:4], "account": initial,
                             "sizing": "whole" if whole else "fractional",
                             "trades": len(tr), "CAGR": st["cagr"], "maxDD": st["max_drawdown"],
                             "Sharpe(ex)": st["sharpe_excess"],
                             "SPY/QQQ/IWM": "/".join(str(by_sym.get(s, 0)) for s in CORE)})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, formatters={
        "account": "${:,.0f}".format, "CAGR": "{:+.1%}".format, "maxDD": "{:+.1%}".format,
        "Sharpe(ex)": "{:.2f}".format}))
    df.to_csv("data/results/account_size.csv", index=False)


if __name__ == "__main__":
    main()
