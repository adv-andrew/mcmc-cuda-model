"""Stage 2: which confirmation filters improve the dip-buying debit spread?

Starting from the best structure family found in
``research_options_structures.py`` (call debit spreads, Connors-style exit),
this tests every combination of four *pre-registered*, economically
motivated filters:

- ``atr``:    pullback of at least 1.5 ATR from the 5-day high (a real dip)
- ``calm``:   20-day realized vol below 20% (avoid crash regimes, where
              dips keep falling and options are expensive)
- ``weekly``: weekly close above its 10-week SMA (higher-timeframe trend)
- ``nofri``:  no Friday entries (avoids paying weekend theta immediately)

Combinations are ranked on in-sample (2011-2019) Sharpe. Out-of-sample
(2020-present) is reported next to it but never used for selection.

Usage:
    python scripts/research_filters.py
"""

import sys

sys.path.insert(0, ".")

import itertools
from pathlib import Path

import pandas as pd

from backtesting.market_data import load_universe
from backtesting.options_backtest import ExitRules, OptionsSwingBacktester, StructureSpec
from backtesting.options_pricing import atm_iv_series
from trading.features import build_features

SYMBOLS = ["SPY", "QQQ", "IWM"]
SPLIT = "2020-01-01"
START = "2011-01-01"
OUT = Path("data/results")

BASES = {
    "rsi2<10&up": lambda f: (f.rsi2 < 10) & f.above200,
    "dip_composite": lambda f: f.above200 & (
        (f.rsi2 < 10) | ((f.ibs < 0.25) & (f.rsi2 < 15)) | (f.bb_z < -2) | (f.pullback_atr >= 2)
    ),
}
FILTERS = {
    "atr": lambda f: f.pullback_atr >= 1.5,
    "calm": lambda f: f.rv20 < 0.20,
    "weekly": lambda f: f.weekly_up,
    "nofri": lambda f: f.dow != 4,
}
STRUCTURES = {
    "cds_21d": StructureSpec("call_debit_spread", dte=21, long_delta=0.55, short_delta=0.30),
    "cds_30d": StructureSpec("call_debit_spread", dte=30, long_delta=0.60, short_delta=0.35),
}
EXITS = {
    "min3_max7_t60": ExitRules(min_hold=3, max_hold=7, profit_target=0.6, signal_exit=True),
    "fixed3": ExitRules(min_hold=3, max_hold=3, profit_target=None, signal_exit=False),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_universe(SYMBOLS + ["VIX"])
    feats = {s: build_features(data[s], data["VIX"]) for s in SYMBOLS}
    ivs = {s: atm_iv_series(data[s], data["VIX"], data["SPY"]) for s in SYMBOLS}
    exit_sig = {s: feats[s].close > feats[s].sma5 for s in SYMBOLS}
    prio = {s: feats[s].pullback_atr for s in SYMBOLS}

    combos = [c for r in range(len(FILTERS) + 1) for c in itertools.combinations(FILTERS, r)]
    rows = []
    for (bn, bfn), combo, (sn, spec), (xn, xr) in itertools.product(
        BASES.items(), combos, STRUCTURES.items(), EXITS.items()
    ):
        entries = {}
        for s in SYMBOLS:
            m = bfn(feats[s])
            for name in combo:
                m = m & FILTERS[name](feats[s])
            entries[s] = m
        for mc in (1, 3):
            bt = OptionsSwingBacktester(spec, xr, risk_per_trade=0.05, max_concurrent=mc)
            res = bt.run(feats, ivs, entries, exit_sig, 1, prio, start=START)
            row = {"base": bn, "filters": "+".join(combo) or "-", "structure": sn,
                   "exit": xn, "max_conc": mc}
            for tag, (a, b) in {"IS": (START, SPLIT), "OOS": (SPLIT, None)}.items():
                st = res.stats(a, b)
                for k in ("trades", "win_rate", "avg_ret", "cagr", "max_drawdown", "sharpe"):
                    row[f"{k}_{tag}"] = st.get(k)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "filters.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["base", "filters", "structure", "exit", "max_conc", "trades_IS", "win_rate_IS",
            "avg_ret_IS", "sharpe_IS", "max_drawdown_IS", "trades_OOS", "win_rate_OOS",
            "avg_ret_OOS", "cagr_OOS", "max_drawdown_OOS", "sharpe_OOS"]
    print("=== Top 25 by IN-SAMPLE Sharpe (min 60 IS trades) ===")
    top = df[df.trades_IS >= 60].sort_values("sharpe_IS", ascending=False)
    print(top[cols].head(25).round(3).to_string(index=False))
    print("\n=== Marginal effect of each filter (median Sharpe with vs without) ===")
    for name in FILTERS:
        has = df.filters.str.contains(name)
        print(f"{name:<7} IS {df[has].sharpe_IS.median():+.2f} vs {df[~has].sharpe_IS.median():+.2f}"
              f" | OOS {df[has].sharpe_OOS.median():+.2f} vs {df[~has].sharpe_OOS.median():+.2f}")


if __name__ == "__main__":
    main()
