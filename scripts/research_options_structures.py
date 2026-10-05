"""Which option structure best monetizes the 3-5 day dip-buying edge?

Grid: entry signal x option structure x exit rule, on SPY/QQQ/IWM with the
realistic pricing model (VIX-derived IV, skew, spreads, commissions) and
daily mark-to-market. Configurations are ranked by *in-sample* (2011-2019)
results only; the out-of-sample (2020-present) columns are shown alongside
as the honest check.

Usage:
    python scripts/research_options_structures.py [--stocks]
Outputs:
    data/results/options_structures.csv
"""

import sys

sys.path.insert(0, ".")

import argparse
import itertools
from pathlib import Path

import pandas as pd

from backtesting.market_data import load_universe, split_factor
from backtesting.options_backtest import ExitRules, OptionsSwingBacktester, StructureSpec
from backtesting.options_pricing import atm_iv_series
from trading.features import build_features

SPLIT = "2020-01-01"
START = "2011-01-01"
OUT = Path("data/results")

ENTRIES = {
    "rsi2<10&up": lambda f: (f.rsi2 < 10) & f.above200,
    "rsi2<5&up": lambda f: (f.rsi2 < 5) & f.above200,
    "dip_composite": lambda f: f.above200 & (
        (f.rsi2 < 10) | ((f.ibs < 0.25) & (f.rsi2 < 15)) | (f.bb_z < -2) | (f.pullback_atr >= 2)
    ),
    "weekly_up&rsi2<10": lambda f: f.weekly_up & (f.rsi2 < 10),
}

STRUCTURES = {
    "long_call_30d": StructureSpec("long_call", dte=30, long_delta=0.50),
    "cds_21d": StructureSpec("call_debit_spread", dte=21, long_delta=0.55, short_delta=0.30),
    "cds_14d": StructureSpec("call_debit_spread", dte=14, long_delta=0.50, short_delta=0.25),
    "pcs_d30_14d": StructureSpec("put_credit_spread", dte=14, short_delta=0.30, width_pct=0.02),
    "pcs_d40_10d": StructureSpec("put_credit_spread", dte=10, short_delta=0.40, width_pct=0.02),
    "pcs_d20_21d": StructureSpec("put_credit_spread", dte=21, short_delta=0.20, width_pct=0.02),
}

EXITS = {
    "fixed3": ExitRules(min_hold=3, max_hold=3, profit_target=None, signal_exit=False),
    "fixed5": ExitRules(min_hold=5, max_hold=5, profit_target=None, signal_exit=False),
    "sma5_min1_max7_t50": ExitRules(min_hold=1, max_hold=7, profit_target=0.5, signal_exit=True),
    "sma5_min3_max7_t60": ExitRules(min_hold=3, max_hold=7, profit_target=0.6, signal_exit=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", action="store_true", help="use AAPL/NVDA instead of ETFs")
    args = ap.parse_args()
    symbols = ["AAPL", "NVDA"] if args.stocks else ["SPY", "QQQ", "IWM"]

    OUT.mkdir(parents=True, exist_ok=True)
    data = load_universe(symbols + ["SPY", "VIX"])
    feats = {s: build_features(data[s], data["VIX"]) for s in symbols}
    # Strike grid / min tick are set at the actually-traded (split-unadjusted) price
    scale = {s: split_factor(data[s].index, s) for s in symbols}
    ivs = {s: atm_iv_series(data[s], data["VIX"], data["SPY"]) for s in symbols}
    # Connors exit: close back above the 5-day SMA
    exit_sig = {s: feats[s].close > feats[s].sma5 for s in symbols}
    # Priority: deepest oversold first
    prio = {s: -feats[s].rsi2 for s in symbols}

    rows = []
    for (en, efn), (sn, spec), (xn, xr) in itertools.product(
        ENTRIES.items(), STRUCTURES.items(), EXITS.items()
    ):
        entries = {s: efn(feats[s]) for s in symbols}
        bt = OptionsSwingBacktester(spec, xr, risk_per_trade=0.05, max_concurrent=3)
        res = bt.run(feats, ivs, entries, exit_sig, 1, prio, start=START, price_scale=scale)
        row = {"entry": en, "structure": sn, "exit": xn}
        for tag, (a, b) in {"IS": (START, SPLIT), "OOS": (SPLIT, None)}.items():
            st = res.stats(a, b)
            for k in ("trades", "win_rate", "avg_ret", "profit_factor", "avg_days",
                      "cagr", "max_drawdown", "sharpe"):
                row[f"{k}_{tag}"] = st.get(k)
        rows.append(row)
        print(f"{en:<18} {sn:<13} {xn:<20} IS: n={row['trades_IS']:>3} "
              f"win={row['win_rate_IS'] or 0:.2f} avg={row['avg_ret_IS'] or 0:+.3f} "
              f"cagr={row['cagr_IS']:+.3f} dd={row['max_drawdown_IS']:+.3f} sh={row['sharpe_IS']:.2f} | "
              f"OOS: n={row['trades_OOS']:>3} win={row['win_rate_OOS'] or 0:.2f} "
              f"avg={row['avg_ret_OOS'] or 0:+.3f} cagr={row['cagr_OOS']:+.3f} "
              f"dd={row['max_drawdown_OOS']:+.3f} sh={row['sharpe_OOS']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    tag = "stocks" if args.stocks else "etf"
    df.to_csv(OUT / f"options_structures_{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== Top 15 by IN-SAMPLE Sharpe (OOS shown for honesty) ===")
    cols = ["entry", "structure", "exit", "trades_IS", "win_rate_IS", "avg_ret_IS", "sharpe_IS",
            "trades_OOS", "win_rate_OOS", "avg_ret_OOS", "cagr_OOS", "max_drawdown_OOS", "sharpe_OOS"]
    print(df.sort_values("sharpe_IS", ascending=False)[cols].head(15).round(3).to_string(index=False))
    print("\n=== By structure (median across entries/exits) ===")
    print(df.groupby("structure")[["win_rate_IS", "sharpe_IS", "win_rate_OOS", "sharpe_OOS"]]
          .median().round(3).to_string())


if __name__ == "__main__":
    main()
