"""Audit: does the MCMC forecast actually predict short-horizon returns?

Compares the original GBM-based ``MCMCIndicator`` with the new
``RegimeMCMC`` (Markov chain over stretch x volatility states) on
SPY, QQQ, IWM, AAPL, NVDA, walking forward one day at a time with no
lookahead. Reports, separately for in-sample (2011-2019) and out-of-sample
(2020-present):

- IC: Spearman rank correlation between the forecast and the realized
  forward return (0 = no skill; 0.03-0.05 is already useful daily).
- Directional hit rate of the model's long/short calls.
- Calibration of RegimeMCMC's P(up) by bucket.

Usage:
    python scripts/research_mcmc_audit.py
"""

import sys

sys.path.insert(0, ".")

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.market_data import load_universe
from trading.features import forward_returns
from trading.indicator import MCMCIndicator
from trading.regime_mcmc import RegimeMCMC

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "NVDA"]
SPLIT = "2020-01-01"
START = "2011-01-01"
OUT = Path("data/results")


def old_indicator_series(close: pd.Series, window: int = 120) -> pd.DataFrame:
    ind = MCMCIndicator(n_simulations=2000, n_steps=30, slope_threshold=15.0, enable_gpu=False)
    np.random.seed(0)
    rows = []
    t0 = close.index.searchsorted(pd.Timestamp(START))
    for t in range(max(t0, window), len(close)):
        hist = close.iloc[t - window + 1 : t + 1].to_frame("Close")
        sig = ind.generate_signal("X", hist, "1d")
        rows.append((close.index[t], sig["slope_degrees"], sig["signal_strength"],
                     sig["suggested_action"]))
    return pd.DataFrame(rows, columns=["date", "slope", "strength", "action"]).set_index("date")


def ic(x: pd.Series, y: pd.Series) -> float:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 30:
        return float("nan")
    return float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()))


def split(df: pd.DataFrame):
    return {"IS": df.loc[:SPLIT], "OOS": df.loc[SPLIT:]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_universe(SYMBOLS)
    results = {}
    calib_rows = []

    for sym in SYMBOLS:
        close = data[sym]["Close"]
        fwd = forward_returns(close, (1, 3, 5))
        print(f"\n=== {sym} ===")

        old = old_indicator_series(close).join(fwd)
        res = {}
        for name, part in split(old).items():
            buys = part[part.action == "BUY"]
            sells = part[part.action == "SELL"]
            res[f"old_{name}"] = {
                "ic_slope_fwd3": ic(part.slope, part.fwd3),
                "ic_slope_fwd5": ic(part.slope, part.fwd5),
                "n_buy": len(buys),
                "buy_hit3": float((buys.fwd3 > 0).mean()) if len(buys) else None,
                "n_sell": len(sells),
                "sell_hit3": float((sells.fwd3 < 0).mean()) if len(sells) else None,
                "base_up3": float((part.fwd3 > 0).mean()),
            }
            r = res[f"old_{name}"]
            print(f"  OLD {name}: IC(slope,fwd3)={r['ic_slope_fwd3']:+.3f} "
                  f"IC5={r['ic_slope_fwd5']:+.3f} | BUY n={r['n_buy']} hit={r['buy_hit3']} "
                  f"| SELL n={r['n_sell']} hit={r['sell_hit3']} | base up={r['base_up3']:.3f}")

        for h in (3, 5):
            new = RegimeMCMC(horizon=h, n_paths=3000).forecast_series(close, start=START)
            new = new.join(fwd)
            for name, part in split(new).items():
                long_ = part[part.p_up >= 0.60]
                short = part[part.p_up <= 0.45]
                r = {
                    "ic_exp": ic(part.exp_return, part[f"fwd{h}"]),
                    "ic_pup": ic(part.p_up, part[f"fwd{h}"]),
                    "n_long": len(long_),
                    "long_hit": float((long_[f"fwd{h}"] > 0).mean()) if len(long_) else None,
                    "long_mean": float(long_[f"fwd{h}"].mean()) if len(long_) else None,
                    "n_short": len(short),
                    "short_hit": float((short[f"fwd{h}"] < 0).mean()) if len(short) else None,
                    "base_up": float((part[f"fwd{h}"] > 0).mean()),
                    "base_mean": float(part[f"fwd{h}"].mean()),
                }
                res[f"new_h{h}_{name}"] = r
                print(f"  NEW h={h} {name}: IC(exp)={r['ic_exp']:+.3f} IC(p_up)={r['ic_pup']:+.3f} "
                      f"| p_up>=.60 n={r['n_long']} hit={r['long_hit'] and round(r['long_hit'],3)} "
                      f"mean={r['long_mean'] and round(r['long_mean']*100,2)}% "
                      f"| p_up<=.45 n={r['n_short']} hit={r['short_hit'] and round(r['short_hit'],3)} "
                      f"| base up={r['base_up']:.3f} mean={r['base_mean']*100:.2f}%")
                if h == 3:
                    b = pd.cut(part.p_up, [0, .45, .5, .55, .6, .65, 1.0])
                    g = part.groupby(b, observed=True)["fwd3"].agg(
                        n="size", realized_up=lambda x: (x > 0).mean(), mean="mean")
                    for k, row in g.iterrows():
                        calib_rows.append({"symbol": sym, "split": name, "bucket": str(k),
                                           **{c: float(row[c]) for c in g.columns}})
        results[sym] = res

    calib = pd.DataFrame(calib_rows)
    print("\n=== RegimeMCMC P(up, 3d) calibration (pooled) ===")
    pooled = calib.assign(w=calib.n * calib.realized_up, m=calib.n * calib["mean"]).groupby(
        ["split", "bucket"]).agg(n=("n", "sum"), w=("w", "sum"), m=("m", "sum"))
    pooled["realized_up"] = pooled.w / pooled.n
    pooled["mean_ret_%"] = 100 * pooled.m / pooled.n
    print(pooled[["n", "realized_up", "mean_ret_%"]].round(3))

    with open(OUT / "mcmc_audit.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved {OUT / 'mcmc_audit.json'}")


if __name__ == "__main__":
    main()
