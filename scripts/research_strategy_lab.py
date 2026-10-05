"""Strategy lab: event-study every catalog signal across tickers and horizons.

For each signal x ticker x holding period (1, 2, 3, 5, 10 days) we take
*non-overlapping* trades (after an entry, the next entry is allowed only
once the previous trade has exited) and measure the underlying's forward
return from the signal-day close. Results are split into in-sample
(2010-2019) and out-of-sample (2020-present) so a signal has to work on
data it was not chosen on.

A robustness variant enters at the *next day's open* instead of the signal
close, to check the edge survives a realistic execution delay.

Usage:
    python scripts/research_strategy_lab.py
Outputs:
    data/results/strategy_lab.csv
"""

import sys

sys.path.insert(0, ".")

from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.market_data import load_universe
from trading.features import build_features
from trading.swing_signals import SIGNALS

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "NVDA"]
ETFS = {"SPY", "QQQ", "IWM"}
HORIZONS = (1, 2, 3, 5, 10)
SPLIT = pd.Timestamp("2020-01-01")
START = pd.Timestamp("2011-01-01")  # leave a warm-up year for SMA200 etc.
OUT = Path("data/results")


def non_overlapping(mask: np.ndarray, h: int) -> np.ndarray:
    """Indices of entries such that each trade (h bars) finishes before the next."""
    out = []
    nxt = 0
    for i in np.flatnonzero(mask):
        if i >= nxt:
            out.append(i)
            nxt = i + h
    return np.asarray(out, dtype=int)


def summarize(rets: np.ndarray, direction: int) -> dict:
    r = rets * direction
    n = len(r)
    if n == 0:
        return {"n": 0, "mean_%": np.nan, "win": np.nan, "t": np.nan, "pf": np.nan}
    mean = r.mean()
    sd = r.std(ddof=1) if n > 1 else np.nan
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return {
        "n": n,
        "mean_%": 100 * mean,
        "win": float((r > 0).mean()),
        "t": t,
        "pf": gains / losses if losses > 0 else np.inf,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_universe(SYMBOLS + ["VIX"])
    vix = data["VIX"]
    rows = []
    for sym in SYMBOLS:
        df = data[sym]
        f = build_features(df, vix)
        f = f.loc[START:]
        close = f.close.to_numpy()
        opens = f.open.to_numpy()
        dates = f.index
        is_mask = dates < SPLIT
        for spec in SIGNALS.values():
            sig = spec.rule(f).to_numpy()
            for h in HORIZONS:
                valid = np.arange(len(f)) < len(f) - h - 1
                for period, pmask in (("IS", is_mask), ("OOS", ~is_mask)):
                    idx = non_overlapping(sig & valid & pmask, h)
                    fwd_close = close[idx + h] / close[idx] - 1.0
                    fwd_open = close[idx + h] / opens[idx + 1] - 1.0
                    s = summarize(fwd_close, spec.direction)
                    so = summarize(fwd_open, spec.direction)
                    rows.append({
                        "symbol": sym, "group": "ETF" if sym in ETFS else "STOCK",
                        "signal": spec.name, "family": spec.family,
                        "direction": spec.direction, "h": h, "period": period,
                        **s, "mean_open_%": so["mean_%"], "win_open": so["win"],
                    })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "strategy_lab.csv", index=False)

    # Pooled view over ETFs: trade-weighted mean and win rate, with a pooled t
    def pool(g):
        n = g.n.sum()
        if n == 0:
            return pd.Series({"n": 0, "mean_%": np.nan, "win": np.nan, "mean_open_%": np.nan})
        return pd.Series({
            "n": n,
            "mean_%": (g["mean_%"] * g.n).sum() / n,
            "win": (g.win * g.n).sum() / n,
            "mean_open_%": (g["mean_open_%"] * g.n).sum() / n,
        })

    for group in ("ETF", "STOCK"):
        sub = res[res.group == group]
        pooled = sub.groupby(["signal", "h", "period"]).apply(pool, include_groups=False).unstack("period")
        pooled.columns = [f"{a}_{b}" for a, b in pooled.columns]
        base = pooled.xs("always_long", level="signal")
        for h in (3, 5):
            tbl = pooled.xs(h, level="h").copy()
            tbl["edge_IS_%"] = tbl["mean_%_IS"] - base.loc[h, "mean_%_IS"] * np.sign(1)
            tbl["edge_OOS_%"] = tbl["mean_%_OOS"] - base.loc[h, "mean_%_OOS"]
            tbl = tbl.sort_values("mean_%_OOS", ascending=False)
            cols = ["n_IS", "win_IS", "mean_%_IS", "n_OOS", "win_OOS", "mean_%_OOS",
                    "mean_open_%_OOS"]
            print(f"\n===== {group} pooled, hold {h} days (direction-adjusted returns) =====")
            print(tbl[cols].round(3).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
