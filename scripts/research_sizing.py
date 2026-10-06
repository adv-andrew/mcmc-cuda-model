"""Sizing menu for portfolio mode: option risk per trade vs return and drawdown.

Profit can always be raised by risking more per trade; the question is
what that costs in drawdown, and where returns stop improving. For each
option risk level, the core weight is set so everything fits in one
account: core = min(0.85, 1 - max_concurrent * risk - 0.03).

Also reports the Kelly fraction implied by the per-trade return
distribution. Full Kelly is an upper bound that assumes the distribution
is known exactly; with model risk, practitioners use a fraction of it.

Usage:
    python scripts/research_sizing.py
"""

import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from backtesting.market_data import load_long_history, load_universe, split_factor
from backtesting.options_backtest import summarize
from backtesting.options_pricing import SkewModel, atm_iv_series
from backtesting.portfolio import faber_signal, simulate_core_overlay
from trading.features import build_features
from trading.options_swing import SwingConfig, confidence_score, entry_signal, exit_signal

CORE = ["SPY", "QQQ", "IWM"]
END = "2026-10-03"
START = "1996-01-01"


def kelly(r: np.ndarray, grid=np.linspace(0.0, 1.0, 201)) -> float:
    """Fraction of equity per trade maximizing E[log(1 + f r)]."""
    r = np.asarray(r)
    vals = [np.mean(np.log1p(np.clip(f * r, -0.999999, None))) for f in grid]
    return float(grid[int(np.argmax(vals))])


def main() -> None:
    cfg = SwingConfig.from_yaml()
    vix = load_universe(["VIX"])["VIX"]
    data = {s: load_long_history(s) for s in CORE}
    data = {s: d[d.index < END] for s, d in data.items()}
    feats = {s: build_features(data[s], vix) for s in CORE}
    ivs = {s: atm_iv_series(data[s], vix, data["SPY"]) for s in CORE}
    ivs_rich = {s: atm_iv_series(data[s], vix, data["SPY"], vix_to_atm=1.0) for s in CORE}
    sig = {s: entry_signal(feats[s], cfg) for s in CORE}
    ext = {s: exit_signal(feats[s]) for s in CORE}
    prio = {s: confidence_score(feats[s]) + feats[s].pullback_atr for s in CORE}
    scale = {s: split_factor(data[s].index, s) for s in CORE}
    spy_tr, fs = data["SPY"]["AdjClose"], faber_signal(data["SPY"]["Close"])

    rows, trade_rets = [], None
    for risk in (0.03, 0.045, 0.06, 0.08, 0.10, 0.12):
        weight = min(0.85, 1 - cfg.max_concurrent * risk - 0.03)
        for harsh in (False, True):
            eq, tr = simulate_core_overlay(
                spy_tr, fs, feats, ivs_rich if harsh else ivs, sig, ext, cfg.structure(),
                cfg.exits(), weight, risk, cfg.max_concurrent, prio, scale, start=START,
                skew=SkewModel(call_slope=0.25) if harsh else None,
                cost_mult=1.5 if harsh else 1.0)
            if risk == cfg.risk_per_trade and not harsh:
                trade_rets = tr.ret_on_risk.to_numpy()
            st = summarize(pd.DataFrame(), eq)
            yr = pd.concat([eq.iloc[:1], eq.resample("YE").last()]).pct_change().iloc[1:]
            rows.append({"risk/trade": risk, "core": weight, "pricing": "harsh" if harsh else "base",
                         "CAGR": st["cagr"], "maxDD": st["max_drawdown"],
                         "Sharpe(ex)": st["sharpe_excess"], "worst yr": yr.min(),
                         "losing yrs": int((yr < 0).sum())})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for pricing in ("base", "harsh"):
        print(f"\n=== {pricing} option pricing, 1996-2026 single account ===")
        sub = df[df.pricing == pricing].drop(columns="pricing")
        print(sub.to_string(index=False, formatters={
            "risk/trade": "{:.1%}".format, "core": "{:.0%}".format, "CAGR": "{:+.1%}".format,
            "maxDD": "{:+.1%}".format, "Sharpe(ex)": "{:.2f}".format,
            "worst yr": "{:+.1%}".format}))

    k = kelly(trade_rets)
    print(f"\nKelly fraction from {len(trade_rets)} trades (mean {trade_rets.mean():+.1%}, "
          f"sd {trade_rets.std():.2f}): full {k:.0%}, half {k / 2:.0%}, quarter {k / 4:.0%}")
    # Smaller edge, same dispersion: shift every trade down by 40% of the mean
    hk = kelly(trade_rets - 0.4 * trade_rets.mean())
    print(f"If the true edge is 40% smaller (closer to harsh pricing): full Kelly {hk:.0%}")
    df.to_csv("data/results/sizing_menu.csv", index=False)


if __name__ == "__main__":
    main()
