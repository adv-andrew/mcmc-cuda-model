"""Single-account validation of the trend core + options overlay (V4).

``scripts/research_portfolio.py`` blends sleeves with daily rebalancing, a
convenient approximation. This script simulates one real account instead:
SPY shares for the core (rebalanced monthly and on signal flips), option
premiums paid from cash and sized off total equity, interest on idle cash.

It then checks that the result does not hinge on exact settings (Faber
lookback 8/10/12 months, core weight 80/85/90%) or on optimistic option
pricing (IV +11%, steep call skew, 1.5x bid/ask).

Usage:
    python scripts/research_core_overlay.py
"""

import sys

sys.path.insert(0, ".")

from pathlib import Path

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
ERAS = {"1996-2010": ("1996-01-01", "2011-01-01"), "2011-2019": ("2011-01-01", "2020-01-01"),
        "2020-2026": ("2020-01-01", END), "ALL": ("1996-01-01", END)}
OUT = Path("data/results")


def row(label: str, eq: pd.Series) -> dict:
    out = {"variant": label}
    for era, (a, b) in ERAS.items():
        e = eq[(eq.index >= a) & (eq.index < b)]
        st = summarize(pd.DataFrame(), e)
        out[f"{era} CAGR"] = st["cagr"]
        if era == "ALL":
            yr = pd.concat([e.iloc[:1], e.resample("YE").last()]).pct_change().iloc[1:]
            out.update({"maxDD": st["max_drawdown"], "Sharpe(ex)": st["sharpe_excess"],
                        "worst yr": yr.min(), "losing yrs": int((yr < 0).sum())})
    return out


def main() -> None:
    cfg = SwingConfig.from_yaml()
    vix = load_universe(["VIX"])["VIX"]
    data = {s: load_long_history(s) for s in CORE}
    data = {s: d[d.index < END] for s, d in data.items()}
    feats = {s: build_features(data[s], vix) for s in CORE}
    ivs = {s: atm_iv_series(data[s], vix, data["SPY"]) for s in CORE}
    ivs_rich = {s: atm_iv_series(data[s], vix, data["SPY"], vix_to_atm=1.0) for s in CORE}
    sig = {s: entry_signal(feats[s], cfg) for s in CORE}
    none = {s: sig[s] & False for s in CORE}
    ext = {s: exit_signal(feats[s]) for s in CORE}
    prio = {s: confidence_score(feats[s]) + feats[s].pullback_atr for s in CORE}
    scale = {s: split_factor(data[s].index, s) for s in CORE}
    spy_tr, spy_close = data["SPY"]["AdjClose"], data["SPY"]["Close"]

    def run(label, months=10, weight=0.85, entries=sig, iv=ivs, harsh=False, risk=None):
        eq, tr = simulate_core_overlay(
            spy_tr, faber_signal(spy_close, months), feats, iv, entries, ext,
            cfg.structure(), cfg.exits(), weight, risk or cfg.risk_per_trade,
            cfg.max_concurrent, prio, scale, start=START,
            skew=SkewModel(call_slope=0.25) if harsh else None,
            cost_mult=1.5 if harsh else 1.0)
        return row(label, eq), eq

    rows, curves = [], {}
    for label, kw in {
        "V4 default (10m, 85%)": {},
        "  Faber 8 months": {"months": 8},
        "  Faber 12 months": {"months": 12},
        "  core weight 80%": {"weight": 0.80},
        "  core weight 90%": {"weight": 0.90},
        "  harsh option pricing": {"iv": ivs_rich, "harsh": True},
        "  harsh + Faber 12m + 80%": {"iv": ivs_rich, "harsh": True, "months": 12,
                                       "weight": 0.80},
        "core only, 85% (no options)": {"entries": none},
        "core only, 100% (no options)": {"entries": none, "weight": 1.0},
    }.items():
        r, eq = run(label, **kw)
        rows.append(r)
        curves[label.strip()] = eq
    spy = spy_tr.loc[START:]
    rows.append(row("SPY buy & hold", spy / spy.iloc[0] * 100_000))

    pd.set_option("display.width", 250)
    df = pd.DataFrame(rows).set_index("variant")
    fmt = {c: "{:+.1%}".format for c in df.columns if "CAGR" in c or c in ("maxDD", "worst yr")}
    fmt["Sharpe(ex)"] = "{:.2f}".format
    print(df.to_string(formatters=fmt))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "core_overlay_validation.csv")
    pd.DataFrame(curves).to_csv(OUT / "core_overlay_equity.csv")


if __name__ == "__main__":
    main()
