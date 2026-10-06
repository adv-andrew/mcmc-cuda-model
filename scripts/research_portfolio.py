"""Put the idle capital to work: trend core + dip overlay.

Variants (fixed before running; all idle cash earns T-bills):
    V0  SPY buy & hold (dividends included)
    V1  Trend core: SPY while its month-end close > 10-month SMA, else T-bills
    V2  Options dip overlay alone (current strategy, 6% premium per trade)
    V3  Shares dip overlay alone (3 slots x 33%)
    V4  85% trend core + 15% options sleeve (each trade = 6% of total equity)
    V5  85% SPY buy & hold + 15% options sleeve (no trend filter)
    V6  70% trend core + 30% options sleeve (each trade = 12% of total; a
        sizing choice, shown for the profit/risk trade-off)
V4 is also run with harsh option pricing (IV +11%, steep call skew, 1.5x
bid/ask). Eras: 1996-2010 (before the dip rules were designed), 2011-2019,
2020-2026. The Faber rule was published in 2007, so 2007+ is out of sample
for it as well.

Usage:
    python scripts/research_portfolio.py
"""

import sys

sys.path.insert(0, ".")

from pathlib import Path

import pandas as pd

from backtesting.market_data import load_long_history, load_universe, split_factor
from backtesting.options_backtest import OptionsSwingBacktester, summarize
from backtesting.options_pricing import SkewModel, atm_iv_series
from backtesting.portfolio import blend, equity_to_returns, faber_signal, trend_core_returns
from backtesting.shares_backtest import SharesSwingBacktester
from trading.features import build_features
from trading.options_swing import SwingConfig, confidence_score, entry_signal, exit_signal

CORE = ["SPY", "QQQ", "IWM"]
END = "2026-10-03"  # last day covered by the dividend-adjusted release data
START = "1996-01-01"
ERAS = {"1996-2010": ("1996-01-01", "2011-01-01"), "2011-2019": ("2011-01-01", "2020-01-01"),
        "2020-2026": ("2020-01-01", END), "ALL 1996-2026": ("1996-01-01", END)}
OUT = Path("data/results")


def era_table(eq: pd.Series) -> dict:
    out = {}
    for name, (a, b) in ERAS.items():
        e = eq[(eq.index >= a) & (eq.index < b)]
        st = summarize(pd.DataFrame(), e)
        yr = pd.concat([e.iloc[:1], e.resample("YE").last()]).pct_change().iloc[1:]
        out[name] = {"CAGR": st["cagr"], "maxDD": st["max_drawdown"],
                     "Sharpe(ex)": st["sharpe_excess"], "worst_yr": yr.min(),
                     "losing": f"{int((yr < 0).sum())}/{len(yr)}"}
    return out


def show(label: str, eq: pd.Series) -> dict:
    t = era_table(eq)
    print(f"\n{label}")
    for era, r in t.items():
        print(f"   {era:<14} CAGR={r['CAGR']:+6.1%}  maxDD={r['maxDD']:+6.1%}  "
              f"Sharpe(excess)={r['Sharpe(ex)']:5.2f}  worst year={r['worst_yr']:+6.1%}  "
              f"losing years={r['losing']}")
    return t


def main() -> None:
    cfg = SwingConfig.from_yaml()
    vix = load_universe(["VIX"])["VIX"]
    data = {s: load_long_history(s).loc[:END] for s in CORE}
    data = {s: d[d.index < END] for s, d in data.items()}
    feats = {s: build_features(data[s], vix) for s in CORE}
    ivs = {s: atm_iv_series(data[s], vix, data["SPY"]) for s in CORE}
    sig = {s: entry_signal(feats[s], cfg) for s in CORE}
    ext = {s: exit_signal(feats[s]) for s in CORE}
    prio = {s: confidence_score(feats[s]) + feats[s].pullback_atr for s in CORE}
    scale = {s: split_factor(data[s].index, s) for s in CORE}

    def options_sleeve(risk: float, harsh: bool = False) -> pd.Series:
        bt = OptionsSwingBacktester(
            cfg.structure(), cfg.exits(), risk_per_trade=risk,
            max_concurrent=cfg.max_concurrent, etfs=tuple(CORE), cash_yield=True,
            cost_multiplier=1.5 if harsh else 1.0,
            skew=SkewModel(call_slope=0.25) if harsh else None)
        if harsh:
            ivs_h = {s: atm_iv_series(data[s], vix, data["SPY"], vix_to_atm=1.0) for s in CORE}
            return bt.run(feats, ivs_h, sig, ext, 1, prio, start=START,
                          price_scale=scale).equity
        return bt.run(feats, ivs, sig, ext, 1, prio, start=START, price_scale=scale).equity

    spy_tr = data["SPY"]["AdjClose"].loc[START:]
    core_ret = trend_core_returns(spy_tr, faber_signal(data["SPY"]["Close"]))
    spy_ret = spy_tr.pct_change().fillna(0.0)

    curves = {}
    curves["V0"] = 100_000 * (1 + spy_ret).cumprod()
    curves["V1"] = 100_000 * (1 + core_ret).cumprod()
    curves["V2"] = options_sleeve(cfg.risk_per_trade)
    curves["V3"] = SharesSwingBacktester(cfg.exits(), cfg.shares_position_frac, 3, 3.0,
                                         cash_yield=True).run(
        {s: feats[s].close for s in CORE}, sig, ext, prio, start=START).equity
    sleeve_15 = equity_to_returns(options_sleeve(cfg.risk_per_trade / 0.15))
    sleeve_15_h = equity_to_returns(options_sleeve(cfg.risk_per_trade / 0.15, harsh=True))
    sleeve_30 = equity_to_returns(options_sleeve(2 * cfg.risk_per_trade / 0.30))
    curves["V4"] = blend({"core": core_ret, "dip": sleeve_15}, {"core": 0.85, "dip": 0.15})
    curves["V4 harsh"] = blend({"core": core_ret, "dip": sleeve_15_h},
                               {"core": 0.85, "dip": 0.15})
    curves["V5"] = blend({"spy": spy_ret, "dip": sleeve_15}, {"spy": 0.85, "dip": 0.15})
    curves["V6"] = blend({"core": core_ret, "dip": sleeve_30}, {"core": 0.70, "dip": 0.30})

    labels = {
        "V0": "V0 SPY buy & hold",
        "V1": "V1 trend core alone (SPY > 10-month SMA, else T-bills)",
        "V2": f"V2 options dip overlay alone ({cfg.risk_per_trade:.0%} per trade)",
        "V3": "V3 shares dip overlay alone (3 x 33%)",
        "V4": "V4 85% trend core + 15% options sleeve (6% of total per trade)",
        "V4 harsh": "V4 with harsh option pricing (IV +11%, call skew .25, 1.5x bid/ask)",
        "V5": "V5 85% SPY buy & hold + 15% options sleeve",
        "V6": "V6 70% trend core + 30% options sleeve (12% of total per trade)",
    }
    tables = {k: show(labels[k], v) for k, v in curves.items()}
    rows = [{"variant": labels[k], "era": era, **r} for k, t in tables.items()
            for era, r in t.items()]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "portfolio_variants.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "portfolio_equity.csv")
    r1, r2 = equity_to_returns(curves["V1"]), sleeve_15
    print(f"\nDaily-return correlation, trend core vs options sleeve: {r1.corr(r2):.2f}")
    print(f"Trend core invested {faber_signal(data['SPY']['Close']).loc[START:].mean():.0%} "
          f"of days")


if __name__ == "__main__":
    main()
