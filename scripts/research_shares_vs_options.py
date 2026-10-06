"""Shares vs options: which way of trading the dip signal is most reliable?

The timing edge replicated on 16 ETFs (t ~4-5 per group) but options only
monetize it on SPY/QQQ/IWM. This compares, from 2000 to 2026:

- Options, core 3 ETFs (the current options strategy)
- Shares, core 3 ETFs (3 slots x 33% of equity)
- Shares, broad 16 ETFs (4 slots x 25%; never more than 100% invested)
- Hybrid: half the capital in each of options-core and shares-broad
- SPY buy & hold (dividends included)

Sizing and costs were fixed before running: 3 bps per side for shares,
stressed at 10 bps and with entries one day late. Exits are the
strategy's existing rules.

Usage:
    python scripts/research_shares_vs_options.py
"""

import sys

sys.path.insert(0, ".")

from pathlib import Path

import pandas as pd

from backtesting.market_data import load_long_history, load_universe, split_factor
from backtesting.options_backtest import OptionsSwingBacktester, summarize
from backtesting.options_pricing import atm_iv_series
from backtesting.shares_backtest import SharesSwingBacktester
from trading.features import build_features
from trading.options_swing import SwingConfig, confidence_score, entry_signal, exit_signal

CORE = ["SPY", "QQQ", "IWM"]
BROAD = CORE + ["DIA", "MDY", "IJR", "RSP", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU",
                "XLV", "XLY", "EFA", "EEM"]
PERIODS = {"2000-2010": ("2000-01-01", "2011-01-01"), "2011-2019": ("2011-01-01", "2020-01-01"),
           "2020-2026": ("2020-01-01", None), "ALL 2000-2026": ("2000-01-01", None)}
OUT = Path("data/results")


def describe(eq: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    out = {}
    for name, (a, b) in PERIODS.items():
        e = eq.loc[a:b] if b else eq.loc[a:]
        e = e[e.index < pd.Timestamp(b)] if b else e
        st = summarize(pd.DataFrame(), e)
        yr = pd.concat([e.iloc[:1], e.resample("YE").last()]).pct_change().iloc[1:]
        row = {"CAGR": st["cagr"], "maxDD": st["max_drawdown"], "Sharpe": st["sharpe"],
               "losing_yrs": f"{int((yr < 0).sum())}/{len(yr)}", "worst_yr": yr.min()}
        if trades is not None and len(trades):
            t = trades[(trades.entry_date >= a) & ((trades.entry_date < b) if b else True)]
            yrs = (e.index[-1] - e.index[0]).days / 365.25
            row.update(trades_yr=len(t) / yrs, win=(t.ret_on_risk > 0).mean())
        out[name] = row
    return out


def show(label: str, d: dict) -> None:
    print(f"\n{label}")
    for p, r in d.items():
        extra = (f"  trades/yr={r['trades_yr']:5.1f}  win={r['win']:.0%}"
                 if "trades_yr" in r else "")
        print(f"   {p:<14} CAGR={r['CAGR']:+6.1%}  maxDD={r['maxDD']:+6.1%}  "
              f"Sharpe={r['Sharpe']:5.2f}  losing years={r['losing_yrs']:<5} "
              f"worst year={r['worst_yr']:+.1%}{extra}")


def main() -> None:
    cfg = SwingConfig.from_yaml()
    vix = load_universe(["VIX"])["VIX"]
    data = {s: load_long_history(s) for s in BROAD}
    feats = {s: build_features(data[s], vix) for s in BROAD}
    sig = {s: entry_signal(feats[s], cfg) for s in BROAD}
    ext = {s: exit_signal(feats[s]) for s in BROAD}
    prio = {s: confidence_score(feats[s]) + feats[s].pullback_atr for s in BROAD}
    close = {s: feats[s]["close"] for s in BROAD}
    exits = cfg.exits()
    curves = {}

    # Options, core (current strategy)
    ivs = {s: atm_iv_series(data[s], vix, data["SPY"]) for s in CORE}
    bt = OptionsSwingBacktester(cfg.structure(), exits, risk_per_trade=cfg.risk_per_trade,
                                max_concurrent=cfg.max_concurrent, etfs=tuple(CORE))
    sub = lambda d, ss: {s: d[s] for s in ss}  # noqa: E731
    r_opt = bt.run(sub(feats, CORE), ivs, sub(sig, CORE), sub(ext, CORE), 1, sub(prio, CORE),
                   start="2000-01-01",
                   price_scale={s: split_factor(data[s].index, s) for s in CORE})
    curves["options core"] = r_opt.equity
    show(f"OPTIONS, core 3 ETFs ({cfg.structure().label()}, {cfg.max_concurrent} x "
         f"{cfg.risk_per_trade:.1%} premium)", describe(r_opt.equity, r_opt.trade_frame()))

    variants = {
        "SHARES, core 3 ETFs (3 x 33%)": (CORE, 1 / 3, 3, 3.0, 0),
        "SHARES, broad 16 ETFs (4 x 25%)": (BROAD, 0.25, 4, 3.0, 0),
        "  stress: broad, 10 bps per side": (BROAD, 0.25, 4, 10.0, 0),
        "  stress: broad, enter 1 day late": (BROAD, 0.25, 4, 3.0, 1),
    }
    for label, (syms, frac, conc, bps, delay) in variants.items():
        entries = {s: sig[s].shift(delay, fill_value=False) for s in syms}
        res = SharesSwingBacktester(exits, frac, conc, bps).run(
            sub(close, syms), entries, sub(ext, syms), sub(prio, syms), start="2000-01-01")
        curves[label.strip()] = res.equity
        show(label, describe(res.equity, res.trade_frame()))
        if label.startswith("SHARES, broad"):
            tf = res.trade_frame()
            exposure = (tf.days_held.sum() * frac) / len(res.equity)
            print(f"   average capital invested: {exposure:.0%} of equity; "
                  f"avg trade {tf.ret_on_risk.mean():+.2%}, median hold {tf.days_held.median():.0f}d")
            tf.to_csv(OUT / "shares_broad_trades.csv", index=False)

    # Hybrid: half capital in each sleeve, rebalanced daily
    a = curves["options core"].pct_change()
    b = curves["SHARES, broad 16 ETFs (4 x 25%)"].pct_change()
    hyb = (1 + (0.5 * a + 0.5 * b).fillna(0)).cumprod() * 100_000
    curves["hybrid"] = hyb
    show("HYBRID: 50% options-core + 50% shares-broad", describe(hyb))
    print(f"   correlation of daily returns, options-core vs shares-broad: {a.corr(b):.2f}")

    spy = data["SPY"]["AdjClose"].loc["2000-01-01":"2026-10-02"]
    show("SPY buy & hold (dividends included)", describe(spy / spy.iloc[0] * 100_000))
    sb = curves["SHARES, broad 16 ETFs (4 x 25%)"].pct_change()
    print(f"\nCorrelation with SPY daily returns: shares-broad "
          f"{sb.corr(spy.pct_change().reindex(sb.index)):.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).to_csv(OUT / "shares_vs_options_equity.csv")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
