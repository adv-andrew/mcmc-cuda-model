"""Backtest + stress-test the Options Swing Strategy (trading/options_swing.py).

Prints:
  1. Headline in-sample (2011-2019) vs out-of-sample (2020-present) stats
  2. Per-year results, per-symbol results, exit-reason mix
  3. Results by confidence tier (does a higher score mean better trades?)
  4. Stress tests: 2x costs, 1-day-late entries, richer IV, steeper skew,
     different sizing / concurrency
  5. Parameter-neighbourhood check (is the chosen setting a lucky spike?)

Usage:
    python scripts/backtest_options_swing.py [--quick]
Outputs:
    data/results/options_swing_backtest.json
    data/results/options_swing_trades.csv
    data/results/options_swing_equity.csv
"""

import sys

sys.path.insert(0, ".")

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.market_data import load_universe
from backtesting.options_backtest import OptionsSwingBacktester, monte_carlo_equity, summarize
from backtesting.options_pricing import SkewModel, atm_iv_series
from trading.features import build_features
from trading.options_swing import (
    SwingConfig,
    confidence_score,
    confidence_tier,
    entry_signal,
    exit_signal,
)

START = "2011-01-01"
SPLIT = "2020-01-01"
OUT = Path("data/results")


def fmt(st: dict) -> str:
    if not st.get("trades"):
        return "no trades"
    return (f"n={st['trades']:>3} win={st['win_rate']:.0%} avg={st['avg_ret']:+.1%} "
            f"PF={st['profit_factor']:.2f} CAGR={st.get('cagr', 0):+.1%} "
            f"maxDD={st.get('max_drawdown', 0):+.1%} Sharpe={st.get('sharpe', 0):.2f}")


class Runner:
    def __init__(self, cfg: SwingConfig):
        self.cfg = cfg
        syms = list(cfg.symbols)
        data = load_universe(syms + ["SPY", "VIX"])
        self.data = data
        self.feats = {s: build_features(data[s], data["VIX"]) for s in syms}
        self.scores = {s: confidence_score(self.feats[s]) for s in syms}

    def run(self, cfg: SwingConfig, cost_mult=1.0, delay=0, vix_to_atm=0.90,
            skew=None, min_score=0, start=START, end=None):
        syms = list(cfg.symbols)
        ivs = {s: atm_iv_series(self.data[s], self.data["VIX"], self.data["SPY"],
                                vix_to_atm=vix_to_atm) for s in syms}
        entries = {
            s: (entry_signal(self.feats[s], cfg) & (self.scores[s] >= min_score))
            .shift(delay, fill_value=False)
            for s in syms
        }
        exits = {s: exit_signal(self.feats[s]) for s in syms}
        prio = {s: self.scores[s] + self.feats[s].pullback_atr for s in syms}
        bt = OptionsSwingBacktester(cfg.structure(), cfg.exits(),
                                    risk_per_trade=cfg.risk_per_trade,
                                    max_concurrent=cfg.max_concurrent,
                                    skew=skew, cost_multiplier=cost_mult)
        return bt.run(self.feats, ivs, entries, exits, 1, prio, start=start, end=end)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the neighbourhood grid")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    cfg = SwingConfig.from_yaml()
    runner = Runner(cfg)
    res = runner.run(cfg)
    tf = res.trade_frame()
    tf["score"] = [runner.scores[r.symbol].loc[r.entry_date] for r in tf.itertuples()]
    tf["tier"] = [confidence_tier(s, cfg) for s in tf.score]
    tf["period"] = np.where(tf.entry_date < pd.Timestamp(SPLIT), "IS", "OOS")

    print("=" * 78)
    print("OPTIONS SWING STRATEGY - call debit spreads on dips in uptrends (SPY/QQQ/IWM)")
    print("=" * 78)
    print(f"Structure: {cfg.structure().label()}  | exits: +{cfg.profit_target:.0%} of max "
          f"profit, close>SMA5 after {cfg.min_hold}d, max {cfg.max_hold}d")
    print(f"Sizing: {cfg.risk_per_trade:.0%} of equity at risk per trade, "
          f"max {cfg.max_concurrent} concurrent")
    full, is_, oos = res.stats(START), res.stats(START, SPLIT), res.stats(SPLIT)
    print(f"\n  FULL  {START[:4]}-now : {fmt(full)}")
    print(f"  IS    2011-2019 : {fmt(is_)}")
    print(f"  OOS   2020-now  : {fmt(oos)}")

    spy = runner.data["SPY"]["AdjClose"].loc[START:]
    spy_eq = spy / spy.iloc[0]
    spy_st = summarize(pd.DataFrame(), spy_eq)
    print(f"  (SPY buy&hold   : CAGR={spy_st['cagr']:+.1%} maxDD={spy_st['max_drawdown']:+.1%} "
          f"Sharpe={spy_st['sharpe']:.2f})")
    corr = res.equity.pct_change().corr(spy.pct_change().reindex(res.equity.index))
    print(f"  Correlation of daily strategy returns with SPY: {corr:.2f}")

    print("\n--- By year ---")
    yearly = tf.groupby(tf.entry_date.dt.year).agg(
        trades=("ret_on_risk", "size"), win=("ret_on_risk", lambda r: (r > 0).mean()),
        avg_ret=("ret_on_risk", "mean"), pnl=("pnl", "sum"))
    eq_y = pd.concat([res.equity.iloc[:1], res.equity.resample("YE").last()])
    yearly["equity_ret"] = eq_y.pct_change().iloc[1:].set_axis(eq_y.index[1:].year)
    print(yearly.round(3).to_string())

    print("\n--- By symbol ---")
    print(tf.groupby(["symbol", "period"]).ret_on_risk.agg(
        n="size", win=lambda r: (r > 0).mean(), avg="mean").unstack("period").round(3).to_string())

    print("\n--- Exit reasons ---")
    print(tf.groupby("exit_reason").ret_on_risk.agg(
        n="size", win=lambda r: (r > 0).mean(), avg="mean").round(3).to_string())
    print(f"  average holding period: {tf.days_held.mean():.1f} trading days "
          f"(median {tf.days_held.median():.0f})")

    print("\n--- By confidence tier (score at entry) ---")
    tier_tbl = tf.groupby(["tier", "period"]).ret_on_risk.agg(
        n="size", win=lambda r: (r > 0).mean(), avg="mean").unstack("period")
    print(tier_tbl.round(3).to_string())

    # ---------------- stress tests ----------------
    print("\n--- Stress tests (IS | OOS) ---")
    stress = {
        "base": dict(),
        "2x bid/ask costs": dict(cost_mult=2.0),
        "3x bid/ask costs": dict(cost_mult=3.0),
        "enter 1 day late": dict(delay=1),
        "IV 11% richer (VIX x1.0)": dict(vix_to_atm=1.0),
        "steeper skew": dict(skew=SkewModel(put_slope=0.45, call_slope=0.05)),
        "HIGH tier only": dict(min_score=cfg.high_confidence),
        "weekly trend required": replace(cfg, require_weekly_trend=True),
        "1 position at a time": replace(cfg, max_concurrent=1),
        "3 concurrent": replace(cfg, max_concurrent=3),
        "2% risk per trade": replace(cfg, risk_per_trade=0.02),
        "10% risk per trade": replace(cfg, risk_per_trade=0.10),
    }
    stress_out = {}
    for name, kw in stress.items():
        if isinstance(kw, SwingConfig):
            r = runner.run(kw)
        else:
            r = runner.run(cfg, **kw)
        a, b = r.stats(START, SPLIT), r.stats(SPLIT)
        stress_out[name] = {"IS": a, "OOS": b}
        print(f"  {name:<26} IS: {fmt(a)}")
        print(f"  {'':<26} OOS: {fmt(b)}")

    # ---------------- neighbourhood ----------------
    grid_out = []
    if not args.quick:
        print("\n--- Parameter neighbourhood (Sharpe IS / OOS) ---")
        for atr_min in (1.25, 1.5, 1.75, 2.0):
            for dte in (14, 21, 30):
                for max_hold in (5, 7, 10):
                    c = replace(cfg, min_pullback_atr=atr_min, dte=dte, max_hold=max_hold)
                    r = runner.run(c)
                    a, b = r.stats(START, SPLIT), r.stats(SPLIT)
                    grid_out.append({"min_pullback_atr": atr_min, "dte": dte, "max_hold": max_hold,
                                     "sharpe_IS": a.get("sharpe"), "sharpe_OOS": b.get("sharpe"),
                                     "avg_IS": a.get("avg_ret"), "avg_OOS": b.get("avg_ret")})
        g = pd.DataFrame(grid_out)
        print(g.pivot_table(index=["min_pullback_atr"], columns=["dte", "max_hold"],
                            values="sharpe_IS").round(2).to_string())
        print("OOS:")
        print(g.pivot_table(index=["min_pullback_atr"], columns=["dte", "max_hold"],
                            values="sharpe_OOS").round(2).to_string())
        print(f"Share of neighbourhood with positive Sharpe: IS {(g.sharpe_IS > 0).mean():.0%}, "
              f"OOS {(g.sharpe_OOS > 0).mean():.0%}")

    # bootstrap CI on mean return per trade
    rng = np.random.default_rng(0)
    boots = [rng.choice(tf.ret_on_risk.to_numpy(), len(tf)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nMean return on risk per trade: {tf.ret_on_risk.mean():+.1%} "
          f"(95% bootstrap CI {lo:+.1%} .. {hi:+.1%})")

    # Monte Carlo over the trade sequence: what does a year look like by size?
    years = (res.equity.index[-1] - res.equity.index[0]).days / 365.25
    tpy = len(tf) / years
    print(f"\n--- Monte Carlo: 1-year outcomes by size ({tpy:.0f} trades/yr, 10k resamples) ---")
    mc_out = {}
    for risk in (0.02, 0.03, 0.05, 0.075, 0.10):
        mc = monte_carlo_equity(tf.ret_on_risk.to_numpy(), tpy, risk)
        mc_out[f"{risk:.3f}"] = mc
        print(f"  risk {risk:>5.1%}/trade: median {mc['median_return']:+.1%}  "
              f"5th pct {mc['p05_return']:+.1%}  P(losing year) {mc['prob_loss']:.0%}  "
              f"median maxDD {mc['median_max_dd']:+.1%}  bad-case maxDD {mc['p95_max_dd']:+.1%}")
    print("  (single-position compounding; concurrent positions add correlation risk)")

    tf.to_csv(OUT / "options_swing_trades.csv", index=False)
    res.equity.to_csv(OUT / "options_swing_equity.csv")
    with open(OUT / "options_swing_backtest.json", "w") as fh:
        json.dump({
            "config": cfg.__dict__ | {"symbols": list(cfg.symbols)},
            "full": full, "in_sample": is_, "out_of_sample": oos,
            "spy_buy_hold": spy_st, "stress": stress_out, "neighbourhood": grid_out,
            "mean_ret_ci95": [lo, hi], "monte_carlo_1y": mc_out,
        }, fh, indent=2, default=float)
    print(f"\nSaved results to {OUT}/options_swing_*")


if __name__ == "__main__":
    main()
