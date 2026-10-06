"""Adversarial bias checks for the Options Swing Strategy.

Each test targets a specific way a backtest can mislead:

1. Untouched history  - run the frozen rules on the S&P 500 index 1991-2009
   (dot-com bust, 2008 crash), data never used while designing them.
2. Placebo / random entry - price a trade on *every* eligible day, then
   compare the real signal's trades with thousands of random-day samples of
   the same size. If random days in an uptrend do as well, the "edge" is
   just market drift plus the option model, not the timing rule.
3. No option model - the same entries/exits on the underlying alone.
4. Harsher pricing - steeper call skew, extra implied vol paid at entry,
   wider spreads, late entries and exits, and all of them at once.
5. Deflated Sharpe ratio - corrects the Sharpe ratio for how many
   configurations were tried during research.

Usage:
    python scripts/research_bias_checks.py
"""

import sys

sys.path.insert(0, ".")

import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtesting.market_data import load_universe
from backtesting.options_backtest import (
    OptionsSwingBacktester,
    PricingScenario,
    non_overlapping_entries,
    simulate_trades,
)
from backtesting.options_pricing import SkewModel, atm_iv_series
from trading.features import build_features
from trading.options_swing import SwingConfig, confidence_score, entry_signal, exit_signal

SPX_URL = "https://raw.githubusercontent.com/vijinho/sp500/master/csv/sp500.csv"
CACHE = Path("data/cache/daily/SPX_1950_2018.csv")
OUT = Path("data/results")
RNG = np.random.default_rng(42)


Scenario = PricingScenario  # backwards-compatible name


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------

def load_spx() -> pd.DataFrame:
    """S&P 500 index daily OHLC, scaled /10 so prices sit on SPY's scale
    (strike spacing and $ minimum ticks then behave like SPY options)."""
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SPX_URL, timeout=60) as resp:  # noqa: S310
            CACHE.write_bytes(resp.read())
    raw = pd.read_csv(CACHE, parse_dates=["Date"]).set_index("Date").sort_index()
    df = raw[["Open", "High", "Low", "Close"]].astype(float) / 10.0
    df["AdjClose"] = df["Close"]
    df["Volume"] = raw["Volume"].astype(float)
    return df


# ----------------------------------------------------------------------
# per-trade simulator
# ----------------------------------------------------------------------

def simulate_all_days(f: pd.DataFrame, iv: pd.Series, cfg: SwingConfig, sc: Scenario,
                      start: str, end: str) -> pd.DataFrame:
    """Simulate the strategy's option trade as if entered on *every* day."""
    return simulate_trades(f["close"], iv, exit_signal(f), cfg.structure(), cfg.exits(), sc,
                           start=start, end=end)


def strategy_days(sig: pd.Series, held: pd.Series) -> pd.DatetimeIndex:
    """Entry days the single-symbol strategy would actually take (no overlap)."""
    return non_overlapping_entries(sig, held)


def stats(r: pd.Series) -> str:
    r = r.dropna()
    if len(r) < 2:
        return "n/a"
    t = r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))
    return f"n={len(r):>4}  win={(r > 0).mean():.0%}  avg={r.mean():+.2%}  t={t:+.2f}"


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------

def placebo(trades: dict, pools: dict, n_draws: int = 5000) -> tuple:
    """Distribution of the mean return of random-day samples matching
    the real strategy's trade count per symbol."""
    actual = np.concatenate([trades[s] for s in trades])
    sims = np.empty(n_draws)
    for k in range(n_draws):
        parts = [RNG.choice(pools[s], size=len(trades[s]), replace=True) for s in trades]
        sims[k] = np.concatenate(parts).mean()
    return actual.mean(), sims


def deflated_sharpe(daily: pd.Series, trial_sharpes: np.ndarray, n_trials: int) -> dict:
    """Bailey & Lopez de Prado deflated Sharpe ratio (daily returns)."""
    r = daily.dropna()
    t = len(r)
    sr = r.mean() / r.std(ddof=1)
    skew = float(((r - r.mean()) ** 3).mean() / r.std() ** 3)
    kurt = float(((r - r.mean()) ** 4).mean() / r.std() ** 4)
    var_trials = np.var(trial_sharpes / math.sqrt(252), ddof=1)
    gamma = 0.5772156649
    sr0 = math.sqrt(var_trials) * ((1 - gamma) * norm.ppf(1 - 1 / n_trials)
                                   + gamma * norm.ppf(1 - 1 / (n_trials * math.e)))
    z = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return {"sharpe_ann": sr * math.sqrt(252), "sr0_ann": sr0 * math.sqrt(252),
            "n_trials": n_trials, "dsr": float(norm.cdf(z)),
            "psr_vs_zero": float(norm.cdf(sr * math.sqrt(t - 1)
                                          / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)))}


def main() -> None:
    pd.set_option("display.width", 200)
    cfg = SwingConfig.from_yaml()
    data = load_universe(list(cfg.symbols) + ["SPY", "VIX"])
    vix = data["VIX"]

    universes = {
        "2011-2026 (design data)": ({s: data[s] for s in cfg.symbols}, "2011-01-01", "2026-12-31"),
        "1991-2009 (UNTOUCHED)": ({"SPX": load_spx()}, "1991-01-01", "2009-12-31"),
    }
    harsh = Scenario(skew=SkewModel(call_slope=0.25), entry_iv_mult=1.10, cost_mult=1.5,
                     entry_delay=1, exit_delay=1)
    scenarios = {
        "base": Scenario(),
        "call skew 0.05 (flatter)": Scenario(skew=SkewModel(call_slope=0.05)),
        "call skew 0.15": Scenario(skew=SkewModel(call_slope=0.15)),
        "call skew 0.20": Scenario(skew=SkewModel(call_slope=0.20)),
        "call skew 0.25 (steep)": Scenario(skew=SkewModel(call_slope=0.25)),
        "pay +10% IV at entry": Scenario(entry_iv_mult=1.10),
        "1.5x bid/ask": Scenario(cost_mult=1.5),
        "enter AND exit 1 day late": Scenario(entry_delay=1, exit_delay=1),
        "ALL HARSH AT ONCE": harsh,
    }

    for uname, (dfs, start, end) in universes.items():
        print("\n" + "=" * 90)
        print(f"{uname}: {', '.join(dfs)}  {start[:4]}-{end[:4]}")
        print("=" * 90)
        feats = {s: build_features(df, vix) for s, df in dfs.items()}
        ref = data["SPY"] if "SPY" in dfs or "QQQ" in dfs else dfs["SPX"]
        ivs = {s: atm_iv_series(dfs[s], vix, ref) for s in dfs}
        sigs = {s: entry_signal(feats[s], cfg) for s in dfs}

        base_all = {}
        for sname, sc in scenarios.items():
            per = {s: simulate_all_days(feats[s], ivs[s], cfg, sc, start, end) for s in dfs}
            if sname == "base":
                base_all = per
            taken = {s: per[s].loc[strategy_days(sigs[s].reindex(per[s].index, fill_value=False),
                                                  per[s].held)] for s in dfs}
            allt = pd.concat(taken.values())
            print(f"  {sname:<27} options: {stats(allt.ret)}")

        # ---- placebo on base pricing ----
        taken = {s: base_all[s].loc[strategy_days(sigs[s].reindex(base_all[s].index,
                                                                   fill_value=False),
                                                  base_all[s].held)] for s in dfs}
        allt = pd.concat(taken.values()).sort_index()
        print(f"\n  Underlying only (no options), same days: {stats(allt.und_ret)}")
        for pool_name, pool_fn in {
            "random days, any trend": lambda f: pd.Series(True, index=f.index),
            "random days ABOVE 200-day SMA": lambda f: f.above200,
        }.items():
            pools_opt = {s: base_all[s].ret[pool_fn(feats[s]).reindex(base_all[s].index,
                                                                     fill_value=False)].to_numpy()
                         for s in dfs}
            pools_und = {s: base_all[s].und_ret[pool_fn(feats[s]).reindex(base_all[s].index,
                                                                         fill_value=False)].to_numpy()
                         for s in dfs}
            act, sims = placebo({s: taken[s].ret.to_numpy() for s in dfs}, pools_opt)
            act_u, sims_u = placebo({s: taken[s].und_ret.to_numpy() for s in dfs}, pools_und)
            print(f"  Placebo vs {pool_name:<31}: options avg {act:+.2%} vs random "
                  f"{np.median(sims):+.2%} (p={np.mean(sims >= act):.4f}) | underlying "
                  f"{act_u:+.2%} vs {np.median(sims_u):+.2%} (p={np.mean(sims_u >= act_u):.4f})")

        # ---- sub-periods ----
        print("\n  By market phase (base pricing, strategy trades):")
        phases = ([("1991-1999 bull", "1991", "1999"), ("2000-2002 bust", "2000", "2002"),
                   ("2003-2007 bull", "2003", "2007"), ("2008-2009 crisis", "2008", "2009")]
                  if "SPX" in dfs else
                  [("2011-2015", "2011", "2015"), ("2016-2019", "2016", "2019"),
                   ("2020-2022", "2020", "2022"), ("2023-2026", "2023", "2026")])
        for name, a, b in phases:
            sub = allt.loc[a:b]
            print(f"    {name:<18} {stats(sub.ret)}   underlying avg {sub.und_ret.mean():+.2%}")

        # ---- portfolio backtest on the untouched data ----
        if "SPX" in dfs:
            bt = OptionsSwingBacktester(cfg.structure(), cfg.exits(),
                                        risk_per_trade=cfg.risk_per_trade,
                                        max_concurrent=cfg.max_concurrent, etfs=("SPX",))
            res = bt.run(feats, ivs, sigs, {s: exit_signal(feats[s]) for s in dfs}, 1,
                         {s: confidence_score(feats[s]) for s in dfs}, start=start, end=end)
            st = res.stats()
            spx = dfs["SPX"].Close.loc[start:end]
            yrs = (spx.index[-1] - spx.index[0]).days / 365.25
            print(f"\n  Portfolio (5% risk): CAGR {st['cagr']:+.1%}  maxDD {st['max_drawdown']:+.1%}  "
                  f"Sharpe {st['sharpe']:.2f}  | S&P 500 price CAGR "
                  f"{(spx.iloc[-1] / spx.iloc[0]) ** (1 / yrs) - 1:+.1%}")
            yearly = res.equity.resample("YE").last().pct_change()
            yearly.iloc[0] = res.equity.resample("YE").last().iloc[0] / res.equity.iloc[0] - 1
            print("  Yearly: " + "  ".join(f"{d.year % 100:02d}:{v:+.0%}" for d, v in yearly.items()))

    # ---- deflated Sharpe on the design-period equity curve ----
    eq_path = OUT / "options_swing_equity.csv"
    trial_files = [OUT / "options_structures_etf.csv", OUT / "filters.csv"]
    if eq_path.exists() and all(p.exists() for p in trial_files):
        eq = pd.read_csv(eq_path, index_col=0, parse_dates=True).iloc[:, 0]
        trials = pd.concat([pd.read_csv(p) for p in trial_files])
        tsh = trials["sharpe_IS"].dropna().to_numpy()
        print("\n" + "=" * 90)
        print("Deflated Sharpe ratio (2011-2026 equity, daily returns)")
        for n in (30, 100, len(tsh) + 200):
            d = deflated_sharpe(eq.pct_change(), tsh, n)
            print(f"  assuming {n:>4} independent trials: Sharpe {d['sharpe_ann']:.2f} vs "
                  f"best-of-noise {d['sr0_ann']:.2f} -> P(real edge) = {d['dsr']:.1%}")
        print(f"  probability Sharpe > 0 ignoring multiple testing: {d['psr_vs_zero']:.1%}")


if __name__ == "__main__":
    main()
