"""Re-price the strategy's actual option trades on REAL historical quotes.

Every options result so far used modelled prices (Black-Scholes, IV from
VIX). This takes the trades portfolio mode actually made between 2008 and
2025 and prices each one with real end-of-day bids and asks for SPY, QQQ
and IWM (lambdaclass/options_backtester `data-v1` release, SHA-256 verified;
download with the commands in docs/OPTIONS_SWING_STRATEGY.md).

Protocol (fixed before looking at any real prices)
--------------------------------------------------
Contract: on the entry day, first expiry at least 30 days out, the call
whose quoted delta is closest to 0.80 among valid quotes (bid > 0,
ask > bid, spread < 50% of mid). Exit: same contract on the model's exit
day (next quote within 3 trading days if missing). Fills:

    pessimistic  buy at the ask, sell at the bid
    realistic    limit order 25% into the spread on both sides
    optimistic   mid on both sides

plus $0.65 per contract per side. Verdict: the options leg is CONFIRMED
only if realistic-fill returns are positive with t > 2 AND portfolio mode
with real-quote option P&L beats the core alone over 2008-2025.

Usage:
    python scripts/research_real_quotes.py
"""

import sys

sys.path.insert(0, ".")

import math
from pathlib import Path

import pandas as pd

from backtesting.market_data import load_long_history, load_universe, split_factor
from backtesting.options_backtest import summarize
from backtesting.options_pricing import SkewModel, atm_iv_series, bs_price, risk_free_rate
from backtesting.portfolio import faber_signal, simulate_core_overlay
from trading.features import build_features
from trading.options_swing import SwingConfig, confidence_score, entry_signal, exit_signal

CORE = ["SPY", "QQQ", "IWM"]
QUOTES = Path("data/cache/options")
START, END = "2008-01-02", "2025-12-13"
COMMISSION = 0.65 / 100.0  # $ per share per side
TARGET_DELTA = 0.80


def load_calls(sym: str, dates) -> pd.DataFrame:
    df = pd.read_parquet(
        QUOTES / f"{sym}_options.parquet",
        columns=["date", "expiration", "strike", "type", "bid", "ask", "delta",
                 "implied_volatility"],
        filters=[("type", "==", "call"), ("date", "in", list(pd.DatetimeIndex(dates)))])
    df["type"] = df["type"].astype(str)
    return df[(df.bid > 0) & (df.ask > df.bid)].assign(
        mid=lambda d: (d.bid + d.ask) / 2).query("(ask - bid) / mid < 0.5")


def fills(bid, ask, side):
    mid, spr = (bid + ask) / 2, ask - bid
    if side == "buy":
        return {"pessimistic": ask, "realistic": mid + 0.25 * spr, "optimistic": mid}
    return {"pessimistic": bid, "realistic": mid - 0.25 * spr, "optimistic": mid}


def main() -> None:
    cfg = SwingConfig.from_yaml()
    vix = load_universe(["VIX"])["VIX"]
    data = {s: load_long_history(s) for s in CORE}
    data = {s: d[d.index < END] for s, d in data.items()}
    feats = {s: build_features(data[s], vix) for s in CORE}
    ivs = {s: atm_iv_series(data[s], vix, data["SPY"]) for s in CORE}
    ent = {s: entry_signal(feats[s], cfg) for s in CORE}
    ext = {s: exit_signal(feats[s]) for s in CORE}
    prio = {s: confidence_score(feats[s]) + feats[s].pullback_atr for s in CORE}
    scale = {s: split_factor(data[s].index, s) for s in CORE}
    spy = data["SPY"]
    fs = faber_signal(spy["Close"])

    eq_model, trades = simulate_core_overlay(
        spy["AdjClose"], fs, feats, ivs, ent, ext, cfg.structure(), cfg.exits(),
        cfg.core_weight, cfg.risk_per_trade, cfg.max_concurrent, prio, scale, start=START)
    eq_core, _ = simulate_core_overlay(
        spy["AdjClose"], fs, feats, ivs, {s: ent[s] & False for s in CORE}, ext,
        cfg.structure(), cfg.exits(), cfg.core_weight, cfg.risk_per_trade,
        cfg.max_concurrent, prio, scale, start=START)
    trades["entry_date"] = pd.to_datetime(trades.entry_date)
    trades["exit_date"] = pd.to_datetime(trades.exit_date)
    print(f"Model trades 2008-2025: {len(trades)}")

    skew = SkewModel()
    rows = []
    for sym in CORE:
        tr = trades[trades.symbol == sym]
        if tr.empty:
            continue
        cal = feats[sym].index
        need = set(tr.entry_date)
        for d in tr.exit_date:
            k = cal.searchsorted(d)
            need.update(cal[k:k + 4])
        q = load_calls(sym, sorted(need))
        by_day = dict(tuple(q.groupby("date")))
        for t in tr.itertuples():
            row = {"symbol": sym, "entry_date": t.entry_date, "exit_date": t.exit_date,
                   "model_ret": t.ret_on_risk, "risk": t.risk, "status": "ok"}
            chain = by_day.get(t.entry_date)
            if chain is None:
                rows.append({**row, "status": "no entry quotes"})
                continue
            exps = sorted(e for e in chain.expiration.unique()
                          if (e - t.entry_date).days >= 30)
            if not exps:
                rows.append({**row, "status": "no 30d expiry"})
                continue
            c = chain[(chain.expiration == exps[0]) & chain.delta.notna()]
            if c.empty:
                rows.append({**row, "status": "no deltas"})
                continue
            pick = c.iloc[(c.delta - TARGET_DELTA).abs().argmin()]
            # exit quote: same contract on the exit day, or the next 3 trading days
            k = cal.searchsorted(t.exit_date)
            exit_q = None
            for d in cal[k:k + 4]:
                day = by_day.get(d)
                if day is None:
                    continue
                m = day[(day.expiration == pick.expiration) & (day.strike == pick.strike)]
                if not m.empty:
                    exit_q = m.iloc[0]
                    break
            if exit_q is None:
                rows.append({**row, "status": "no exit quote"})
                continue
            buy, sell = fills(pick.bid, pick.ask, "buy"), fills(exit_q.bid, exit_q.ask, "sell")
            for k_ in buy:
                cost = buy[k_] + COMMISSION
                row[f"ret_{k_}"] = (sell[k_] - COMMISSION - cost) / cost
            # how well did the model price this exact real contract at entry?
            spot = float(feats[sym]["close"].loc[t.entry_date])
            tau = (pick.expiration - t.entry_date).days / 365.0
            atm = float(ivs[sym].loc[t.entry_date])
            model_px = bs_price(spot, pick.strike, tau, skew.iv(atm, spot, pick.strike, tau),
                                risk_free_rate(t.entry_date), True)
            row.update(delta=float(pick.delta), dte=(pick.expiration - t.entry_date).days,
                       real_mid=float((pick.bid + pick.ask) / 2), real_ask=float(pick.ask),
                       spread_pct=float((pick.ask - pick.bid) / ((pick.ask + pick.bid) / 2)),
                       model_px=model_px)
            rows.append(row)

    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"].copy()
    print(f"Priced on real quotes: {len(ok)} of {len(df)} "
          f"({df.status.value_counts().to_dict()})")
    print(f"Chosen contracts: median delta {ok.delta.median():.2f}, median DTE {ok.dte.median():.0f}, "
          f"median bid/ask spread {ok.spread_pct.median():.2%} of mid")
    gap_mid = ok.real_mid / ok.model_px - 1
    gap_ask = ok.real_ask / ok.model_px - 1
    print(f"Model price vs real, same contract at entry: real mid is {gap_mid.median():+.2%} "
          f"(median) / {gap_mid.mean():+.2%} (mean) vs model; real ask {gap_ask.median():+.2%} "
          f"(median)")

    def line(name, r):
        r = r.dropna()
        t = r.mean() / r.std(ddof=1) * math.sqrt(len(r))
        print(f"  {name:<34} n={len(r):>4} win={(r > 0).mean():.0%} avg={r.mean():+.2%} "
              f"median={r.median():+.2%} t={t:+.2f}")
        return t

    print("\nPer-trade return on premium, same trades:")
    line("model (Black-Scholes, VIX IV)", ok.model_ret)
    t_real = {}
    for k_ in ("optimistic", "realistic", "pessimistic"):
        t_real[k_] = line(f"REAL quotes, {k_} fills", ok[f"ret_{k_}"])
    print(f"  correlation model vs real (realistic): "
          f"{ok.model_ret.corr(ok.ret_realistic):.2f}")
    ok["year"] = ok.entry_date.dt.year
    print("\nBy period (realistic fills vs model):")
    for lab, (a, b) in {"2008-2010": (2008, 2010), "2011-2019": (2011, 2019),
                        "2020-2025": (2020, 2025)}.items():
        s = ok[(ok.year >= a) & (ok.year <= b)]
        print(f"  {lab}: n={len(s):>3}  real {s.ret_realistic.mean():+.2%} "
              f"(win {(s.ret_realistic > 0).mean():.0%})  model {s.model_ret.mean():+.2%}")

    # Portfolio impact: replace each trade's model P&L with the real-quote P&L.
    # (Approximation: later position sizes are not re-derived from the changed equity.)
    print("\nPortfolio mode 2008-2025 (85% SPY trend core + options), option P&L swapped:")
    rows_out = {}
    base = summarize(pd.DataFrame(), eq_core)
    rows_out["core only, no options"] = (base, eq_core)
    rows_out["model option prices"] = (summarize(pd.DataFrame(), eq_model), eq_model)
    for k_ in ("optimistic", "realistic", "pessimistic"):
        r = ok[f"ret_{k_}"]
        diff = (ok.risk * (r - ok.model_ret)).groupby(ok.exit_date).sum()
        # trades that could not be priced keep zero option P&L (as if skipped)
        miss = df[df.status != "ok"]
        diff_miss = (-miss.risk * miss.model_ret).groupby(miss.exit_date).sum()
        adj = pd.concat([diff, diff_miss]).groupby(level=0).sum()
        adj = adj.reindex(eq_model.index, fill_value=0.0).cumsum()
        eq = eq_model + adj
        rows_out[f"REAL quotes, {k_} fills"] = (summarize(pd.DataFrame(), eq), eq)
    spy_bh = spy["AdjClose"].loc[START:END]
    rows_out["SPY buy & hold"] = (summarize(pd.DataFrame(), spy_bh), spy_bh)
    for name, (st, eq) in rows_out.items():
        yr = pd.concat([eq.iloc[:1], eq.resample("YE").last()]).pct_change().iloc[1:]
        print(f"  {name:<30} CAGR={st['cagr']:+6.1%}  maxDD={st['max_drawdown']:+6.1%}  "
              f"Sharpe(ex)={st['sharpe_excess']:5.2f}  worst year={yr.min():+6.1%}")

    real_cagr = rows_out["REAL quotes, realistic fills"][0]["cagr"]
    confirmed = t_real["realistic"] > 2 and real_cagr > base["cagr"]
    print("\nPre-registered verdict:",
          "CONFIRMED - the options leg makes money on real quotes with realistic fills"
          if confirmed else "NOT CONFIRMED on real quotes")
    df.to_csv("data/results/real_quote_trades.csv", index=False)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
