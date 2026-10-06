"""Today's Options Swing signals (ITM calls on SPY/QQQ/IWM dips).

Run this ~15 minutes before the close. For each ETF it prints whether the
entry setup is live, the confidence score, and a concrete order ticket
(strike(s), expiry, estimated debit, take-profit value) plus the exit plan.

Usage:
    python scripts/options_swing_signals.py [--json]
"""

import sys

sys.path.insert(0, ".")

import argparse
import json

import numpy as np
import pandas as pd

from backtesting.market_data import load_universe
from backtesting.options_pricing import atm_iv_series
from trading.features import build_features
from trading.options_swing import SwingConfig, contracts_for, core_status, scan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print machine-readable output")
    ap.add_argument("--vix", type=float, default=None,
                    help="today's VIX close, if the data source is stale (e.g. 16.4)")
    ap.add_argument("--account", type=float, default=None,
                    help="account size in $: prints whole contracts / shares to trade")
    args = ap.parse_args()

    cfg = SwingConfig.from_yaml()
    syms = list(cfg.symbols)
    # allow_partial: run ~15 min before the close and use the live bar
    data = load_universe(syms + ["SPY", "VIX"], refresh=True, allow_partial=True)
    last_px = max(data[s].index[-1] for s in syms)
    if args.vix is not None:  # manual override for today's VIX
        vix = data["VIX"]
        row = pd.DataFrame({c: [args.vix] for c in ("Open", "High", "Low", "Close", "AdjClose")},
                           index=[last_px]).assign(Volume=0.0)
        data["VIX"] = pd.concat([vix[vix.index < last_px], row]).sort_index()
    feats = {s: build_features(data[s], data["VIX"]) for s in syms}
    ivs = {s: atm_iv_series(data[s], data["VIX"], data["SPY"]) for s in syms}
    result = scan(feats, ivs, cfg)

    core = None
    if cfg.portfolio_mode:
        core_df = (data[cfg.core_symbol] if cfg.core_symbol in data
                   else load_universe([cfg.core_symbol], refresh=True)[cfg.core_symbol])
        core = core_status(core_df["Close"], cfg)
        result = {"core": core, **result}

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return
    result.pop("core", None)

    last = max(pd.Timestamp(v["date"]) for v in result.values())
    vix_last = data["VIX"].index[-1]
    print("=" * 72)
    print(f"OPTIONS SWING SIGNALS  -  buy the dip in an uptrend ({cfg.structure_kind})")
    print("=" * 72)
    print(f"Data through {last.date()}  |  VIX data through {vix_last.date()}"
          + ("  (stale: IV estimated from realized vol; pass --vix <today's VIX>)"
             if np.busday_count(vix_last.date(), last.date()) > 1 else ""))
    age = (pd.Timestamp.now().normalize() - last).days
    if age > 4:
        print(f"WARNING: price data is {age} days old - signals may be out of date. "
              "Check your connection or run where Yahoo Finance is reachable.")

    if core:
        state = (f"IN - hold {core['core_weight']:.0%} of the account in {core['symbol']}"
                 if core["in_market"] else
                 f"OUT - keep the core {core['core_weight']:.0%} in T-bills / money market")
        print(f"\nPORTFOLIO CORE: {state}")
        print(f"     {core['symbol']} {core['last_month_end']} close ${core['last_month_close']:.2f} "
              f"vs {cfg.faber_months}-month SMA ${core['sma']:.2f}")
        if opts_account := args.account:
            px = float(core_df["Close"].iloc[-1])
            shares = int(core["core_weight"] * opts_account // px) if core["in_market"] else 0
            print(f"     for a ${opts_account:,.0f} account: core = {shares} {core['symbol']} shares")
        print(f"     month-end re-check (using today's close): would be "
              f"{'IN' if core['month_end_preview_in'] else 'OUT'} "
              f"(SMA ${core['preview_sma']:.2f}); act on it at the last close of the month")

    for sym, st in result.items():
        trend = "UP" if st["above200"] else "DOWN (no longs)"
        print(f"\n{sym:<4} ${st['close']:.2f}  trend {trend}  RSI2 {st['rsi2']:.1f}  "
              f"IBS {st['ibs']:.2f}  pullback {st['pullback_atr']:.2f} ATR  "
              f"weekly {'up' if st['weekly_up'] else 'down'}")
        if not st["signal"]:
            print("     no entry today")
        else:
            t = st["ticket"]
            print(f"  >> ENTRY  confidence {t['confidence']}/100 ({t['tier']})")
            print(f"     BUY  {sym} {t['expiry']} {t['long_strike']:g} CALL  (~{t['long_delta']:.2f} delta)")
            if t["short_strike"] is not None:
                print(f"     SELL {sym} {t['expiry']} {t['short_strike']:g} CALL")
            unit = "spread" if t["short_strike"] is not None else "contract"
            cap = (f"max profit ${t['max_profit']:.2f}/sh" if t["max_profit"] is not None
                   else "max loss = debit")
            print(f"     est. debit ${t['est_debit']:.2f}/sh (${t['est_debit'] * 100:.0f}/{unit}), "
                  f"{cap}, model IV {t['atm_iv']:.1%}")
            target = "of max profit" if t["max_profit"] is not None else "gain"
            print(f"     take profit when the position is worth ${t['take_profit_value']:.2f} "
                  f"(+{cfg.profit_target:.0%} {target})")
            if args.account:
                n = contracts_for(t["est_debit"], args.account, cfg)
                cost = n * t["est_debit"] * 100
                if n:
                    print(f"     for a ${args.account:,.0f} account: buy {n} contract(s) "
                          f"(~${cost:,.0f}, {cost / args.account:.1%} of the account)")
                else:
                    print(f"     for a ${args.account:,.0f} account: SKIP the option - one contract "
                          f"(${t['est_debit'] * 100:,.0f}) is over 1.5x your "
                          f"{cfg.risk_per_trade:.0%} target; use the shares line instead")
            print(f"     why: {', '.join(t['reasons'])}")
            print(f"     shares alternative (most reliable in testing): buy "
                  f"{t['shares_position_frac']:.0%} of the account in {sym} at the close, "
                  f"same exits")
        print(f"     if already holding: exit signal (close > 5-day SMA) = "
              f"{'YES' if st['exit_signal_if_held'] else 'no'}")

    print("\nExit plan for every position: take profit at "
          f"+{cfg.profit_target:.0%}, otherwise exit at the first close above "
          f"the 5-day SMA once held >= {cfg.min_hold} days, and always by day {cfg.max_hold}.")
    print(f"Size: risk {cfg.risk_per_trade:.0%} of the TOTAL account per position (the debit is "
          f"the max loss), at most {cfg.max_concurrent} open at once"
          + (", paid from the cash not held in the core." if cfg.portfolio_mode else "."))
    print("Prices are model estimates. SKIP the trade if you cannot fill within ~2% of the "
          "estimated debit:\n  the bias checks showed that overpaying ~1.5 vol points "
          "(~4-5% of the premium) erases most of the edge.")


if __name__ == "__main__":
    main()
