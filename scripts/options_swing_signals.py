"""Today's Options Swing signals (call debit spreads on SPY/QQQ/IWM dips).

Run this ~15 minutes before the close. For each ETF it prints whether the
entry setup is live, the confidence score, and a concrete spread ticket
(strikes, expiry, estimated debit, take-profit value) plus the exit plan.

Usage:
    python scripts/options_swing_signals.py [--json]
"""

import sys

sys.path.insert(0, ".")

import argparse
import json

import pandas as pd

from backtesting.market_data import load_universe
from backtesting.options_pricing import atm_iv_series
from trading.features import build_features
from trading.options_swing import SwingConfig, scan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print machine-readable output")
    args = ap.parse_args()

    cfg = SwingConfig.from_yaml()
    syms = list(cfg.symbols)
    data = load_universe(syms + ["SPY", "VIX"], refresh=True)
    feats = {s: build_features(data[s], data["VIX"]) for s in syms}
    ivs = {s: atm_iv_series(data[s], data["VIX"], data["SPY"]) for s in syms}
    result = scan(feats, ivs, cfg)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    last = max(pd.Timestamp(v["date"]) for v in result.values())
    vix_last = data["VIX"].index[-1]
    print("=" * 72)
    print("OPTIONS SWING SIGNALS  -  buy the dip in an uptrend (call debit spreads)")
    print("=" * 72)
    print(f"Data through {last.date()}  |  VIX data through {vix_last.date()}"
          + ("  (stale: IV estimated from realized vol)" if (last - vix_last).days > 3 else ""))

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
            print(f"     SELL {sym} {t['expiry']} {t['short_strike']:g} CALL")
            print(f"     est. debit ${t['est_debit']:.2f}/sh (${t['est_debit'] * 100:.0f}/spread), "
                  f"max profit ${t['max_profit']:.2f}/sh, model IV {t['atm_iv']:.1%}")
            print(f"     take profit when spread is worth ${t['take_profit_value']:.2f} "
                  f"(+{cfg.profit_target:.0%} of max profit)")
            print(f"     why: {', '.join(t['reasons'])}")
        print(f"     if already holding: exit signal (close > 5-day SMA) = "
              f"{'YES' if st['exit_signal_if_held'] else 'no'}")

    print("\nExit plan for every position: take profit at "
          f"{cfg.profit_target:.0%} of max profit, otherwise exit at the first close above "
          f"the 5-day SMA once held >= {cfg.min_hold} days, and always by day {cfg.max_hold}.")
    print(f"Size: risk {cfg.risk_per_trade:.0%} of the account per spread (the debit is the "
          f"max loss), at most {cfg.max_concurrent} open at once.")
    print("Prices are model estimates - check the live bid/ask and pay no more than "
          "~5% above the estimated debit.")


if __name__ == "__main__":
    main()
