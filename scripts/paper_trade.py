"""Paper-trade portfolio mode with the same engine as the backtest.

Run once a day after the close (or any time: missed days are replayed):

    python scripts/paper_trade.py update            # process new trading days
    python scripts/paper_trade.py report            # equity, positions, fill gap
    python scripts/paper_trade.py fill SPY-2026-10-14 31.85
                                                    # record a real option price
    python scripts/paper_trade.py update --start 2025-01-02 --ledger data/paper/demo.json
                                                    # start a ledger in the past (replay)

The ledger lives in data/paper/ledger.json by default (not committed).
"""

import sys

sys.path.insert(0, ".")

import argparse

from backtesting.market_data import load_universe
from trading.options_swing import SwingConfig
from trading.paper_ledger import DEFAULT_LEDGER, PaperLedger


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["update", "report", "fill"], nargs="?", default="update")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    ap.add_argument("--start", help="first day for a NEW ledger (default: latest bar)")
    ap.add_argument("--initial", type=float, default=100_000.0)
    ap.add_argument("--fractional", action="store_true",
                    help="fractional option contracts (research sizing) for a NEW ledger")
    opts = ap.parse_args()

    cfg = SwingConfig.from_yaml()
    ledger = PaperLedger(opts.ledger, cfg, opts.initial, whole_contracts=not opts.fractional)
    if opts.command == "fill":
        if len(opts.args) != 2:
            ap.error("usage: fill <position-id> <actual price per share>")
        ledger.record_fill(opts.args[0], float(opts.args[1]))
        print(f"Recorded fill for {opts.args[0]}")
    elif opts.command == "update":
        syms = sorted(set(cfg.symbols) | {cfg.core_symbol, "SPY", "VIX"})
        data = load_universe(syms, refresh=True)
        for line in ledger.update(data, start=opts.start):
            print(line)
    print(ledger.report())


if __name__ == "__main__":
    main()
