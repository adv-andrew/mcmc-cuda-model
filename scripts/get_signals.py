"""Get current trading signals for all watchlist stocks.

Usage:
    python scripts/get_signals.py           # MCMC only (fast, free)
    python scripts/get_signals.py --llm     # MCMC + LLM validation
"""

import sys
sys.path.insert(0, ".")

import argparse
import yaml
from backtesting.data_loader import DataLoader
from trading.indicator import MCMCIndicator


def main():
    parser = argparse.ArgumentParser(description="Get trading signals")
    parser.add_argument("--llm", action="store_true", help="Add LLM validation")
    parser.add_argument("--ticker", type=str, help="Single ticker (default: all watchlist)")
    args = parser.parse_args()

    # Load config
    with open("config/tickers.yaml") as f:
        tickers = yaml.safe_load(f).get("watchlist", [])

    if args.ticker:
        tickers = [args.ticker.upper()]

    loader = DataLoader(cache_dir="data/cache")
    indicator = MCMCIndicator(n_simulations=25000, n_regimes=3, slope_threshold=15.0)

    validator = None
    if args.llm:
        from integrations.openai_validator import OpenAIValidator
        validator = OpenAIValidator(model="gpt-4o-mini")

    print()
    print("=" * 70)
    print("TRADING SIGNALS" + (" + LLM" if args.llm else ""))
    print("=" * 70)

    buys = []
    sells = []
    holds = []

    for ticker in tickers:
        # Get 1h and 1d data for MTF
        df_1d = loader.fetch(ticker, timeframe="1d", days=90, use_cache=False)
        df_1h = loader.fetch(ticker, timeframe="1h", days=7, use_cache=False)

        if df_1d.empty or len(df_1d) < 10:
            print(f"{ticker:5} | NO DATA")
            continue

        # Generate signals
        sig_1d = indicator.generate_signal(ticker, df_1d, "1d")

        if not df_1h.empty and len(df_1h) >= 5:
            sig_1h = indicator.generate_signal(ticker, df_1h, "1h")
            mtf_aligned = sig_1h["direction"] == sig_1d["direction"] and sig_1d["direction"] != "NEUTRAL"
            mtf_str = f"1h:{sig_1h['direction'][:4]} 1d:{sig_1d['direction'][:4]}"
        else:
            mtf_aligned = True
            mtf_str = "1d only"

        action = sig_1d["suggested_action"]
        strength = sig_1d["signal_strength"]
        price = sig_1d["current_price"]
        slope = sig_1d["slope_degrees"]

        # LLM validation if enabled
        llm_note = ""
        if validator and action != "HOLD":
            llm_result = validator.validate_signal(ticker, sig_1d, df_1d)
            if not llm_result["agreed_with_mcmc"]:
                llm_note = f" [LLM: {llm_result['action']}]"
                action = llm_result["action"]  # Use LLM override

        # MTF filter
        if not mtf_aligned and action != "HOLD":
            action = "HOLD"
            mtf_note = " (MTF conflict)"
        else:
            mtf_note = ""

        # Format output
        if action == "BUY":
            marker = ">>>"
            stop = price * 0.95
            target = price * 1.12
            buys.append((ticker, price, stop, target, strength))
        elif action == "SELL":
            marker = "<<<"
            stop = price * 1.05
            target = price * 0.88
            sells.append((ticker, price, stop, target, strength))
        else:
            marker = "   "
            holds.append(ticker)

        print(f"{marker} {ticker:5} | {action:4} | ${price:>8.2f} | Str:{strength:.2f} | Slope:{slope:>5.1f}° | {mtf_str}{mtf_note}{llm_note}")

    print("=" * 70)

    # Summary
    if buys:
        print("\nBUY SIGNALS:")
        for ticker, price, stop, target, strength in buys:
            print(f"  {ticker}: Entry ${price:.2f} | Stop ${stop:.2f} (-5%) | Target ${target:.2f} (+12%)")

    if sells:
        print("\nSELL/SHORT SIGNALS:")
        for ticker, price, stop, target, strength in sells:
            print(f"  {ticker}: Entry ${price:.2f} | Stop ${stop:.2f} (+5%) | Target ${target:.2f} (-12%)")

    if not buys and not sells:
        print("\nNo actionable signals. All HOLD.")

    print()


if __name__ == "__main__":
    main()
