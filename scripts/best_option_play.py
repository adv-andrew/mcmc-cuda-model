"""Get the single best options play for 30-45 DTE.

Ranks signals by absolute slope (strongest trend) and outputs a clear recommendation.

Usage:
    python scripts/best_option_play.py
"""

import sys
sys.path.insert(0, ".")

import yaml
from datetime import datetime, timedelta
from backtesting.data_loader import DataLoader
from trading.indicator import MCMCIndicator


def main():
    # Load config
    with open("config/tickers.yaml") as f:
        tickers = yaml.safe_load(f).get("watchlist", [])

    loader = DataLoader(cache_dir="data/cache")
    indicator = MCMCIndicator(n_simulations=25000, n_regimes=3, slope_threshold=15.0)

    signals = []

    for ticker in tickers:
        df_1d = loader.fetch(ticker, timeframe="1d", days=90, use_cache=False)
        df_1h = loader.fetch(ticker, timeframe="1h", days=7, use_cache=False)

        if df_1d.empty or len(df_1d) < 10:
            continue

        sig_1d = indicator.generate_signal(ticker, df_1d, "1d")

        # Check MTF alignment
        if not df_1h.empty and len(df_1h) >= 5:
            sig_1h = indicator.generate_signal(ticker, df_1h, "1h")
            mtf_aligned = sig_1h["direction"] == sig_1d["direction"] and sig_1d["direction"] != "NEUTRAL"
        else:
            mtf_aligned = True

        # Only consider actionable signals with MTF alignment
        if sig_1d["suggested_action"] in ["BUY", "SELL"] and mtf_aligned:
            signals.append({
                "ticker": ticker,
                "action": sig_1d["suggested_action"],
                "direction": sig_1d["direction"],
                "price": sig_1d["current_price"],
                "slope": sig_1d["slope_degrees"],
                "abs_slope": abs(sig_1d["slope_degrees"]),
                "strength": sig_1d["signal_strength"],
            })

    if not signals:
        print("\n" + "=" * 60)
        print("NO ACTIONABLE SIGNALS")
        print("=" * 60)
        print("\nAll tickers are HOLD or have MTF conflicts.")
        print("Check back later.")
        return

    # Sort by absolute slope (strongest trend first)
    signals.sort(key=lambda x: x["abs_slope"], reverse=True)

    # Calculate expiration target (30-45 DTE)
    today = datetime.now()
    exp_start = today + timedelta(days=30)
    exp_end = today + timedelta(days=45)

    print("\n" + "=" * 60)
    print("BEST OPTIONS PLAY - 30-45 DTE")
    print("=" * 60)

    best = signals[0]

    if best["action"] == "BUY":
        option_type = "CALL"
        # Slightly OTM for leverage
        strike = round(best["price"] * 1.03, 0)  # 3% OTM
        stop_price = best["price"] * 0.95  # -5% underlying = exit
        target_price = best["price"] * 1.12  # +12% underlying = target
    else:
        option_type = "PUT"
        strike = round(best["price"] * 0.97, 0)  # 3% OTM
        stop_price = best["price"] * 1.05  # +5% underlying = exit
        target_price = best["price"] * 0.88  # -12% underlying = target

    print(f"""
+------------------------------------------------------------+
|  {best['ticker']} {option_type}
|
|  Strike:     ${strike:.0f} (slightly OTM)
|  Expiration: {exp_start.strftime('%b %d')} - {exp_end.strftime('%b %d, %Y')}
|
|  Current Price: ${best['price']:.2f}
|  Slope:         {best['slope']:+.1f} deg (rank #1)
|  Strength:      {best['strength']:.2f}
|  Direction:     {best['direction']} (1h + 1d aligned)
+------------------------------------------------------------+

EXIT RULES:
  * Take profit: Option +100% OR underlying hits ${target_price:.2f}
  * Stop loss:   Option -50% OR underlying hits ${stop_price:.2f}
  * Signal flip: Exit if direction changes on next check
""")

    # Show runners up
    if len(signals) > 1:
        print("RUNNERS UP (by slope strength):")
        print("-" * 60)
        for i, sig in enumerate(signals[1:4], 2):
            opt = "CALL" if sig["action"] == "BUY" else "PUT"
            print(f"  #{i} {sig['ticker']:5} {opt:4} | ${sig['price']:.2f} | Slope: {sig['slope']:+.1f}° | Str: {sig['strength']:.2f}")

    print()


if __name__ == "__main__":
    main()
