"""Get options signals NOW - even in neutral markets.

This is the "I want a trade today" version.
Shows best signals but with confidence levels.

Usage:
    python scripts/options_now.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from trading.indicator import MCMCIndicator


TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "NFLX", "CRM", "ADBE", "INTC", "QCOM", "AVGO", "MU", "SNOW", "PLTR",
    "JPM", "BAC", "GS", "MS", "V", "MA", "PYPL", "SQ",
    "DIS", "NKE", "SBUX", "MCD", "HD", "LOW", "TGT", "COST",
    "UNH", "JNJ", "PFE", "MRNA", "ABBV", "LLY",
    "XOM", "CVX", "COP", "SLB",
    "BA", "CAT", "UPS", "FDX",
    "COIN", "MARA", "RIOT", "SOFI", "HOOD",
]


def fetch_data(ticker, days=120):
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()


def calculate_volatility(df, window=20):
    if len(df) < window + 1:
        return 0.3
    returns = np.log(df["Close"] / df["Close"].shift(1))
    vol = returns.rolling(window).std() * np.sqrt(252)
    return vol.iloc[-1] if len(vol) > 0 else 0.3


def get_regime(spy_df):
    if len(spy_df) < 200:
        return "NEUTRAL", 0, {}

    close = spy_df["Close"]
    current = close.iloc[-1]

    stats = {
        "price": current,
        "ma_20": close.rolling(20).mean().iloc[-1],
        "ma_50": close.rolling(50).mean().iloc[-1],
        "ma_200": close.rolling(200).mean().iloc[-1],
        "mom_5d": (current / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0,
        "mom_1m": (current / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0,
    }

    above_200 = current > stats["ma_200"]
    above_50 = current > stats["ma_50"]
    mom_pos = stats["mom_1m"] > 0

    if above_200 and above_50 and mom_pos:
        return "BULL", 1, stats
    elif not above_200 and not above_50 and not mom_pos:
        return "BEAR", -1, stats
    else:
        return "NEUTRAL", 0, stats


def main():
    print()
    print("=" * 70)
    print("OPTIONS SIGNALS - CURRENT BEST PLAYS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    spy_df = fetch_data("SPY", days=300)
    if spy_df.empty:
        print("ERROR: Could not fetch SPY data")
        return

    regime, regime_dir, spy_stats = get_regime(spy_df)

    print(f"\n--- MARKET STATUS ---")
    print(f"SPY: ${spy_stats['price']:.2f}")
    print(f"vs 200 MA: {'ABOVE' if spy_stats['price'] > spy_stats['ma_200'] else 'BELOW'} (${spy_stats['ma_200']:.2f})")
    print(f"vs 50 MA:  {'ABOVE' if spy_stats['price'] > spy_stats['ma_50'] else 'BELOW'} (${spy_stats['ma_50']:.2f})")
    print(f"1-month:   {spy_stats['mom_1m']:+.1f}%")
    print(f"5-day:     {spy_stats['mom_5d']:+.1f}%")
    print(f"REGIME:    {regime}")

    if regime == "NEUTRAL":
        print("\n*** WARNING: NEUTRAL REGIME ***")
        print("Backtest shows lower win rate in neutral markets.")
        print("Signals below are LOWER CONFIDENCE.")

    indicator = MCMCIndicator(n_simulations=25000, n_regimes=3, slope_threshold=15.0)

    all_signals = []

    print("\nScanning tickers...")

    for ticker in TICKERS:
        df = fetch_data(ticker, days=120)
        if df.empty or len(df) < 60:
            continue

        vol = calculate_volatility(df)
        if vol < 0.15 or vol > 0.90:
            continue

        try:
            sig = indicator.generate_signal(ticker, df, "1d")

            if sig["signal_strength"] < 0.70:
                continue

            if abs(sig["slope_degrees"]) < 15:
                continue

            is_call = sig["suggested_action"] == "BUY"

            mom_5d = (df["Close"].iloc[-1] / df["Close"].iloc[-5] - 1) * 100
            mom_20d = (df["Close"].iloc[-1] / df["Close"].iloc[-20] - 1) * 100

            # Calculate confidence score
            confidence = 0

            # Base score from strength and slope
            confidence += sig["signal_strength"] * 30
            confidence += min(abs(sig["slope_degrees"]) / 2, 20)

            # Momentum alignment
            if is_call and mom_5d > 0:
                confidence += 10
            elif not is_call and mom_5d < 0:
                confidence += 10

            if is_call and mom_20d > 0:
                confidence += 10
            elif not is_call and mom_20d < 0:
                confidence += 10

            # Regime alignment bonus
            if (is_call and regime == "BULL") or (not is_call and regime == "BEAR"):
                confidence += 20
            elif regime == "NEUTRAL":
                confidence -= 10

            # SPY trend alignment
            if (is_call and spy_stats["mom_5d"] > 0) or (not is_call and spy_stats["mom_5d"] < 0):
                confidence += 10

            confidence = min(100, max(0, confidence))

            all_signals.append({
                "ticker": ticker,
                "type": "CALL" if is_call else "PUT",
                "price": sig["current_price"],
                "slope": sig["slope_degrees"],
                "strength": sig["signal_strength"],
                "vol": vol,
                "mom_5d": mom_5d,
                "mom_20d": mom_20d,
                "confidence": confidence,
            })

        except:
            continue

    if not all_signals:
        print("\nNo signals meet minimum criteria.")
        return

    # Sort by confidence
    all_signals.sort(key=lambda x: x["confidence"], reverse=True)

    print(f"\nFound {len(all_signals)} signals. Top 5:")

    print("\n" + "=" * 70)

    for i, s in enumerate(all_signals[:5], 1):
        conf_label = "HIGH" if s["confidence"] >= 70 else "MEDIUM" if s["confidence"] >= 50 else "LOW"
        strike = round(s["price"], 0)

        exp_date = datetime.now() + timedelta(days=35)

        print(f"""
#{i} {s['ticker']} {s['type']} | Confidence: {s['confidence']:.0f}/100 ({conf_label})
   Strike: ${strike:.0f} ATM | Exp: ~{exp_date.strftime('%b %d')}
   Price: ${s['price']:.2f} | Vol: {s['vol']*100:.0f}%
   Strength: {s['strength']:.2f} | Slope: {s['slope']:+.1f}
   5d: {s['mom_5d']:+.1f}% | 20d: {s['mom_20d']:+.1f}%
""")

    print("=" * 70)
    print("EXIT RULES: +80% take profit | -35% stop loss | 35 day max hold")
    print("POSITION SIZE: 10% of portfolio per trade")
    print("=" * 70)

    # Summary table
    print("\nQUICK REFERENCE:")
    print("-" * 70)
    print(f"{'Rank':<5} {'Ticker':<6} {'Type':<5} {'Strike':>8} {'Conf':>6} {'Action':<15}")
    print("-" * 70)

    for i, s in enumerate(all_signals[:5], 1):
        strike = round(s["price"], 0)
        conf_label = "HIGH" if s["confidence"] >= 70 else "MED" if s["confidence"] >= 50 else "LOW"
        action = f"BUY {s['ticker']} ${strike:.0f} {s['type']}"
        print(f"{i:<5} {s['ticker']:<6} {s['type']:<5} ${strike:>7.0f} {conf_label:>6} {action:<15}")

    print("-" * 70)
    print(f"\nBest play: {all_signals[0]['ticker']} ${round(all_signals[0]['price']):.0f} {all_signals[0]['type']}")
    print(f"           Confidence: {all_signals[0]['confidence']:.0f}/100")


if __name__ == "__main__":
    main()
