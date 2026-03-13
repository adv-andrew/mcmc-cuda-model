"""Options Backtest V4 - Looser regime filter but more signals.

V3 was too restrictive (only STRONG regimes). This version:
- Allows trading in BULL and BEAR (not just STRONG)
- Adds additional confirmation filters to compensate
- More trades = better statistical validation

Usage:
    python scripts/backtest_options_v4.py
"""

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from trading.indicator import MCMCIndicator


PERIODS = [
    {"name": "2022 Bear", "start": "2022-01-01", "end": "2022-10-31"},
    {"name": "2023 Bull", "start": "2023-01-01", "end": "2023-12-31"},
    {"name": "2024 Full", "start": "2024-01-01", "end": "2024-12-31"},
    {"name": "2025 YTD", "start": "2025-01-01", "end": "2025-12-31"},
]

TICKERS = [
    "SPY", "QQQ",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "NFLX", "CRM", "ADBE", "INTC", "QCOM", "AVGO",
    "JPM", "BAC", "GS", "V", "MA",
    "DIS", "NKE", "HD", "COST",
    "XOM", "CVX",
    "BA", "CAT",
]


def fetch_data(tickers, start, end):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    warmup_start = (start_dt - timedelta(days=300)).strftime("%Y-%m-%d")

    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=warmup_start, end=end, progress=False)
            if not df.empty and len(df) >= 60:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
        except:
            pass
    return data


def calculate_volatility(df, window=20):
    if len(df) < window + 1:
        return 0.3
    returns = np.log(df["Close"] / df["Close"].shift(1))
    vol = returns.rolling(window).std() * np.sqrt(252)
    return vol.iloc[-1] if len(vol) > 0 else 0.3


def get_regime(spy_data, current_date):
    """Regime detection."""
    hist = spy_data.loc[:current_date]
    if len(hist) < 200:
        return "neutral", 0

    close = hist["Close"]
    ma_20 = close.rolling(20).mean().iloc[-1]
    ma_50 = close.rolling(50).mean().iloc[-1]
    ma_200 = close.rolling(200).mean().iloc[-1]
    current = close.iloc[-1]

    mom_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
    mom_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) >= 63 else 0

    # Simplified regime: above/below 200 MA + momentum
    if current > ma_200 and current > ma_50 and mom_1m > 0:
        return "bull", 1
    elif current < ma_200 and current < ma_50 and mom_1m < 0:
        return "bear", -1
    else:
        return "neutral", 0


def calculate_option_price(underlying_price, strike, is_call, days_to_exp, volatility=0.3):
    T = days_to_exp / 365
    time_value = 0.4 * volatility * np.sqrt(T)

    if is_call:
        intrinsic = max(0, (underlying_price - strike) / underlying_price)
    else:
        intrinsic = max(0, (strike - underlying_price) / underlying_price)

    option_pct = intrinsic + time_value
    return option_pct * underlying_price


def simulate_option_trade(entry_price, price_series, is_call, strike_pct, vol, days_to_exp=35):
    strike = entry_price * strike_pct
    entry_option = calculate_option_price(entry_price, strike, is_call, days_to_exp, vol)

    for day, current_price in enumerate(price_series):
        if day >= days_to_exp:
            if is_call:
                exit_option = max(0, current_price - strike)
            else:
                exit_option = max(0, strike - current_price)

            option_return = (exit_option - entry_option) / entry_option if entry_option > 0 else -1
            return current_price, day, "expiration", option_return

        remaining_days = days_to_exp - day
        current_option = calculate_option_price(current_price, strike, is_call, remaining_days, vol)
        option_return = (current_option - entry_option) / entry_option if entry_option > 0 else -1

        if option_return >= 0.80:  # +80% profit (slightly lower target)
            return current_price, day, "take_profit", option_return
        elif option_return <= -0.35:  # -35% loss (slightly tighter stop)
            return current_price, day, "stop_loss", option_return

    return price_series[-1], len(price_series), "end", option_return


def run_backtest_period(data, start_date, end_date, indicator):
    spy_data = data.get("SPY")
    if spy_data is None:
        return None, {}

    all_dates = []
    for df in data.values():
        all_dates.extend(df.index.tolist())
    all_dates = sorted(set(all_dates))

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    dates = [d for d in all_dates if start_ts <= d <= end_ts]

    if len(dates) < 50:
        return None, {}

    trades = []
    check_interval = 5

    i = 0
    while i < len(dates) - 40:
        current_date = dates[i]
        regime, regime_dir = get_regime(spy_data, current_date)

        if i % check_interval != 0:
            i += 1
            continue

        # Allow bull and bear (not neutral)
        if regime == "neutral":
            i += 1
            continue

        candidates = []

        for ticker, df in data.items():
            if ticker in ["SPY", "QQQ"]:
                continue

            hist = df.loc[:current_date]
            if len(hist) < 60:
                continue

            stock_vol = calculate_volatility(hist)
            if stock_vol < 0.20 or stock_vol > 0.85:
                continue

            try:
                sig = indicator.generate_signal(ticker, hist, "1d")

                # Filters (slightly relaxed from V3)
                if sig["signal_strength"] < 0.72:
                    continue
                if abs(sig["slope_degrees"]) < 20:
                    continue

                is_call = sig["suggested_action"] == "BUY"

                # Regime alignment
                if is_call and regime != "bull":
                    continue
                if not is_call and regime != "bear":
                    continue

                # Momentum confirmation
                recent_5d = (hist["Close"].iloc[-1] / hist["Close"].iloc[-5] - 1) * 100
                recent_20d = (hist["Close"].iloc[-1] / hist["Close"].iloc[-20] - 1) * 100

                if is_call:
                    if recent_5d < 0.5 or recent_20d < 1:
                        continue
                else:
                    if recent_5d > -0.5 or recent_20d > -1:
                        continue

                # Additional filter: Stock trending same direction as SPY
                spy_5d = (spy_data.loc[:current_date]["Close"].iloc[-1] /
                         spy_data.loc[:current_date]["Close"].iloc[-5] - 1) * 100
                if is_call and spy_5d < 0:
                    continue
                if not is_call and spy_5d > 0:
                    continue

                candidates.append({
                    "ticker": ticker,
                    "action": sig["suggested_action"],
                    "price": sig["current_price"],
                    "slope": sig["slope_degrees"],
                    "strength": sig["signal_strength"],
                    "is_call": is_call,
                    "vol": stock_vol,
                })

            except:
                pass

        if candidates:
            candidates.sort(key=lambda x: x["strength"] * abs(x["slope"]), reverse=True)
            best = candidates[0]

            ticker = best["ticker"]
            future_start_idx = dates.index(current_date)
            future_dates = dates[future_start_idx:future_start_idx + 40]

            if ticker in data and len(future_dates) >= 35:
                df = data[ticker]
                price_series = []
                for d in future_dates:
                    if d in df.index:
                        price_series.append(float(df.loc[d, "Close"]))

                if len(price_series) >= 35:
                    strike_pct = 1.00  # ATM

                    exit_price, days_held, exit_reason, option_return = simulate_option_trade(
                        best["price"],
                        price_series[1:],
                        best["is_call"],
                        strike_pct,
                        best["vol"],
                        days_to_exp=35
                    )

                    trades.append({
                        "ticker": ticker,
                        "entry_date": current_date,
                        "is_call": best["is_call"],
                        "days_held": days_held,
                        "option_return": option_return,
                        "exit_reason": exit_reason,
                        "strength": best["strength"],
                        "slope": best["slope"],
                        "regime": regime,
                    })

                    i += max(days_held, 5)
                    continue

        i += 1

    return trades, {}


def main():
    print("=" * 70)
    print("OPTIONS BACKTEST V4: More Trades, Still Profitable?")
    print("=" * 70)
    print("\nChanges from V3:")
    print("  - Allow BULL and BEAR regimes (not just STRONG)")
    print("  - Slightly relaxed filters (strength 0.72, slope 20)")
    print("  - Adjusted targets: +80% TP, -35% SL")
    print("  - Added SPY trend confirmation")
    print()

    indicator = MCMCIndicator(n_simulations=25000, n_regimes=3, slope_threshold=15.0)

    all_trades = []
    period_results = []

    for period in PERIODS:
        print(f"\n{'-' * 70}")
        print(f"Period: {period['name']} ({period['start']} to {period['end']})")
        print(f"{'-' * 70}")

        data = fetch_data(TICKERS, period["start"], period["end"])

        if len(data) < 5 or "SPY" not in data:
            print("  Insufficient data.")
            continue

        trades, _ = run_backtest_period(data, period["start"], period["end"], indicator)

        if trades is None or len(trades) == 0:
            print("  No trades")
            period_results.append({"period": period["name"], "trades": 0, "win_rate": 0, "total_return": 0, "profit_factor": 0})
            continue

        returns = [t["option_return"] for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        position_size = 0.10
        portfolio = 1.0
        for r in returns:
            portfolio *= (1 + r * position_size)
        total_return = (portfolio - 1) * 100

        win_rate = len(wins) / len(returns) * 100 if returns else 0
        avg_win = np.mean(wins) * 100 if wins else 0
        avg_loss = np.mean(losses) * 100 if losses else 0

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        print(f"\n  Trades:        {len(trades)}")
        print(f"  Win Rate:      {win_rate:.1f}%")
        print(f"  Avg Win:       {avg_win:+.1f}%")
        print(f"  Avg Loss:      {avg_loss:+.1f}%")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print(f"  Total Return:  {total_return:+.1f}%")

        print(f"\n  {'Ticker':<6} {'Type':<5} {'Str':>5} {'Days':>5} {'Return':>10} {'Exit':<12}")
        print(f"  {'-'*50}")
        for t in trades:
            opt_type = "CALL" if t["is_call"] else "PUT"
            print(f"  {t['ticker']:<6} {opt_type:<5} {t['strength']:>5.2f} {t['days_held']:>5} {t['option_return']*100:>+9.1f}% {t['exit_reason']:<12}")

        period_results.append({
            "period": period["name"],
            "trades": len(trades),
            "win_rate": win_rate,
            "total_return": total_return,
            "profit_factor": profit_factor,
        })

        all_trades.extend(trades)

    # Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    if all_trades:
        returns = [t["option_return"] for t in all_trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        position_size = 0.10
        portfolio = 1.0
        for r in returns:
            portfolio *= (1 + r * position_size)
        total_return = (portfolio - 1) * 100

        win_rate = len(wins) / len(returns) * 100

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        print(f"\n  Total Trades:     {len(all_trades)}")
        print(f"  Overall Win Rate: {win_rate:.1f}%")
        print(f"  Profit Factor:    {profit_factor:.2f}")
        print(f"  Portfolio Return: {total_return:+.1f}%")

        print(f"\n  {'Period':<15} {'Trades':>7} {'Win Rate':>10} {'PF':>6} {'Total':>12}")
        print(f"  {'-'*55}")
        for pr in period_results:
            print(f"  {pr['period']:<15} {pr['trades']:>7} {pr['win_rate']:>9.1f}% {pr['profit_factor']:>6.2f} {pr['total_return']:>+11.1f}%")

        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)

        if profit_factor >= 1.5 and total_return > 30 and win_rate >= 45:
            print(f"\n  [PROFITABLE] PF={profit_factor:.2f}, Win={win_rate:.0f}%, Return={total_return:+.0f}%")
            print("  Strategy confirmed with more trades!")
        elif profit_factor >= 1.3 and total_return > 15:
            print(f"\n  [MARGINAL EDGE] PF={profit_factor:.2f}, Win={win_rate:.0f}%, Return={total_return:+.0f}%")
        else:
            print(f"\n  [NOT PROFITABLE] PF={profit_factor:.2f}, Win={win_rate:.0f}%, Return={total_return:+.0f}%")

    with open("data/results/options_backtest_v4.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "version": "v4_more_trades",
            "periods": period_results,
            "total_trades": len(all_trades),
            "win_rate": win_rate if all_trades else 0,
            "profit_factor": profit_factor if all_trades else 0,
            "total_return": total_return if all_trades else 0,
        }, f, indent=2)

    print("\nResults saved to data/results/options_backtest_v4.json")


if __name__ == "__main__":
    main()
