# MCMC Options Trading System

Monte Carlo Markov Chain trading system for options. Uses regime-switching simulations to generate probabilistic price forecasts and high-probability options signals.

## Backtested Performance (2022-2025)

| Metric | Value |
|--------|-------|
| **Total Trades** | 40 |
| **Win Rate** | 60% |
| **Profit Factor** | 5.5x |
| **Total Return** | +1012% |

*Backtest uses 10% position sizing, ATM options, 35 DTE, +80% TP / -35% SL*

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Get today's options signals
python scripts/options_now.py

# Run options backtest
python scripts/backtest_options_v4.py
```

---

## How to Use the Strategy

### The System

1. **MCMC simulates 25,000 price paths** to predict direction
2. **Filters for high-probability setups** (strength, slope, momentum, regime)
3. **Outputs ranked options signals** with confidence scores

### Your Weekly Routine

| Day | Action |
|-----|--------|
| **Monday AM** | Run `python scripts/options_now.py` |
| **If signal 70+** | Buy ATM option, ~35 DTE |
| **Set alerts** | +80% take profit, -35% stop loss |
| **Wait** | Let it play out |

### Position Sizing

| Rule | Value |
|------|-------|
| Position size | 10% of portfolio per trade |
| Max positions | 3-4 open at once |
| Strike | ATM (at the money) |
| Expiration | 30-40 DTE |

### Exit Rules

| Condition | Action |
|-----------|--------|
| Option up +80% | **Sell - Take Profit** |
| Option down -35% | **Sell - Stop Loss** |
| 30+ days held | **Sell - Time Exit** |

### Example ($10,000 Account)

```
Monday:
  1. Run: python scripts/options_now.py
  2. Output: "HD $339 PUT - Confidence 72/100"
  3. Buy 2x HD $339 PUT @ $4.80 = $960
  4. Set alerts: TP at $8.64 (+80%), SL at $3.12 (-35%)
  5. Done. Check again next Monday.

Outcome A: HD drops, option hits $9.00 → Sell → +87% win
Outcome B: HD rises, option hits $3.00 → Sell → -37% loss
Outcome C: 30 days pass → Sell at market → Variable
```

### Key Rules

- **Only buy on Monday/Tuesday** (avoid weekend theta decay)
- **ATM strikes only** (higher probability than OTM)
- **Same dollar amount per trade** (not same # of contracts)
- **Max 4 positions** at once
- **Different tickers** (don't stack same stock)

---

## Commands

| Command | Description |
|---------|-------------|
| `python scripts/options_now.py` | Get today's signals with confidence scores |
| `python scripts/options_signal_v4.py` | Conservative - only shows signals in clear regimes |
| `python scripts/backtest_options_v4.py` | Run full backtest (2022-2025) |
| `python scripts/get_signals.py` | Stock signals (not options) |
| `python scripts/run_backtest.py` | Stock backtest |

---

## Understanding the Output

```
#1 HD PUT | Confidence: 72/100 (HIGH)
   Strike: $339 ATM | Exp: ~Apr 17
   Price: $338.93 | Vol: 23%
   Strength: 0.78 | Slope: -60.1
   5d: -4.7% | 20d: -12.6%
```

| Field | Meaning |
|-------|---------|
| **Confidence** | 70+ = HIGH (take it), 50-69 = MEDIUM (caution), <50 = LOW (skip) |
| **Strength** | MCMC signal strength (0.72+ required) |
| **Slope** | Trend steepness in degrees (20+ required) |
| **Vol** | Stock's volatility (20-85% range required) |
| **5d/20d** | Recent momentum confirmation |

---

## System Architecture

```
scripts/
  options_now.py         # Main signal generator (use this)
  options_signal_v4.py   # Conservative signal generator
  backtest_options_v4.py # Options backtester
  get_signals.py         # Stock signals

trading/
  indicator.py           # MCMCIndicator - core signal engine

config/
  tickers.yaml           # Watchlist (50+ stocks)
  best_params.json       # Optimized parameters
```

---

## How It Works (Technical)

### Signal Generation

1. **Monte Carlo Simulation** - 25,000 price paths using historical volatility
2. **Slope Calculation** - Forecast median vs current price, normalized
3. **Regime Detection** - SPY determines bull/bear/neutral market
4. **Multi-Filter** - Strength >= 0.72, Slope >= 20°, momentum confirmation

### Regime Filter

| SPY Condition | Regime | Allowed Trades |
|---------------|--------|----------------|
| Above 50 & 200 MA + positive momentum | BULL | CALLs only |
| Below 50 & 200 MA + negative momentum | BEAR | PUTs only |
| Mixed signals | NEUTRAL | Lower confidence |

### Why It Works

- **Momentum continuation** - We bet trends continue, not reverse
- **Multiple filters** - Only ~1-2 trades per week qualify
- **Asymmetric payoff** - +80% wins vs -35% losses = profitable at 45% win rate
- **Regime alignment** - Trade with the market, not against it

---

## Backtest Results by Period

| Period | Trades | Win Rate | Profit Factor | Return |
|--------|--------|----------|---------------|--------|
| 2022 Bear | 7 | 29% | 1.34 | +5% |
| 2023 Bull | 7 | 29% | 1.70 | +12% |
| 2024 Full | 12 | 67% | 6.84 | +119% |
| 2025 YTD | 14 | 86% | 23.65 | +336% |

---

## Risk Warning

- Past backtest results do not guarantee future performance
- Options can expire worthless (100% loss on position)
- Only trade money you can afford to lose
- Consider paper trading first to learn the system

---

## Configuration

### Key Parameters (config/best_params.json)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_simulations` | 25,000 | Monte Carlo paths |
| `slope_threshold` | 15.0 | Degrees for signal |
| `signal_strength_min` | 0.72 | Minimum to trade |
| `position_size_pct` | 0.10 | 10% per trade |

### Ticker Universe (config/tickers.yaml)

50+ liquid stocks including:
- Mega caps: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
- Tech: AMD, NFLX, CRM, ADBE, AVGO
- Finance: JPM, BAC, GS, V, MA
- Consumer: HD, COST, NKE, DIS
- And more...

---

## GPU Acceleration

Uses CuPy for CUDA-accelerated simulations when available:

- **CPU:** ~1,000 simulations (NumPy)
- **GPU:** ~50,000 simulations (CuPy/CUDA)

Falls back to NumPy automatically if no GPU detected.

---

## License

MIT License
