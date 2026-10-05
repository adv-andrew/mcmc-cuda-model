# MCMC Options Trading System

Monte Carlo research and trading toolkit for options. The main strategy is
the **Options Swing Strategy**: call debit spreads on SPY / QQQ / IWM, bought
when an index ETF pulls back sharply inside a long-term uptrend, and held
about 3-7 trading days until it recovers.

> Full research write-up: [`docs/OPTIONS_SWING_STRATEGY.md`](docs/OPTIONS_SWING_STRATEGY.md)

## Quick Start

```bash
pip install -r requirements.txt

# Today's signals (run ~15 min before the close)
python scripts/options_swing_signals.py

# Full backtest with in/out-of-sample split, stress tests, Monte Carlo sizing
python scripts/backtest_options_swing.py
```

Data comes from public GitHub mirrors (daily SPY/QQQ/IWM/AAPL/NVDA + CBOE VIX)
and falls back to yfinance, so it also works where Yahoo is blocked.

---

## The Strategy in One Screen

| | Rule |
|---|---|
| **Universe** | SPY, QQQ, IWM |
| **Setup** | Close > 200-day SMA **and** ≥ 1.5 ATR below the 5-day high |
| **Trigger** (any) | RSI(2) < 10 · IBS < 0.25 with RSI(2) < 15 · close < lower Bollinger(20, 2) · pullback ≥ 2 ATR |
| **Position** | Buy ~0.55Δ call / sell ~0.30Δ call, nearest Friday ≥ 21 days out |
| **Exit** | First close above the 5-day SMA after ≥ 3 days held · or day 7 · or +60% of max profit |
| **Size** | 5% of equity per spread (debit = max loss), max 2 open |
| **Confidence** | 40 base, +30 if 21-day/weekly/monthly trends are all up, +30 if 20-day realized vol < 15%. HIGH = 100 |

### Backtest (realistic pricing: VIX-based IV, skew, bid/ask, commissions)

| Period | Trades | Win | Avg return on risk | Profit factor | CAGR | Max DD | Sharpe |
|---|---|---|---|---|---|---|---|
| 2011-2019 (rules chosen here) | 212 | 67% | +8.2% | 1.72 | +9.4% | −20.3% | 0.73 |
| 2020-2026 (held out) | 144 | 64% | +7.9% | 1.66 | +8.2% | −15.6% | 0.71 |
| HIGH confidence only, 2020-2026 | 44 | 82% | +23.1% | 6.34 | +7.7% | −7.2% | 1.31 |

Average hold is 4 trading days, at about 23 trades a year. All 36 neighboring
parameter settings are profitable in both periods. The edge **does not
survive 3× normal bid/ask costs**, so always use limit orders near the mid.

### What the research found

- **The original MCMC slope signal predicts the opposite direction** at
  3-5 days: rank IC −0.05 to −0.12, and SELL signals were followed by
  up-moves 53-62% of the time. That's because its GBM median just
  extrapolates recent drift.
- Of 34 trader setups tested, **dips inside uptrends** reliably beat the
  baseline, both in-sample and out-of-sample. Breakouts and trend stacks ≈
  baseline. **Every bearish setup lost money.**
- **Put credit spreads win 70-87% of the time but lose money** on 3-5 day
  holds after costs and skew. **Call debit spreads** capture the bounce best.
- Single-stock options (AAPL, NVDA) under the same rules lose to their
  wider spreads. Trade the index ETFs.

---

## Commands

| Command | Description |
|---|---|
| `python scripts/options_swing_signals.py` | **Today's signals** with spread tickets and exit status |
| `python scripts/backtest_options_swing.py` | Backtest, per-year/tier tables, stress tests, Monte Carlo sizing |
| `python scripts/research_strategy_lab.py` | Event study: 34 setups × 1/2/3/5/10-day holds, IS vs OOS |
| `python scripts/research_options_structures.py [--stocks]` | Long calls vs debit spreads vs credit spreads |
| `python scripts/research_filters.py` | All 16 combinations of the confirmation filters |
| `python scripts/research_mcmc_audit.py` | Walk-forward skill of the MCMC models |
| `python scripts/get_signals.py` / `run_backtest.py` | Legacy stock signals / stock backtest |

`options_now.py`, `options_signal_v4.py` and `backtest_options_v4.py` are
**deprecated**. Their option model (`intrinsic + 0.4·RV·√T`, no implied-vol
premium, no skew, no costs) overstated results.

## Architecture

```
backtesting/
  market_data.py        # GitHub-mirror + yfinance loader, VIX, split handling
  options_pricing.py    # Black-Scholes, VIX-based IV model, skew, cost model
  options_backtest.py   # Portfolio options backtester (daily MTM) + Monte Carlo
  engine.py, metrics.py # Stock backtesting engine
trading/
  options_swing.py      # Strategy rules, confidence score, trade tickets
  features.py           # Indicators incl. weekly/monthly (no-lookahead) features
  swing_signals.py      # Catalog of 34 trader setups used by the strategy lab
  regime_mcmc.py        # Regime-switching Markov-chain Monte Carlo (research)
  indicator.py          # Original GBM MCMCIndicator
config/
  default.yaml          # incl. the options_swing section
docs/
  OPTIONS_SWING_STRATEGY.md
```

## Configuration

Strategy parameters live in the `options_swing` section of
`config/default.yaml`: universe, entry thresholds, deltas/DTE, exits, sizing,
and confidence tiers. The research docs explain how each value was chosen.
Change them only if you re-run the in-sample/out-of-sample checks.

## GPU Acceleration

The original `MCMCIndicator` uses CuPy for CUDA-accelerated simulation when
available and falls back to NumPy automatically.

## Risk Warning

- Backtests use modeled option prices (from VIX), not historical option
  chains, and daily closes only. Past results do not guarantee future
  performance.
- A debit spread can lose 100% of the premium paid. Expect roughly one
  losing year in six (see the Monte Carlo table in the docs).
- Paper trade first, and size so that a −15% to −20% drawdown is tolerable.

## License

MIT License
