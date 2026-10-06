# MCMC Options Trading System

Monte Carlo research and trading toolkit for options. The main strategy is
the **Options Swing Strategy**: in-the-money calls on SPY / QQQ / IWM, bought
when an index ETF pulls back sharply inside a long-term uptrend, and held
about 3-10 trading days until it recovers. The same signal traded in shares
was the most reliable variant tested.

> Full research write-up: [`docs/OPTIONS_SWING_STRATEGY.md`](docs/OPTIONS_SWING_STRATEGY.md)

### Recommended: portfolio mode

| 1996-2026, single-account backtest | CAGR | Max DD | Sharpe (excess) |
|---|---|---|---|
| **85% SPY trend core + options dip overlay** | **+13.5%** | **−27%** | **0.67** |
| same, with harsh option pricing | +10.7% | −29% | 0.54 |
| SPY buy & hold | +10.4% | −55% | 0.49 |

Hold 85% of the account in SPY while SPY's month-end close is above its
10-month average (otherwise T-bills), and buy the dip calls below with 6% of
the account each. Returns were 13-14% in each era: 1996-2010, 2011-2019 and
2020-2026. The scanner prints the core instruction. See section 7 of the
write-up and `scripts/research_core_overlay.py`.

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
| **Position** | Buy one ~0.80Δ (in-the-money) call, nearest Friday ≥ 30 days out |
| **Alternative** | Buy ~1/3 of the account in the ETF's shares, same exits (most reliable) |
| **Exit** | First close above the 5-day SMA after ≥ 3 days held · or day 10 · or +60% |
| **Size** | 6% of equity per position (premium = max loss), max 2 open |
| **Fill rule** | Skip the trade if you can't fill within ~2% of the estimated debit |
| **Confidence** | 40 base, +30 if 21-day/weekly/monthly trends are all up, +30 if 20-day realized vol < 15%. HIGH = 100 |

### Backtest (realistic pricing: VIX-based IV, skew, bid/ask, commissions)

| Period | Trades | Win | Avg return on risk | CAGR | Max DD | Sharpe |
|---|---|---|---|---|---|---|
| 2011-2019 (rules chosen here) | 209 | 66% | +4.8% | +6.5% | −16.2% | 0.67 |
| 2020-2026 (held out) | 143 | 61% | +4.0% | +4.9% | −11.2% | 0.57 |
| *SPY buy & hold, 2011-2026* | | | | +14.1% | −33.7% | 0.86 |

| 2000-2026, same signal | CAGR | Max DD | Sharpe | Worst year |
|---|---|---|---|---|
| Options (calls) on SPY/QQQ/IWM | +4.9% | −19.9% | 0.52 | −11.8% |
| **Shares** on SPY/QQQ/IWM | +3.9% | **−11.8%** | **0.66** | **−7.5%** |
| SPY buy & hold | +8.3% | −55.2% | 0.51 | −36.8% |

The strategy is invested only ~14% of the time, so keep idle cash in
T-bills (not credited above).

### Is it real? (bias audit: `scripts/research_bias_checks.py`)

- **The timing edge is confirmed.** On untouched 1991-2009 data the same
  entry/exit days returned +0.62% on the underlying with a 72% win rate
  (t = 5.7), the same as 2011-2026 (t = 5.5). It beats random uptrend days
  at p ≈ 0.001 in both eras, with no option model involved.
- **The options profit is real but thin.** Paying ~1.5 vol points more than
  the model at entry removes most of it, hence the 2% fill rule. The deflated
  Sharpe ratio (correcting for ~550 configurations tried) gives only an
  11-51% chance that the portfolio Sharpe reflects a real edge.
- **The original 0.55/0.30 call debit spread was overstated.** Its edge
  depended on the call-skew assumption and faded on 1991-2009. It's still
  available via `structure_kind: call_debit_spread`.
- **Expect a few percent a year**, with roughly one losing year in five.
  Paper trade first and compare your real fills to the scanner's estimates.

### Robustness round (`scripts/research_robustness.py`)

- **The structure was chosen by worst case**, picked on 2011-2019 under
  pricing stress: 0.80Δ call, hold ≤10. It held up better than the previous
  0.70Δ default in both held-out windows when paying +10% IV, and cut the
  number of −75% trades from 25 to 6.
- **13 never-used ETFs** (DIA, MDY, IJR, RSP, sector SPDRs, EFA, EEM) show
  the same timing edge on the underlying (t = 4-5, placebo p < 0.0002). But
  their wider option spreads turn it into a loss, so the options universe
  stays SPY/QQQ/IWM.
- **Shares on the core 3 ETFs** had the best risk-adjusted returns and were
  positive in every era (`scripts/research_shares_vs_options.py`).

### What the research found

- **The original MCMC slope signal predicts the opposite direction** at
  3-5 days: rank IC −0.05 to −0.12, and SELL signals were followed by
  up-moves 53-62% of the time. Its GBM median just extrapolates recent drift.
- Of 34 trader setups tested, **dips inside uptrends** reliably beat the
  baseline. Breakouts and trend stacks ≈ baseline. **Every bearish setup
  lost money.**
- **Put credit spreads win 70-87% of the time but lose money** on 3-5 day
  holds after costs and skew.
- Single-stock options (AAPL, NVDA) under the same rules lose to their wider
  spreads. Trade the index ETFs.

---

## Commands

| Command | Description |
|---|---|
| `python scripts/options_swing_signals.py` | **Today's signals**: core instruction, order tickets, exit status |
| `python scripts/options_swing_signals.py --account 50000` | Same, with exact whole-contract and share counts for your account |
| `python scripts/paper_trade.py update` / `report` / `fill ID PRICE` | **Paper-trade** portfolio mode with the backtest engine; tracks real fills vs model |
| `python scripts/backtest_options_swing.py` | Backtest, per-year/tier tables, stress tests, Monte Carlo sizing |
| `python scripts/research_strategy_lab.py` | Event study: 34 setups × 1/2/3/5/10-day holds, IS vs OOS |
| `python scripts/research_options_structures.py [--stocks]` | Long calls vs debit spreads vs credit spreads |
| `python scripts/research_filters.py` | All 16 combinations of the confirmation filters |
| `python scripts/research_mcmc_audit.py` | Walk-forward skill of the MCMC models |
| `python scripts/research_bias_checks.py` | Untouched 1991-2009 test, placebo, no-options, pricing stress, deflated Sharpe |
| `python scripts/research_robustness.py` | Pre-registered worst-case structure choice; validation on 13 fresh ETFs |
| `python scripts/research_shares_vs_options.py` | Shares vs options vs hybrid portfolios, 2000-2026 |
| `python scripts/research_core_overlay.py` | Portfolio mode: trend core + options overlay, single-account validation |
| `python scripts/research_exits.py` | Pre-registered exit study (winner failed validation; no change) |
| `python scripts/get_signals.py` / `run_backtest.py` | Legacy stock signals / stock backtest |

`options_now.py`, `options_signal_v4.py` and `backtest_options_v4.py` are
**deprecated**. Their option model (`intrinsic + 0.4·RV·√T`, no implied-vol
premium, no skew, no costs) overstated results.

## Architecture

```
backtesting/
  market_data.py        # GitHub-mirror + yfinance loader, VIX, split handling
  options_pricing.py    # Black-Scholes, VIX-based IV model, skew, cost model
  options_backtest.py   # Portfolio options backtester (daily MTM), per-trade simulator, Monte Carlo
  shares_backtest.py    # Same signal traded in ETF shares
  portfolio.py          # Trend core (Faber), sleeve blending, single-account simulator
  core_overlay_engine.py # Step engine shared by the backtest and the paper ledger
  engine.py, metrics.py # Stock backtesting engine
trading/
  options_swing.py      # Strategy rules, confidence score, trade tickets, core status
  paper_ledger.py       # Paper-trading ledger (JSON state, fill tracking)
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
- An option can lose 100% of the premium paid. Expect roughly one losing
  year in five (see the Monte Carlo table in the docs).
- Paper trade first, and size so that a −15% to −20% drawdown is tolerable.

## License

MIT License
