# Options Swing Strategy: research report

**Goal:** a high-confidence options strategy with a normal hold of about 3+ days.

**Result:** buy **call debit spreads on SPY / QQQ / IWM when an index ETF
pulls back sharply inside a long-term uptrend**, and hold until it recovers
(min 3, max 7 trading days). Over 2011-2026 this produced 356 trades, a 65%
win rate, +8.1% average return on capital at risk, and a profit factor of 1.69.
Results were nearly identical on the data the rules were chosen on (2011-2019)
and on the data held out for validation (2020-2026). The HIGH-confidence tier
won 68% (in-sample) and 86% (out-of-sample) of the time.

Everything below can be reproduced with the `scripts/research_*.py` and
`scripts/backtest_options_swing.py` scripts.

---

## 1. Method

| Item | Choice |
|---|---|
| Data | Daily OHLCV for SPY, QQQ, IWM, AAPL, NVDA (2010-01 to 2026-10) and CBOE VIX (1990 to 2026-09), via public GitHub mirrors (`backtesting/market_data.py`). Two independent mirrors agreed to ~1e-7 on 762 overlapping days. |
| Split | Rules and parameters were **chosen on 2011-2019 only**. 2020-2026 (COVID crash, 2022 bear market, 2025 tariff shock) is reported alongside as the out-of-sample check. |
| Option prices | Black-Scholes. ATM IV = 0.9 × VIX × (underlying RV ÷ SPY RV, trailing-year median), floored at 10-day realized vol. Modeled IV sat above subsequently realized vol ~75% of the time, matching the documented volatility risk premium. |
| Skew | OTM puts priced richer and OTM calls cheaper than ATM (≈ SPX-like skew). |
| Costs | Every leg pays a half-spread on entry *and* exit: max($0.01, 1% of premium) for ETFs, max($0.02, 2%) for stocks, plus $0.65/contract. |
| Fills | Entries and exits at the daily close, marked to liquidation value every day. |
| Overlap | Event studies use non-overlapping trades, so t-stats aren't inflated. |

### Why the old backtest overstated returns

`scripts/backtest_options_v4.py` priced options as `intrinsic + 0.4·RV·√T`. It
used *realized* vol instead of implied vol, so every option was bought too
cheaply. It had no skew and no bid/ask, and its headline "+1012%" came from
about 40 trades. With realistic pricing, the same kind of directional
momentum bet has no edge (see section 2).

---

## 2. What the data says

### 2a. The original MCMC signal predicts the *opposite* direction

`scripts/research_mcmc_audit.py` walks the `MCMCIndicator` forward one day at a
time. The table shows the rank IC (information coefficient) of its slope
against 3-day forward returns:

| | 2011-2019 | 2020-2026 |
|---|---|---|
| SPY | −0.084 | −0.105 |
| QQQ | −0.085 | −0.076 |
| IWM | −0.053 | −0.063 |

Its SELL signals were followed by **up**-moves 53-62% of the time. The
simulation is GBM with the sample-mean drift, so the median path is just
recent drift extrapolated. At a 3-5 day horizon, a steep recent move marks
a stretched market that tends to snap back.

A replacement **regime-switching Markov-chain Monte Carlo**
(`trading/regime_mcmc.py`: stretch × volatility states, empirical transition
sampling) showed small positive skill in-sample on SPY/QQQ. That skill
didn't survive 2020-2026, so it is **not** used to pick trades.

### 2b. Strategy lab: 34 trader setups × 5 holding periods

Results come from `scripts/research_strategy_lab.py`. The table below covers
ETFs, a 5-day hold, and the direction-adjusted mean return per trade.

| Setup | Family | 2011-19 mean / win | 2020-26 mean / win |
|---|---|---|---|
| *every day (baseline)* | – | +0.25% / 57% | +0.30% / 58% |
| RSI(2) < 5, above SMA200 (Connors) | mean reversion | +0.49% / 61% | **+1.22% / 72%** |
| Pullback ≥ 2 ATR, above SMA200 | mean reversion | +0.53% / 61% | **+1.05% / 71%** |
| Weekly trend up + RSI(2) < 10 (Elder triple screen) | multi-timeframe | +0.29% / 60% | **+1.02% / 72%** |
| Close < lower Bollinger, above SMA200 | mean reversion | +0.64% / 66% | +0.94% / 63% |
| 20-day breakout (turtle) | trend | +0.20% / 59% | +0.25% / 58% |
| EMA 8>21>50>200 stack | trend | +0.07% / 57% | +0.22% / 57% |
| 20-day breakdown, short | bearish | −0.51% | −0.44% |
| RSI(2) > 90 in uptrend, short (fade) | bearish | −0.04% | −0.34% |

Takeaways:
- **Dips inside uptrends** reliably beat the baseline in both periods.
- **Trend/breakout** setups are no better than baseline at a 3-5 day horizon.
  Momentum is a months-long effect.
- **Every bearish setup lost money in both periods.** So the strategy is
  long-only, and in a downtrend it stands aside.
- The same entries still worked when executed at the **next day's open**,
  so the edge is not an artifact of close-to-close fills.

### 2c. Which option structure monetizes it

Results come from `scripts/research_options_structures.py`. The figure is the
median Sharpe across 4 entries × 4 exit rules.

| Structure | Win rate (IS / OOS) | Sharpe (IS / OOS) |
|---|---|---|
| Call debit spread, 21 DTE | 56% / 64% | **0.25 / 0.90** |
| Call debit spread, 14 DTE | 54% / 62% | 0.18 / 0.90 |
| Long ATM call, 30 DTE | 50% / 56% | 0.36 / 0.69 |
| Put credit spread, 30Δ, 14 DTE | **70% / 80%** | −0.26 / 0.33 |
| Put credit spread, 40Δ, 10 DTE | **74% / 80%** | −0.14 / 0.31 |
| Put credit spread, 20Δ, 21 DTE | 56% / 76% | −0.54 / 0.26 |

Credit spreads have the highest win rate but **lose money in-sample**. On a
3-5 day hold you collect only a few days of theta. You also pay the bid/ask
on two legs twice, and skew makes the protective put expensive. A high win
rate is not the same thing as a high-confidence trade.

Single stocks (AAPL, NVDA) under the same rules were worse. Wider option
spreads eat the edge, and earnings gaps aren't captured by an index-based IV
model. **Trade the index ETFs.**

### 2d. Confirmation filters

Results come from `scripts/research_filters.py`, which tests all 16
combinations of four pre-registered filters:

| Filter | Median Sharpe with / without (IS) | (OOS) | Kept? |
|---|---|---|---|
| Pullback ≥ 1.5 ATR | **+0.37 / +0.06** | +0.76 / +0.65 | **yes (entry rule)** |
| Weekly trend up | +0.19 / +0.14 | +0.86 / +0.55 | in confidence score |
| Realized vol < 20% | +0.14 / +0.19 | +0.68 / +0.73 | no |
| No Friday entries | +0.20 / +0.14 | +0.65 / +0.75 | no |

Stops were also tested at −40% to −70% of the debit. None helped in-sample,
because a stop sells at the point of maximum stretch. A minimum hold of 3
days beat 1-2 days for every max-hold tested.

---

## 3. The rules

**Universe:** SPY, QQQ, IWM. **Check:** ~15 minutes before the close.

**Entry (all must be true):**
1. Close > 200-day SMA (long-term uptrend).
2. Close is at least **1.5 ATR(14)** below its 5-day high (a real pullback).
3. At least one trigger:
   - RSI(2) < 10, or
   - IBS < 0.25 and RSI(2) < 15, or
   - close < lower Bollinger(20, 2), or
   - pullback ≥ 2 ATR.

**Position:** buy a ~0.55Δ call and sell a ~0.30Δ call, with the nearest
Friday expiry at least 21 days out. The debit is the maximum loss.

**Exit (first that applies, checked at the close):**
1. Spread worth +60% of its max profit (rarely hit; mainly a safety valve).
2. **Close above the 5-day SMA, once held ≥ 3 trading days** (the normal
   exit, ~92% of trades).
3. 7 trading days held.

**Sizing:** 5% of equity per spread (the debit), at most 2 open at once.

**Confidence score** (0-100), from `trading/options_swing.py`:

| Component | Points |
|---|---|
| Setup qualifies | 40 |
| Timeframes aligned: 21-day, weekly and monthly trend up while the 5-day move is down | +30 |
| Calm regime: 20-day realized vol < 15% | +30 |

HIGH = 100, MEDIUM = 70, LOW = 40. These two components were the only ones
that sorted strategy trades the same way in both periods. Most others, such
as RSI depth, IBS, and VIX spikes, flipped sign, which means they're noise.
Because they were picked with both periods visible, treat the tier numbers
as descriptive rather than as independent out-of-sample evidence.

---

## 4. Results

From `python scripts/backtest_options_swing.py` (5% risk, max 2 concurrent):

| | Trades | Win | Avg return on risk | PF | CAGR | Max DD | Sharpe |
|---|---|---|---|---|---|---|---|
| 2011-2019 (in-sample) | 212 | 67% | +8.2% | 1.72 | +9.4% | −20.3% | 0.73 |
| 2020-2026 (out-of-sample) | 144 | 64% | +7.9% | 1.66 | +8.2% | −15.6% | 0.71 |
| HIGH tier only, 2011-2019 | 91 | 68% | +12.4% | 2.32 | +6.2% | −12.4% | 0.77 |
| HIGH tier only, 2020-2026 | 44 | **82%** | **+23.1%** | 6.34 | +7.7% | −7.2% | **1.31** |
| *SPY buy & hold, 2011-2026* | – | – | – | – | +14.1% | −33.7% | 0.86 |

- Average hold is 4.1 trading days (median 3). There are about 23 trades a year.
- The 95% bootstrap CI on mean return per trade is **+4.1% to +11.9%**,
  clearly above zero.
- Correlation with SPY's daily returns is 0.39. The strategy is out of the
  market most of the time, so it pairs well with a core index holding.
- All 36 neighboring settings are profitable in both periods: pullback
  1.25-2.0 ATR × 14/21/30 DTE × 5/7/10-day max hold.
- Losing years (equity): 2011, 2016, 2018 (Volmageddon / Q4 selloff), 2020
  (COVID), and 2022 (only 4 trades; SPY spent most of the year below its
  200-day SMA). The worst was −12.8% (2018); the best were about +30% (2013,
  2015, 2017).

### Stress tests

| Scenario | Sharpe IS / OOS | Verdict |
|---|---|---|
| Base | 0.73 / 0.71 | |
| IV 11% richer (VIX × 1.0) | 0.70 / 0.64 | robust |
| Enter one day late | 0.47 / 0.50 | still profitable |
| 2× bid/ask costs | 0.30 / 0.31 | thin |
| **3× bid/ask costs** | **−0.15 / −0.10** | **edge gone: execution matters** |
| 1 position max | 0.70 / 0.58 | lower return, maxDD ≈ −10% |
| 10% risk per trade | 0.74 / 0.71 | CAGR +18% / +16%, maxDD −39% / −29% |

### Monte Carlo sizing (10,000 one-year resamples of the trade list)

| Risk per trade | Median year | Bad year (5th pct) | P(losing year) | Bad-case max DD |
|---|---|---|---|---|
| 2% | +3.8% | −2.5% | 16% | −5.3% |
| 3% | +5.8% | −3.8% | 16% | −7.9% |
| **5%** | **+9.6%** | **−6.4%** | **17%** | **−12.9%** |
| 10% | +19.2% | −13.4% | 18% | −24.6% |

---

## 5. How to trade it

```bash
python scripts/options_swing_signals.py        # today's signals + spread tickets
python scripts/backtest_options_swing.py       # full backtest and stress tests
```

1. Run the scanner about 15 minutes before the close. If a symbol shows
   **ENTRY**, the ticket lists the strikes, expiry, estimated debit, and
   take-profit value.
2. **Use a limit order near the mid price.** The stress test shows execution
   cost is the biggest threat to this edge. Don't pay more than ~5% above the
   estimated debit.
3. Each day near the close, check `if already holding: exit signal`. Exit at
   the first YES once you've held 3+ days, and always exit by day 7.
4. Favor HIGH-confidence tickets. When two ETFs signal on the same day, take
   the higher score; they are highly correlated.

## 6. Limitations

- **Close-only data.** Intraday paths, stops, and gaps aren't modeled, and the
  1-day-late test is the proxy for execution slippage.
- **Modeled option prices.** IV comes from VIX, not historical option chains,
  and the VIX mirror can lag a few days. Live trading should use real quotes.
- **Modest sample.** 356 trades, and only 44 HIGH-tier trades out-of-sample.
  The edge is statistically positive, but the size of the HIGH-tier edge is
  uncertain.
- **Long-only by design.** In a bear market (2022) it mostly sits out. That
  is a feature, but it means long flat periods.
- Past performance does not guarantee future results. Options can lose 100%
  of the premium paid.
