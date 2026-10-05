# Options Swing Strategy: research report

**Goal:** a high-confidence options strategy with a normal hold of about 3+ days.

**Result:** buy a **~0.70-delta in-the-money call on SPY / QQQ / IWM when an
index ETF pulls back sharply inside a long-term uptrend**, and hold until it
recovers (min 3, max 7 trading days).

**Bottom line after the bias audit (section 5):**
- **The timing edge is real.** On untouched 1991-2009 S&P 500 data, the
  same entry/exit days returned +0.62% on the underlying with a 72% win rate
  (t = 5.7), matching 2011-2026 (+0.57%, 72%, t = 5.5). It beats random
  uptrend days at p ≈ 0.001 in both eras.
- **The options profit is real under base assumptions but thin.** It was
  +5.6% per trade on 1991-2009 (t = 2.6) and +5.0% on 2011-2026 (t = 3.5).
  But overpaying about 1.5 vol points at entry would erase most of it, and
  the deflated Sharpe ratio puts the probability that the portfolio Sharpe
  isn't selection noise at only 11-51%.
- **Expect a few percent a year, not a money machine.** That's 2.5% CAGR on
  1991-2009 and 6.1% on 2011-2026 at 5% risk per trade, versus 14% for
  buying and holding SPY over 2011-2026. Paper trade before risking money.

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
| *(added in the bias audit)* ITM call 0.70Δ, 30 DTE | see section 5 | 0.72 / 0.52 (strategy rules) |
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

**Position:** buy one **~0.70Δ (in-the-money) call**, with the nearest
Friday expiry at least 30 days out. The premium is the maximum loss.
(`structure_kind: call_debit_spread` switches back to the original
0.55Δ/0.30Δ, 21-DTE debit spread. That version had better 2011-2026 numbers
but failed the bias audit's skew test.)

**Exit (first that applies, checked at the close):**
1. Option up +60% (rarely hit; mainly a safety valve).
2. **Close above the 5-day SMA, once held ≥ 3 trading days** (the normal
   exit, ~90% of trades).
3. 7 trading days held.

**Sizing:** 5% of equity per position (the premium), at most 2 open at once.

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

## 4. Results (default ITM call)

From `python scripts/backtest_options_swing.py` (5% risk, max 2 concurrent):

| | Trades | Win | Avg return on risk | PF | CAGR | Max DD | Sharpe |
|---|---|---|---|---|---|---|---|
| 2011-2019 (in-sample) | 212 | 64% | +6.3% | 1.74 | +7.3% | −18.5% | 0.72 |
| 2020-2026 (out-of-sample) | 144 | 60% | +4.5% | 1.47 | +4.5% | −13.2% | 0.52 |
| HIGH tier only, 2011-2019 | 91 | 66% | +10.1% | 2.42 | +5.0% | −12.0% | 0.79 |
| HIGH tier only, 2020-2026 | 44 | 77% | +14.1% | 4.85 | +4.6% | −5.9% | 1.13 |
| **1991-2009 untouched (S&P 500)** | 164 | 62% | +5.8% | – | **+2.5%** | −8.3% | **0.49** |
| *SPY buy & hold, 2011-2026* | – | – | – | – | +14.1% | −33.7% | 0.86 |
| *original debit spread, 2011-2026* | 356 | 65% | +8.1% | 1.69 | +8.8% | −20.3% | 0.72 |

- Average hold is 4.0 trading days (median 3). There are about 23 trades a
  year on three ETFs, and about 9 a year on one index (1991-2009).
- The 95% bootstrap CI on mean return per trade is +2.4% to +8.7%.
- Correlation with SPY's daily returns is 0.38.
- All 36 neighboring settings are profitable in both periods.
- Losing years: 2011, 2016, 2018 (−11.8%), 2020 and 2022 in 2011-2026;
  1996, 2000 and 2004 in 1991-2009.

### Stress tests (portfolio, 2011-2019 / 2020-2026 Sharpe)

| Scenario | ITM call (default) | Debit spread (original) |
|---|---|---|
| Base | 0.72 / 0.52 | 0.73 / 0.71 |
| Steeper put skew, flatter call skew | 0.69 / 0.50 | 1.05 / 1.03 |
| IV 11% richer (entry *and* exit) | 0.53 / 0.34 | 0.70 / 0.64 |
| Enter one day late | 0.51 / 0.38 | 0.47 / 0.50 |
| 2× bid/ask | 0.49 / 0.29 | 0.30 / 0.31 |
| 3× bid/ask | 0.26 / 0.04 | −0.15 / −0.10 |
| Weekly trend required | 0.61 / 0.97 | 0.57 / 1.15 |

### Monte Carlo sizing (10,000 one-year resamples of the trade list)

| Risk per trade | Median year | Bad year (5th pct) | P(losing year) | Bad-case max DD |
|---|---|---|---|---|
| 2% | +2.6% | −2.3% | 19% | −4.3% |
| 3% | +3.9% | −3.4% | 19% | −6.3% |
| **5%** | **+6.6%** | **−5.7%** | **19%** | **−10.4%** |
| 10% | +13.0% | −11.8% | 20% | −20.1% |

---

## 5. Bias audit

`python scripts/research_bias_checks.py` attacks the result from five
directions. The rules were frozen before running it.

### 5a. Untouched history and placebo tests

Per-trade results for the strategy's actual entries:

| | 2011-2026 (design) | **1991-2009 (never used)** |
|---|---|---|
| Underlying only, no options | 72% win, +0.57%, **t = 5.5** | 72% win, +0.62%, **t = 5.7** |
| ITM call | 61% win, +5.0%, t = 3.5 | 61% win, +5.6%, t = 2.6 |
| Random days *above* SMA200, ITM call | −0.2% | −0.5% |
| Placebo p-value (signal vs 5,000 random samples), ITM call / underlying | p < 0.0002 / 0.0002 | p = 0.005 / 0.001 |

The dip-timing effect replicates almost exactly on 19 years of data that
played no part in designing the rules, including the 2000-02 and 2008 bear
markets. It is not a product of the 2010s bull market or of the option
model: the underlying alone shows it.

By phase, the ITM call made +5.8% per trade in 1991-99, +6.9% in 2003-07 and
+11.0% in 2008-09, and lost −7.6% in 2000-02 (only 9 trades). In 2011-26 it
was strong in 2011-15 and 2023-26 and roughly flat in 2016-19 (+0.8%) and
2020-22 (+2.3%).

### 5b. Option-pricing sensitivity (per trade, avg / t-stat)

| Scenario | Debit spread 2011-26 | Debit spread 1991-09 | ITM call 2011-26 | ITM call 1991-09 |
|---|---|---|---|---|
| Base | +7.6% / 4.2 | +4.9% / 1.8 | +5.0% / 3.5 | +5.6% / 2.6 |
| Call skew 0.15 | +5.7% / 3.2 | +3.4% / 1.3 | +4.9% / 3.4 | +5.6% / 2.6 |
| Call skew 0.25 (steep) | +2.0% / 1.2 | +0.4% / 0.2 | +4.8% / 3.4 | +5.5% / 2.5 |
| Pay +10% IV at entry only | +5.9% / 3.2 | +3.0% / 1.1 | **+0.4% / 0.3** | **+1.2% / 0.6** |
| 1.5× bid/ask | +5.1% / 2.9 | +1.7% / 0.6 | +3.9% / 2.8 | +4.5% / 2.1 |
| Enter and exit 1 day late | +7.9% / 3.9 | +5.7% / 1.7 | +5.5% / 3.4 | +8.0% / 2.9 |
| All of the above at once | −1.7% | −4.4% | −0.1% | +1.6% |

Each structure has its own weak point. **The debit spread depends on the
call-wing skew**, the least certain part of the model, because its short
0.30Δ call is priced there. That's why it was dropped as the default. **The
ITM call depends on entry IV.** Paying ~1.5 vol points more than the model
at entry (about 4-5% of the premium) removes most of the edge. The practical
defense is the scanner's fill rule: **skip the trade if you can't fill within
~2% of the estimated debit.**

The ITM call (0.70Δ) was chosen *after* seeing the 1991-2009 debit-spread
result, so its 1991-2009 numbers are not fully independent. The reason for
choosing it was set beforehand: no short leg means no call-skew exposure.

### 5c. Multiple testing

About 550 configurations were evaluated during research. The deflated Sharpe
ratio (Bailey & López de Prado) asks whether the best result beats what
that much searching would produce by luck:

| Assumed independent trials | 30 | 100 | 552 |
|---|---|---|---|
| P(portfolio Sharpe 0.64 reflects a real edge) | 51% | 29% | 11% |

The configurations are heavily correlated, so the truth lies toward the
low-trial end. Even so, **the options portfolio's Sharpe ratio on its own is
not strong evidence.** The stronger evidence is the frozen-rule replication
on 1991-2009 and the placebo tests, which multiple testing doesn't explain.

### 5d. Verdict

| Claim | Status |
|---|---|
| Short dips in uptrending index ETFs tend to bounce within 3-7 days | **Confirmed** (two eras, placebo p ≈ 0.001, no option model needed) |
| The default ITM-call version makes money after realistic costs | **Likely, but thin.** Positive in both eras under base pricing; ~breakeven if entry IV is ~10% richer than modeled |
| "+8-9% a year" from the original debit spread | **Overstated.** Depends on the 2010s and on the skew assumption |
| The HIGH-confidence tier is much better | **Unproven.** 44 out-of-sample trades, and the components were chosen with both periods visible |

---

## 6. How to trade it

```bash
python scripts/options_swing_signals.py        # today's signals + order tickets
python scripts/backtest_options_swing.py       # full backtest and stress tests
python scripts/research_bias_checks.py         # untouched-data, placebo and pricing audits
```

1. Run the scanner about 15 minutes before the close. If a symbol shows
   **ENTRY**, the ticket lists the strike, expiry, estimated debit, and
   take-profit value.
2. **Use a limit order, and skip the trade if you can't fill within ~2% of
   the estimated debit.** Overpaying for implied vol is the main way this
   edge disappears (section 5b).
3. Each day near the close, check `if already holding: exit signal`. Exit at
   the first YES once you've held 3+ days, and always exit by day 7.
4. Favor HIGH-confidence tickets. When two ETFs signal on the same day, take
   the higher score; they are highly correlated.

## 7. Limitations

- **Close-only data.** Intraday paths, stops, and gaps aren't modeled, and the
  1-day-late test is the proxy for execution slippage.
- **Modeled option prices.** IV comes from VIX, not historical option chains,
  and the VIX mirror can lag a few days. The bias audit shows the result
  depends on entry IV being close to the model's estimate, which only real
  quotes can settle. Paper trading and recording your actual fills against
  the scanner's estimated debit is the most valuable next test.
- **Modest sample.** 356 trades, and only 44 HIGH-tier trades out-of-sample.
  The edge is statistically positive, but the size of the HIGH-tier edge is
  uncertain.
- **Long-only by design.** In a bear market (2022) it mostly sits out. That
  is a feature, but it means long flat periods.
- Past performance does not guarantee future results. Options can lose 100%
  of the premium paid.
