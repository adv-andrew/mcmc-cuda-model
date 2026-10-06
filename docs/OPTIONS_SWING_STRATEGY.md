# Options Swing Strategy: research report

**Goal:** a high-confidence options strategy with a normal hold of about 3+ days.

**Result:** buy a **~0.80-delta in-the-money call on SPY / QQQ / IWM when an
index ETF pulls back sharply inside a long-term uptrend**, and hold until it
recovers (min 3, max 10 trading days). Buying the ETF's shares instead,
with the same entries and exits, was the most reliable version tested
(section 6).

**Recommended setup: portfolio mode (section 7).** Keep 85% of the account
in SPY while SPY's month-end close is above its 10-month average (otherwise
T-bills), and run the dip calls on top at 6% of the total account per trade.
In a single-account backtest from 1996 to 2026 this returned **13.5% a year
with a −27% max drawdown**, versus 10.4% and −55% for buying and holding SPY.
Returns were about 13-14% in each era (1996-2010, 2011-2019, 2020-2026), and
10.7% with harsh option pricing.

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
  1991-2009 and about 6% on 2011-2026, versus 14% for buying and holding SPY
  over 2011-2026. Paper trade before risking money.
- **The robustness round (section 6)** made the strategy more reliable, not
  more profitable. Drawdowns are smaller, there are far fewer −75% trades,
  and recent-period results are better. Widening the universe to 13 more
  ETFs failed for options.

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

**Position:** buy one **~0.80Δ (in-the-money) call**, with the nearest
Friday expiry at least 30 days out. The premium is the maximum loss.
**Alternative:** buy about 1/3 of the account in the ETF's shares, with the
same exits (section 6).
(`structure_kind: call_debit_spread` switches back to the original
0.55Δ/0.30Δ, 21-DTE debit spread. That version had better 2011-2026 numbers
but failed the bias audit's skew test.)

**Exit (first that applies, checked at the close):**
1. Option up +60% (rarely hit; mainly a safety valve).
2. **Close above the 5-day SMA, once held ≥ 3 trading days** (the normal
   exit, ~90% of trades).
3. 10 trading days held.

**Sizing:** 6% of equity per position (the premium), at most 2 open at once.

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

From `python scripts/backtest_options_swing.py` (max 2 concurrent):

| | Trades | Win | Avg return on risk | PF | CAGR | Max DD | Sharpe |
|---|---|---|---|---|---|---|---|
| **Default: 0.80Δ call, hold ≤10, 6% risk**, 2011-2019 | 209 | 66% | +4.8% | 1.73 | +6.5% | −16.2% | 0.67 |
| **Default**, 2020-2026 | 143 | 61% | +4.0% | 1.60 | +4.9% | **−11.2%** | **0.57** |
| Default, HIGH tier only, 2020-2026 | 44 | 80% | +10.5% | 4.25 | +4.1% | −6.8% | 1.06 |
| *Previous: 0.70Δ call, hold ≤7, 5% risk*, 2011-2019 | 212 | 64% | +6.3% | 1.74 | +7.3% | −18.5% | 0.72 |
| *Previous*, 2020-2026 | 144 | 60% | +4.5% | 1.47 | +4.5% | −13.2% | 0.52 |
| *Previous*, 1991-2009 untouched (S&P 500) | 164 | 62% | +5.8% | – | +2.5% | −8.3% | 0.49 |
| *Original debit spread, 2011-2026* | 356 | 65% | +8.1% | 1.69 | +8.8% | −20.3% | 0.72 |
| *SPY buy & hold, 2011-2026* | – | – | – | – | +14.1% | −33.7% | 0.86 |

- Average hold is 4.2 trading days (median 3), with about 22 trades a year.
- Only 6 of 352 trades hit the time exit (avg −76%), down from 25 with a
  7-day limit. The extra days let most dips finish recovering.
- The 95% bootstrap CI on mean return per trade is +2.1% to +6.8%.
- Correlation with SPY's daily returns is 0.41.
- All 36 neighboring settings are profitable in both periods.

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

## 6. Robustness round: more reliable, not more profitable

Results come from `scripts/research_robustness.py` and
`scripts/research_shares_vs_options.py`. The protocol was written into the
script before it ran. Only the option structure, max hold, universe and
vehicle were open to change; signal and exits were not.

### 6a. Structure, chosen by worst case

Sixteen single-call variants (0.60-0.90Δ × 30-90 DTE × hold ≤7 / ≤10) plus
the debit spread were scored on SPY/QQQ/IWM 2011-2019. The score was the
**worst** per-trade Sharpe across base pricing, +10% entry IV, steep call
skew and 1.5× bid/ask. The winner was **0.80Δ, 30 DTE, hold ≤10** (worst case
0.125 vs 0.109 for the previous 0.70Δ default).

Validation, per trade (avg / t-stat):

| Window | 0.80Δ base | 0.80Δ entry IV +10% | 0.70Δ base | 0.70Δ entry IV +10% |
|---|---|---|---|---|
| 2020-2026 (held out) | +6.1% / 2.4 | **+3.4% / 1.4** | +6.6% / 2.0 | +2.0% / 0.7 |
| Pre-2011 (SPY 1996-, QQQ 2000-, IWM 2001-) | +5.7% / 2.3 | **+3.4% / 1.5** | +7.8% / 2.3 | +3.0% / 0.9 |

A deeper-ITM call has less extrinsic premium per unit of delta, so it pays
less for implied vol. That was the weak point the bias audit found.

### 6b. Wider universe: the edge is real, but options can't capture it

The same rules were run on **13 ETFs never used before** (DIA, MDY, IJR,
RSP, 9 sector SPDRs, EFA, EEM), 1999-2026, from a second data mirror:

| Group | Underlying, same days | 0.80Δ call (2% bid/ask tier) |
|---|---|---|
| US broad (4) | 70% win, +0.52%, **t = 4.1** | +0.8%, t = 0.6 |
| Sectors (9) | 69% win, +0.44%, **t = 5.0** | −0.9%, t = −1.0 |
| International (2) | 64% win, +0.34%, t = 1.9 | −3.0%, t = −1.6 |
| Placebo vs random uptrend days (all 13) | p < 0.0002 | |

The dip-timing effect replicates on a universe it was never fitted to.
These ETFs' option markets are wider, though, and the edge doesn't survive
them. Adding 12 of them to the options portfolio at the same total risk
lost money (2000-2026 Sharpe −0.26, max DD −66%). **The options universe
stays SPY/QQQ/IWM.**

### 6c. Walk-forward re-selection (`scripts/research_walk_forward.py`)

Each year from 2003 to 2026, the entry thresholds were re-chosen from 15
combinations (pullback 1.0-2.0 ATR × RSI(2) cap 5/10/15) using only the
previous 8 years, then traded for that year:

| Out of sample, 2003-2026 | Trades | Win | Avg / trade | t |
|---|---|---|---|---|
| Re-picked yearly from past data | 252 | 64% | +4.5% | 2.8 |
| Fixed 1.5 ATR / RSI(2) < 10 | 245 | 61% | +5.4% | 3.6 |

The selection process itself is profitable out of sample, and the yearly
picks stay in the 1.25-1.75 ATR band. The fixed choice ranked top-3 of 15
in 12 of 23 years. Adaptive re-tuning did not beat the fixed thresholds
(difference −2.0% per trade, t = −0.7), so they stay fixed, as
pre-registered.

### 6d. Vehicle: shares vs options, 2000-2026

| Approach | CAGR | Max DD | Sharpe | Sharpe 00-10 / 11-19 / 20-26 | Worst year |
|---|---|---|---|---|---|
| Options, core 3 (0.70Δ, previous default) | +4.8% | −18.5% | 0.52 | 0.35 / 0.72 / 0.52 | −11.8% |
| Options, core 3 (0.80Δ, hold ≤10) | +4.9% | −19.9% | 0.52 | 0.35 / 0.67 / 0.57 | – |
| **Shares, core 3 (3 slots × 33%)** | +3.9% | **−11.8%** | **0.66** | **0.51 / 0.68 / 0.86** | **−7.5%** |
| Shares, broad 16 ETFs (4 × 25%) | +3.7% | −17.5% | 0.46 | 0.45 / 0.55 / 0.38 | −13.0% |
| 50/50 options-core + shares-broad | +4.3% | −17.2% | 0.53 | 0.43 / 0.69 / 0.49 | −12.3% |
| SPY buy & hold | +8.3% | −55.2% | 0.51 | 0.13 / 0.94 / 0.81 | −36.8% |

**Shares on SPY/QQQ/IWM is the most reliable way to trade the signal.** It
had the best Sharpe, was positive in every era, and had the smallest
drawdown and worst year. It costs about 1 bp to trade and has no
implied-vol risk. It still makes money at 10 bps per side with a one-day-late
entry (Sharpe 0.47 at 50% slots). Options earn a little more per year only
through leverage, at lower efficiency and with more model risk.

Why returns are only a few percent a year: the strategy is invested about
14% of the time. **The idle cash should sit in T-bills or a money-market
fund**, which these backtests don't credit. Bigger shares positions don't
help without margin, because cash, not signals, is the binding constraint.

---

## 7. Portfolio mode: put the idle capital to work

On its own, the dip strategy has money in the market only ~14% of the time.
Its return per unit of risk is fine, but most of the account sits idle.
Portfolio mode adds a **trend-following core** for that idle capital, using
a published rule (Faber, 2007). Hold SPY while its last month-end close is
above its 10-month SMA; otherwise hold T-bills. The rule was published in
2007, so 2007-2026 is out of sample for it too.

Single-account simulation (`scripts/research_core_overlay.py`): SPY shares
rebalanced monthly and on signal changes, option premiums paid from cash and
sized off total equity, T-bill interest on idle cash, 5 bps per core trade.

| 1996-2026 | 1996-2010 | 2011-2019 | 2020-2026 | **CAGR** | Max DD | Sharpe (excess) | Worst year |
|---|---|---|---|---|---|---|---|
| **85% core + options (default)** | +13.5% | +13.1% | +13.8% | **+13.5%** | −27.2% | 0.67 | −21.5% |
| Faber 8 / 12 months | | | | +13.3% / +13.7% | −27% | 0.67 / 0.68 | |
| Core weight 80% | | | | +13.2% | −26.4% | 0.67 | |
| Harsh option pricing (IV +11%, call skew .25, 1.5× bid/ask) | +11.3% | +9.4% | +10.8% | +10.7% | −28.9% | 0.54 | |
| Core only, 85% (no options) | +9.8% | +6.9% | +8.9% | +8.8% | −21.9% | 0.61 | −16.7% |
| Options dip strategy alone (with cash interest) | +6.9% | +7.2% | +8.0% | +7.3% | −18.3% | 0.55 | −5.8% |
| SPY buy & hold | +6.6% | +13.1% | +15.2% | +10.4% | −55.2% | 0.49 | −36.8% |

- **The options overlay adds about 4.7 points a year** on top of the core
  (13.5% vs 8.8%) at 6% of the account per trade. The core sidesteps the
  2000-02 and 2008 bear markets, which is where buy-and-hold's −55% came from.
- **The result doesn't hinge on exact settings.** An 8- or 12-month lookback
  and an 80% weight all land at 13.2-13.7%. A 90% core weight drops to 11.8%
  for a mechanical reason: 90% core plus 2 × 6% premium exceeds the account,
  so dip trades get skipped.
- **Daily-rebalanced sleeve blending** (`scripts/research_portfolio.py`)
  overstated the result by ~0.9 points a year (14.4% vs 13.5%). The
  single-account figures are the reference.
- **Where it lags:** in strong, uninterrupted bull markets (2011-2019,
  2020-2026) buy-and-hold beat it. Its edge is in crashes and choppy markets,
  and in a much smaller worst case.
- Twelve percent of the account per trade (70% core) raises CAGR to ~18% but
  the drawdown to −43%. That's a risk-preference choice, not an improvement.

### Where portfolio mode loses money

| Year | Portfolio | Core only | SPY | What happened |
|---|---|---|---|---|
| 2008 | −1.2% | +0.6% | −36.8% | Core out of the market almost all year |
| 2001-2002 | +6.8% / −5.9% | +3.5% / −2.5% | −11.8% / −21.6% | Core mostly out |
| 2011 | −9.7% | −4.6% | +1.9% | Sharp V-shaped correction inside an uptrend |
| 2018 | −11.0% | −3.0% | −4.6% | Same pattern (February and Q4 selloffs) |
| **2022** | **−21.5%** | −16.7% | −18.2% | **Whipsaw**: the trend signal flipped in and out on bear-market rallies, and all 4 dip trades lost |

The worst drawdown, −27.2%, ran from January 2022 to March 2023 and was
recovered by February 2024. The strategy is strongest in slow bear markets
(2001-02, 2008) and weakest in fast, choppy reversals, the textbook
trend-following profile. The best years (2013 +57%, 2024 +42%, 1997 +45%,
2017 +40%) were steady uptrends with many quick dip recoveries.

### Sizing menu (`scripts/research_sizing.py`)

Option risk per trade vs result, 1996-2026, single account. Core weight
shrinks at higher risk so everything fits: core = min(85%, 1 − 2 × risk − 3%).

| Risk / trade | Core | CAGR (base) | Max DD (base) | Sharpe (base) | CAGR (harsh) | Max DD (harsh) | Sharpe (harsh) |
|---|---|---|---|---|---|---|---|
| 3% | 85% | +11.2% | −24.6% | 0.66 | +9.8% | −24.5% | 0.58 |
| 4.5% | 85% | +12.4% | −25.9% | 0.67 | +10.3% | −25.8% | 0.56 |
| **6% (default)** | 85% | **+13.5%** | −27.2% | 0.67 | +10.7% | −28.9% | 0.54 |
| 8% | 81% | +14.8% | −31.4% | 0.67 | +11.0% | −32.9% | 0.52 |
| 10% | 77% | +15.9% | −35.7% | 0.66 | +11.2% | −37.4% | 0.50 |
| 12% | 73% | +17.0% | −39.9% | 0.66 | +11.4% | −42.4% | 0.48 |

With base pricing, risk-adjusted return is the same at every size, so more
risk buys proportionally more return. With harsh pricing, bigger sizes add
almost no return but a lot more drawdown. **6% is the knee**; 4.5% is the
cautious choice if you doubt the option pricing. Full Kelly on the trade
distribution is ~70% of equity per trade (46% if the edge is 40% smaller).
The default is about 1/10 Kelly, because concurrent trades and the core are
correlated in crashes.

### Account size and whole contracts (`scripts/research_account_size.py`)

Real options trade in whole contracts. With integer contracts (buy as many
as fit in 6% of the account, or one if a single contract costs ≤ 1.5× that),
results match fractional sizing for accounts of $50k and up: 2011-start CAGR
13.0-13.4% vs 13.5% fractional. At 2026 prices a 0.80Δ 30-day call costs
about $4,600 (SPY), $5,100 (QQQ) and $2,200 (IWM). So:

| Account | What actually trades |
|---|---|
| ≥ ~$60k | the full SPY/QQQ/IWM rotation |
| ~$25-60k | mostly IWM (SPY/QQQ contracts too large) |
| < ~$25k | almost no options. A $10k account starting in 2020 made 5 option trades and ~9% a year (core only). Use the shares line for dips |

`python scripts/options_swing_signals.py --account 50000` prints the exact
number of contracts (or "skip, use shares") and the core share count. The
paper ledger trades whole contracts by default.

### Paper trading with the backtest engine

`scripts/paper_trade.py` keeps a ledger (`data/paper/ledger.json`, not
committed) driven by the same `CoreOverlayEngine` as the single-account
backtest. Tests confirm the ledger reproduces the backtest's equity curve
exactly, even when it is stopped, saved and resumed. Run `update` after the
close; missed days are replayed. Record what you could really pay for each
option with `fill <id> <price>`. The report compares your real fills with
the model's debit, the one number that decides whether this edge survives
live trading (it can absorb about 2%).

A replay from 2 January 2025 to 5 October 2026 made 40 option trades: 68% won,
averaging +3.6% per trade, and the account rose 23.5% with a −15.9% max
drawdown through the April 2025 tariff shock.

### SPY-only vs three-index core (`scripts/research_core_universe.py`)

This was a single pre-registered test: a core of one-third each SPY/QQQ/IWM,
each with its own 10-month filter. It had to beat the SPY core on excess
Sharpe in all three eras to be adopted.

| Core + options | Sharpe 2001-10 | 2011-19 | 2020-26 | Worst year 2020-26 | CAGR 2001-26 |
|---|---|---|---|---|---|
| SPY core (kept) | **0.55** | 0.71 | 0.65 | −22.6% | +11.9% |
| SPY/QQQ/IWM core | 0.46 | **0.74** | **0.70** | **−14.2%** | +12.2% |

It was better in the QQQ-led 2010s and 2020s but worse through the 2000-02
tech crash: the hindsight pattern the every-era rule is there to catch. The
SPY core stays.

### Whipsaw-resistant core filter (`scripts/research_core_filter.py`)

This was a single pre-registered test: go to T-bills only when SPY is below
its 10-month SMA **and** its 12-month total return is below T-bills
(absolute momentum). It had to win all three eras to be adopted.

| Core + options | Sharpe 1996-10 | 2011-19 | 2020-26 | CAGR 1996-26 | Max DD | Signal changes |
|---|---|---|---|---|---|---|
| 10-month SMA (kept) | 0.64 | 0.72 | **0.65** | +13.5% | **−27.2%** | 44 |
| SMA + 12-month momentum | **0.68** | **0.76** | 0.63 | **+14.8%** | −32.1% | 18 |

This was a near miss. The dual filter trades less, earns about 1.3 points a
year more, and softens 2022 (−18.5% vs −21.5%). But it exits later, so it
rode the 2020 crash deeper, and its worst drawdown is 5 points larger. It
failed one era, so the 10-month SMA stays. If you care more about CAGR than
worst-case drawdown, it's the documented alternative.

**Why the search stops here.** This round ran four pre-registered core and
overlay variants (exits, risk-off asset, core universe, core filter). None
passed every era. Each additional variant raises the chance that one
passes by luck, which is the bias this project has worked to avoid.

### What to expect (`scripts/research_outcomes.py`)

A block bootstrap of the 1996-2026 daily returns (10,000 paths, ~1-month
blocks so crashes stay intact), sampled jointly with SPY:

| | 1 yr: portfolio mode | 1 yr: SPY | 5 yr: portfolio mode | 5 yr: SPY |
|---|---|---|---|---|
| Bad case (5th percentile) | −11.0% | −16.9% | +8.4% | −12.5% |
| Median | +14.0% | +11.6% | +90% (13.7%/yr) | +66% (10.6%/yr) |
| Chance of losing money | 19% | 25% | 3% | 9% |
| Chance of a −30% drawdown | 1% | 6% | 14% | 37% |
| Beats SPY | 57% of paths | | 68% of paths | |

This assumes the next few years resemble 1996-2026 and that the option
pricing model is right. With harsh pricing, subtract about 3 points a year
(section 7).

### Treasuries instead of T-bills when the core is out (`scripts/research_risk_off.py`)

This was a single pre-registered test: adopt IEF (7-10 year Treasuries) as
the risk-off asset only if it beat T-bills on excess Sharpe in all three
eras.

| Core + options | Sharpe 2003-10 | 2011-19 | 2020-26 | Worst year 2020-26 | CAGR 2003-26 |
|---|---|---|---|---|---|
| T-bills when out (kept) | 0.65 | 0.71 | **0.65** | **−22.6%** | +12.9% |
| IEF when out | **0.76** | **0.76** | 0.64 | −28.4% | +14.0% |

IEF helped through 2019 (bonds rallied in 2008) but failed in 2022, when
stocks and bonds fell together. It doesn't pass, so T-bills stay. It's a
reasonable alternative if you expect bonds to hedge stocks again.

**How to run it:** the scanner prints the core instruction at the top.
Re-check it at the last close of each month; the "month-end re-check" line
previews the rule using today's close.

---

## 8. Confirmed on real option quotes

Every result above used *modelled* option prices. `scripts/research_real_quotes.py`
re-prices the trades portfolio mode actually made from 2008 to 2025 with
**real end-of-day bids and asks** for SPY, QQQ and IWM: 4,514 trading days and
about 25 million SPY quotes, from the
[lambdaclass/options_backtester](https://github.com/lambdaclass/options_backtester)
`data-v1` release, SHA-256 verified. The protocol and verdict rule were fixed
before any real price was examined.

For each trade, the contract is the call nearest 0.80Δ at the first expiry
at least 30 days out, using the real chain's quoted deltas. It is bought and
sold on the model's entry and exit days, with $0.65 per contract per side.

| Same 347 trades (17 had no quotes that day) | Win | Avg return / trade | t |
|---|---|---|---|
| Model (Black-Scholes, IV from VIX) | 63% | +4.15% | 3.33 |
| Real quotes, optimistic fills (mid) | 67% | +5.27% | 3.94 |
| **Real quotes, realistic fills** (limit 25% into the spread) | **65%** | **+4.11%** | **3.10** |
| Real quotes, pessimistic fills (pay ask, sell bid) | 63% | +2.98% | 2.26 |

- **Model and reality agree trade by trade** (correlation 0.97). Real
  contracts cost about 3.8% *less* than the model priced them (median), so
  the model was slightly conservative. The real median bid/ask spread was
  1.8% of mid, close to the 1% half-spread assumed.
- **By period (realistic fills):** 2008-2010 −1.4% (23 trades), 2011-2019
  +4.4% (204), 2020-2025 +4.6% (120).
- **Worst trades** were the known shocks: Oct 2018 −93%, Aug 2011 −88%,
  Feb 2018 −86%. This is why each position is capped at 6% of the account.

**Portfolio mode, 2008-2025, with real-quote option P&L:**

| | CAGR | Max DD | Sharpe (excess) | Worst year |
|---|---|---|---|---|
| Core only (no options) | +8.2% | −21.9% | 0.67 | −16.7% |
| With model option prices | +12.8% | −27.2% | 0.69 | −21.5% |
| **With real quotes, realistic fills** | **+12.9%** | −30.1% | 0.68 | −21.3% |
| With real quotes, pessimistic fills | +12.3% | −34.2% | 0.62 | −23.2% |
| SPY buy & hold | +11.0% | −51.9% | 0.56 | −36.2% |

**Pre-registered verdict: confirmed.** Realistic-fill returns are positive
(t = 3.1), and the options add about 4.7 points a year over the core alone.
The real-quote drawdown is deeper than the model's (−30% vs −27%). The
portfolio figures swap each trade's P&L without re-deriving later position
sizes, a small approximation.

To reproduce (about 1.3 GB, saved to the git-ignored `data/cache/options/`):

```bash
mkdir -p data/cache/options && cd data/cache/options
for s in SPY QQQ IWM; do for f in options underlying; do
  curl -LO "https://github.com/lambdaclass/options_backtester/releases/download/data-v1/${s}_${f}.parquet"
done; done
cd - && python scripts/research_real_quotes.py
```

---

## 9. How to trade it

```bash
python scripts/options_swing_signals.py        # today's signals + order tickets
python scripts/backtest_options_swing.py       # full backtest and stress tests
python scripts/research_bias_checks.py         # untouched-data, placebo and pricing audits
python scripts/research_robustness.py          # worst-case structure choice, fresh-ETF validation
python scripts/research_shares_vs_options.py   # shares vs options vs hybrid, 2000-2026
python scripts/research_core_overlay.py        # portfolio mode, single-account validation
python scripts/research_exits.py               # pre-registered exit study (no change adopted)
python scripts/research_sizing.py              # risk per trade vs return and drawdown
python scripts/paper_trade.py update           # paper-trade portfolio mode (same engine)
```

0. **Portfolio mode:** keep 85% of the account in SPY when the scanner says
   `PORTFOLIO CORE: IN`, and in T-bills / a money-market fund when it says
   `OUT`. Rebalance on the last trading day of each month.
1. Run the scanner about 15 minutes before the close. If a symbol shows
   **ENTRY**, the ticket lists the strike, expiry, estimated debit, and
   take-profit value.
2. **Use a limit order, and skip the trade if you can't fill within ~2% of
   the estimated debit.** Overpaying for implied vol is the main way this
   edge disappears (section 5b).
3. Each day near the close, check `if already holding: exit signal`. Exit at
   the first YES once you've held 3+ days, and always exit by day 10.
4. Favor HIGH-confidence tickets. When two ETFs signal on the same day, take
   the higher score; they are highly correlated.
5. **If reliability matters more than leverage, use the shares line on the
   ticket** (about 1/3 of the account per ETF, same exits) instead of the
   call, and keep idle cash in T-bills.

## 10. Limitations

- **Close-only data.** Intraday paths, stops, and gaps aren't modeled, and the
  1-day-late test is the proxy for execution slippage.
- **Modeled option prices** (except section 8). IV comes from VIX, not historical option chains,
  and the VIX mirror can lag a few days. The bias audit shows the result
  depends on entry IV being close to the model's estimate, which only real
  quotes can settle. Paper trading and recording your actual fills against
  the scanner's estimated debit is the most valuable next test.
- **Modest sample.** About 350 trades, and only 44 HIGH-tier trades out-of-sample.
  The edge is statistically positive, but the size of the HIGH-tier edge is
  uncertain.
- **Long-only by design.** In a bear market (2022) it mostly sits out. That
  is a feature, but it means long flat periods.
- Past performance does not guarantee future results. Options can lose 100%
  of the premium paid.
