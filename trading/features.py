"""
Vectorized technical features for swing-trading research.

Every feature at row ``t`` uses only information available at the close of
``t``. Higher-timeframe (weekly / monthly) features are computed on
*completed* bars and then forward-filled onto the daily index, so a daily
row never sees the close of a week or month that has not finished yet.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Primitive indicators
# ----------------------------------------------------------------------

def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0).where(avg_gain.notna())


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    return np.log(close).diff().rolling(window).std() * math.sqrt(TRADING_DAYS)


def ibs(df: pd.DataFrame) -> pd.Series:
    """Internal Bar Strength: where the close sits inside the day's range (0..1)."""
    rng = (df["High"] - df["Low"]).replace(0.0, np.nan)
    return ((df["Close"] - df["Low"]) / rng).fillna(0.5)


def streak(close: pd.Series) -> pd.Series:
    """Consecutive up (+n) or down (-n) closes ending at each bar."""
    sign = np.sign(close.diff()).fillna(0.0).to_numpy()
    out = np.zeros(len(sign))
    for i in range(1, len(sign)):
        if sign[i] == 0:
            out[i] = 0
        elif sign[i] == np.sign(out[i - 1]):
            out[i] = out[i - 1] + sign[i]
        else:
            out[i] = sign[i]
    return pd.Series(out, index=close.index)


def higher_tf_close(close: pd.Series, rule: str) -> pd.Series:
    """Completed higher-timeframe closes forward-filled to the daily index.

    ``rule`` is a pandas offset alias, e.g. ``"W-FRI"`` or ``"ME"``. The value
    on day ``t`` is the close of the last *completed* period strictly before
    the period that contains ``t``.
    """
    htf = close.resample(rule).last().dropna()
    # Label each completed bar with the first daily date *after* it ends.
    shifted = htf.shift(1)
    period_of_day = close.index.to_period(_period_freq(rule))
    period_end = htf.index.to_period(_period_freq(rule))
    mapping = pd.Series(shifted.to_numpy(), index=period_end)
    return pd.Series(mapping.reindex(period_of_day).to_numpy(), index=close.index)


def higher_tf_series(close: pd.Series, rule: str, func) -> pd.Series:
    """Apply ``func`` to completed higher-timeframe closes; map back to daily."""
    htf = close.resample(rule).last().dropna()
    values = func(htf).shift(1)  # only completed bars
    period_of_day = close.index.to_period(_period_freq(rule))
    mapping = pd.Series(values.to_numpy(), index=htf.index.to_period(_period_freq(rule)))
    return pd.Series(mapping.reindex(period_of_day).to_numpy(), index=close.index)


def _period_freq(rule: str) -> str:
    if rule.startswith("W"):
        return rule
    if rule in ("ME", "M", "MS"):
        return "M"
    raise ValueError(f"Unsupported rule {rule}")


# ----------------------------------------------------------------------
# Feature frame
# ----------------------------------------------------------------------

def build_features(df: pd.DataFrame, vix: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute the full research feature set for one daily OHLCV frame."""
    c = df["Close"]
    f = pd.DataFrame(index=df.index)
    f["close"] = c
    f["open"] = df["Open"]
    f["high"] = df["High"]
    f["low"] = df["Low"]
    f["ret1"] = c.pct_change()

    # --- trend / moving averages (daily) --------------------------------
    for n in (5, 10, 20, 50, 100, 200):
        f[f"sma{n}"] = c.rolling(n).mean()
    f["ema8"] = c.ewm(span=8, adjust=False).mean()
    f["ema21"] = c.ewm(span=21, adjust=False).mean()
    f["above200"] = c > f["sma200"]
    f["above50"] = c > f["sma50"]
    f["sma50_slope"] = f["sma50"].pct_change(10)

    # --- momentum at several horizons ("timeframe grouping") -------------
    for n in (2, 3, 5, 10, 21, 63, 126, 252):
        f[f"mom{n}"] = c.pct_change(n)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1.0

    # --- oscillators ------------------------------------------------------
    f["rsi2"] = rsi(c, 2)
    f["rsi3"] = rsi(c, 3)
    f["rsi5"] = rsi(c, 5)
    f["rsi14"] = rsi(c, 14)
    f["crsi2_2d"] = f["rsi2"] + f["rsi2"].shift(1)  # cumulative RSI(2), 2 days
    f["ibs"] = ibs(df)
    f["streak"] = streak(c)
    std20 = c.rolling(20).std()
    f["bb_z"] = (c - f["sma20"]) / std20
    f["low_n10"] = c <= c.rolling(10).min()
    f["low_n7"] = c <= c.rolling(7).min()
    f["high_n7"] = c >= c.rolling(7).max()
    f["high_n20"] = c >= c.rolling(20).max()
    f["high_n252"] = c >= c.rolling(252).max()
    f["low_n20"] = c <= c.rolling(20).min()
    hh = df["High"].rolling(14).max()
    ll = df["Low"].rolling(14).min()
    f["willr14"] = -100.0 * (hh - c) / (hh - ll).replace(0.0, np.nan)

    # --- volatility ---------------------------------------------------------
    f["rv5"] = realized_vol(c, 5)
    f["rv10"] = realized_vol(c, 10)
    f["rv20"] = realized_vol(c, 20)
    f["rv60"] = realized_vol(c, 60)
    f["atr14"] = atr(df, 14)
    f["atr_pct"] = f["atr14"] / c
    rng = df["High"] - df["Low"]
    f["nr7"] = rng <= rng.rolling(7).min()
    f["inside"] = (df["High"] <= df["High"].shift(1)) & (df["Low"] >= df["Low"].shift(1))
    f["gap"] = df["Open"] / c.shift(1) - 1.0
    # Distance below the 5-day high in ATRs: a volatility-normalised pullback
    f["pullback_atr"] = (c.rolling(5).max() - c) / f["atr14"]

    # --- higher timeframes (completed bars only) --------------------------
    f["w_close"] = higher_tf_close(c, "W-FRI")
    f["w_sma10"] = higher_tf_series(c, "W-FRI", lambda s: s.rolling(10).mean())
    f["w_sma30"] = higher_tf_series(c, "W-FRI", lambda s: s.rolling(30).mean())
    f["w_rsi14"] = higher_tf_series(c, "W-FRI", lambda s: rsi(s, 14))
    f["w_mom4"] = higher_tf_series(c, "W-FRI", lambda s: s.pct_change(4))
    f["weekly_up"] = f["w_close"] > f["w_sma10"]
    f["m_close"] = higher_tf_close(c, "ME")
    f["m_sma10"] = higher_tf_series(c, "ME", lambda s: s.rolling(10).mean())
    f["monthly_up"] = f["m_close"] > f["m_sma10"]

    # Multi-timeframe alignment score in [-4, 4]: sign of trend on
    # 5d / 21d / weekly / monthly horizons.
    f["mtf_score"] = (
        np.sign(f["mom5"]).fillna(0)
        + np.sign(f["mom21"]).fillna(0)
        + np.where(f["weekly_up"], 1, -1)
        + np.where(f["monthly_up"], 1, -1)
    )

    # --- calendar -----------------------------------------------------------
    idx = df.index
    f["dow"] = idx.dayofweek
    month = idx.to_period("M")
    pos_in_month = pd.Series(1, index=idx).groupby(month).cumsum()
    days_in_month = pd.Series(1, index=idx).groupby(month).transform("sum")
    f["tdom"] = pos_in_month.to_numpy()                      # trading day of month
    f["tdom_rev"] = (days_in_month - pos_in_month + 1).to_numpy()  # 1 = last day

    # --- VIX context -------------------------------------------------------
    if vix is not None:
        v = vix["Close"].reindex(idx).ffill()
        f["vix"] = v
        f["vix_sma10"] = v.rolling(10).mean()
        f["vix_ratio10"] = v / f["vix_sma10"]
        f["vix_pct252"] = v.rolling(252).rank(pct=True)
        f["vix_chg1"] = v.pct_change()
        f["vix_chg5"] = v.pct_change(5)
        # Implied minus realized: positive = options rich vs recent movement
        f["vrp"] = v / 100.0 - f["rv10"]

    return f


def forward_returns(close: pd.Series, horizons=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    """Close-to-close forward returns (row t = return from close t to close t+h)."""
    return pd.DataFrame({f"fwd{h}": close.shift(-h) / close - 1.0 for h in horizons})
