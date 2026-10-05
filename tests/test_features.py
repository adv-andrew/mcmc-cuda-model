"""Tests for trading/features.py - especially absence of lookahead."""

import numpy as np
import pandas as pd
import pytest

from trading.features import (
    build_features,
    forward_returns,
    higher_tf_close,
    ibs,
    rsi,
    streak,
)


def _ohlc(n=600, seed=0, start="2019-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    idx = pd.bdate_range(start, periods=n)
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    opn = np.clip(close * (1 + rng.normal(0, 0.003, n)), low, high)
    return pd.DataFrame({"Open": opn, "High": high, "Low": low, "Close": close,
                         "Volume": 1e6}, index=idx)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 50, dtype=float))
    down = pd.Series(np.arange(50, 1, -1, dtype=float))
    assert rsi(up, 2).iloc[-1] == pytest.approx(100.0)
    assert rsi(down, 2).iloc[-1] == pytest.approx(0.0)
    r = rsi(_ohlc().Close, 14).dropna()
    assert r.between(0, 100).all()


def test_ibs():
    df = pd.DataFrame({"High": [10.0, 10.0], "Low": [0.0, 0.0], "Close": [0.0, 10.0]})
    assert ibs(df).tolist() == [0.0, 1.0]


def test_streak():
    s = streak(pd.Series([1.0, 2, 3, 2, 1, 0, 1]))
    assert s.tolist() == [0, 1, 2, -1, -2, -3, 1]


def test_weekly_close_uses_only_completed_weeks():
    df = _ohlc()
    w = higher_tf_close(df.Close, "W-FRI")
    # For every Monday-Thursday, the value equals the previous Friday's close
    fridays = df.Close[df.index.dayofweek == 4]
    for day in df.index[50:120]:
        if day.dayofweek < 4:
            prev_fri = fridays[fridays.index < day].index[-1]
            assert w.loc[day] == pytest.approx(df.Close.loc[prev_fri])


@pytest.mark.parametrize("col", ["weekly_up", "monthly_up", "w_rsi14", "rsi2", "pullback_atr",
                                 "mtf_score", "above200", "bb_z"])
def test_features_are_causal(col):
    """Truncating the future must not change any past feature value."""
    df = _ohlc()
    full = build_features(df)
    cut = build_features(df.iloc[:450])
    a = full[col].iloc[:450]
    b = cut[col]
    both = a.notna() & b.notna()
    assert (a[both] == b[both]).all()


def test_forward_returns():
    s = pd.Series([100.0, 110.0, 121.0])
    fr = forward_returns(s, (1, 2))
    assert fr.fwd1.iloc[0] == pytest.approx(0.10)
    assert fr.fwd2.iloc[0] == pytest.approx(0.21)
    assert np.isnan(fr.fwd2.iloc[-1])
