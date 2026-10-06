"""Tests for the per-trade simulator, cost tiers, and the release-data loader."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import backtesting.market_data as md
from backtesting.options_backtest import (
    ExitRules,
    OptionsSwingBacktester,
    PricingScenario,
    StructureSpec,
    non_overlapping_entries,
    simulate_trades,
)
from backtesting.options_pricing import SkewModel


def _path(n=120, drift=0.002, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(drift + rng.normal(0, 0.004, n))), index=idx)
    iv = pd.Series(0.16, index=idx)
    exit_sig = close > close.rolling(5).mean()
    return close, iv, exit_sig


SPEC = StructureSpec("long_call", dte=30, long_delta=0.70)
EXITS = ExitRules(min_hold=3, max_hold=7, profit_target=0.6, signal_exit=True)


class TestSimulateTrades:
    def test_every_day_mode_and_bounds(self):
        close, iv, ex = _path()
        tr = simulate_trades(close, iv, ex, SPEC, EXITS)
        assert len(tr) == len(close) - EXITS.max_hold - 1
        assert tr.held.between(1, EXITS.max_hold).all()
        assert set(tr.reason) <= {"signal", "time", "target", "expiry", "stop"}
        assert (tr.ret >= -1.0).all()

    def test_min_hold_respected_for_signal_exits(self):
        close, iv, ex = _path()
        tr = simulate_trades(close, iv, ex, SPEC, EXITS)
        assert (tr[tr.reason == "signal"].held >= EXITS.min_hold).all()

    def test_uptrend_profitable_and_underlying_consistent(self):
        close, iv, ex = _path(drift=0.004)
        tr = simulate_trades(close, iv, ex, SPEC, EXITS)
        assert tr.ret.mean() > 0
        # option P&L should rise with the underlying's move over the same window
        assert tr.ret.rank().corr(tr.und_ret.rank()) > 0.8

    def test_entry_dates_subset(self):
        close, iv, ex = _path()
        dates = close.index[[10, 30, 50]]
        tr = simulate_trades(close, iv, ex, SPEC, EXITS, entry_dates=dates)
        assert list(tr.index) == list(dates)

    def test_stress_scenarios_cost_money(self):
        close, iv, ex = _path()
        base = simulate_trades(close, iv, ex, SPEC, EXITS).ret.mean()
        rich = simulate_trades(close, iv, ex, SPEC, EXITS,
                               PricingScenario(entry_iv_mult=1.10)).ret.mean()
        wide = simulate_trades(close, iv, ex, SPEC, EXITS,
                               PricingScenario(cost_mult=2.0)).ret.mean()
        assert rich < base and wide < base

    def test_call_skew_hurts_spread_not_much_single_call(self):
        close, iv, ex = _path()
        cds = StructureSpec("call_debit_spread", dte=21, long_delta=0.55, short_delta=0.30)
        steep = PricingScenario(skew=SkewModel(call_slope=0.30))
        d_cds = (simulate_trades(close, iv, ex, cds, EXITS).ret.mean()
                 - simulate_trades(close, iv, ex, cds, EXITS, steep).ret.mean())
        d_call = (simulate_trades(close, iv, ex, SPEC, EXITS).ret.mean()
                  - simulate_trades(close, iv, ex, SPEC, EXITS, steep).ret.mean())
        assert d_cds > abs(d_call)


def test_non_overlapping_entries():
    idx = pd.bdate_range("2022-01-03", periods=12)
    sig = pd.Series([True, True, False, True, True, True, False, False, True, False, False, False],
                    index=idx)
    held = pd.Series(3, index=idx)
    taken = non_overlapping_entries(sig, held)
    # entry 0 holds until bar 3 -> next allowed entry is bar 4, then bar 8
    assert list(taken) == [idx[0], idx[4], idx[8]]


def test_tight_cost_symbols_separate_from_strike_grid():
    bt = OptionsSwingBacktester(SPEC, EXITS, etfs=("SPY", "XLF"), tight_cost_symbols=("SPY",))
    assert bt._costs("SPY").pct_half_spread == pytest.approx(0.01)
    assert bt._costs("XLF").pct_half_spread == pytest.approx(0.02)
    default = OptionsSwingBacktester(SPEC, EXITS, etfs=("SPY", "XLF"))
    assert default._costs("XLF").pct_half_spread == pytest.approx(0.01)


# ----------------------------------------------------------------------
# release-parquet loader (no network)
# ----------------------------------------------------------------------

def _release_table():
    rows = []
    for t, base in (("XLF", 20.0), ("SPY", 100.0)):
        for i, d in enumerate(pd.bdate_range("1999-01-04", periods=5)):
            rows.append({"date": d, "ticker": t, "open": base + i, "high": base + i + 1,
                         "low": base + i - 1, "close": base + i, "adj_close": base + i,
                         "volume": 1e6})
    return pd.DataFrame(rows)


def test_normalise_release_filters_ticker():
    df = md._normalise_release(_release_table(), "XLF")
    assert list(df.columns) == md.COLUMNS
    assert len(df) == 5 and df.Close.iloc[0] == 20.0
    assert md._normalise_release(_release_table(), "NOPE").empty


def test_load_daily_falls_back_to_release(tmp_path):
    with patch.object(md, "_release_table", return_value=_release_table()), \
         patch.object(md, "_fetch_yfinance", side_effect=RuntimeError("offline")):
        df = md.load_daily("XLF", cache_dir=tmp_path)
    assert len(df) == 5 and (tmp_path / "XLF.csv").exists()


def test_load_long_history_splices_recent_mirror(tmp_path):
    recent = pd.DataFrame(
        {c: [200.0, 201.0] for c in md.COLUMNS},
        index=pd.to_datetime(["1999-01-08", "1999-01-11"]))
    with patch.object(md, "_release_table", return_value=_release_table()), \
         patch.object(md, "load_daily", return_value=recent):
        df = md.load_long_history("SPY", cache_dir=tmp_path)
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert len(df) == 6  # 5 release days + 1 newer mirror day
    assert df.Close.iloc[-1] == 201.0


def test_etf_split_table():
    idx = pd.to_datetime(["2025-12-04", "2025-12-05"])
    assert md.split_factor(idx, "XLK").tolist() == [2.0, 1.0]
