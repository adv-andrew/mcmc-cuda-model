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
    # newer source rescaled at the overlap (1999-01-08: 104 vs 200) -> continuous returns
    assert df.Close.iloc[-1] == pytest.approx(201.0 * 104.0 / 200.0)
    assert df.Close.pct_change().iloc[-1] == pytest.approx(201.0 / 200.0 - 1)


def test_etf_split_table():
    idx = pd.to_datetime(["2025-12-04", "2025-12-05"])
    assert md.split_factor(idx, "XLK").tolist() == [2.0, 1.0]


# ----------------------------------------------------------------------
# shares backtester
# ----------------------------------------------------------------------

def test_shares_backtester_pnl_and_costs():
    from backtesting.shares_backtest import SharesSwingBacktester

    idx = pd.bdate_range("2022-01-03", periods=20)
    close = pd.Series(100.0 * 1.01 ** np.arange(20), index=idx)
    entries = pd.Series(False, index=idx)
    entries.iloc[0] = True
    exits = ExitRules(min_hold=3, max_hold=5, signal_exit=False, profit_target=None)
    bt = SharesSwingBacktester(exits, position_frac=0.5, max_concurrent=1, cost_bps=10)
    res = bt.run({"X": close}, {"X": entries}, {"X": pd.Series(False, index=idx)})
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.days_held == 5 and t.exit_reason == "time"
    gross = 1.01 ** 5 - 1
    assert t.ret_on_risk == pytest.approx((1 + gross) * (1 - 0.001) - 1)
    # equity: half invested, both legs pay 10 bps
    expected = 100_000 * 0.5 * (1.01 ** 5) * (1 - 0.001) + 100_000 * 0.5 * (1 - 0.001)
    assert res.equity.iloc[-1] == pytest.approx(expected)


def test_shares_backtester_slots_and_signal_exit():
    from backtesting.shares_backtest import SharesSwingBacktester

    idx = pd.bdate_range("2022-01-03", periods=30)
    close = pd.Series(100.0, index=idx)
    entries = pd.Series(True, index=idx)
    ex = pd.Series(False, index=idx)
    ex.iloc[[4, 12, 20]] = True
    bt = SharesSwingBacktester(ExitRules(min_hold=3, max_hold=10), position_frac=0.3,
                               max_concurrent=2, cost_bps=0)
    res = bt.run({"A": close, "B": close, "C": close},
                 {"A": entries, "B": entries, "C": entries}, {s: ex for s in "ABC"})
    tf = res.trade_frame()
    for d in idx:
        assert ((tf.entry_date <= d) & (tf.exit_date > d)).sum() <= 2
    assert (tf[tf.exit_reason == "signal"].days_held >= 3).all()
    assert res.equity.iloc[-1] == pytest.approx(100_000)


# ----------------------------------------------------------------------
# cash yield and excess-return Sharpe
# ----------------------------------------------------------------------

def test_daily_risk_free_calendar_accrual():
    from backtesting.options_pricing import daily_risk_free

    idx = pd.to_datetime(["2023-03-03", "2023-03-06", "2023-03-07"])  # Fri, Mon, Tue
    rf = daily_risk_free(idx)
    assert rf.iloc[0] == 0.0
    assert rf.iloc[1] == pytest.approx(0.052 * 3 / 365)
    assert rf.iloc[2] == pytest.approx(0.052 * 1 / 365)


def test_idle_cash_earns_tbill_rate():
    from backtesting.shares_backtest import SharesSwingBacktester

    idx = pd.bdate_range("2023-01-02", "2023-12-29")
    close = pd.Series(100.0, index=idx)
    never = pd.Series(False, index=idx)
    res = SharesSwingBacktester(ExitRules(), cash_yield=True).run(
        {"X": close}, {"X": never}, {"X": never})
    growth = res.equity.iloc[-1] / res.equity.iloc[0] - 1
    days = (idx[-1] - idx[0]).days
    assert growth == pytest.approx((1 + 0.052 * 1 / 365) ** days - 1, rel=0.02)
    st = res.stats()
    # all-cash: positive raw Sharpe but ~zero excess Sharpe
    assert abs(st["sharpe_excess"]) < 0.05


def test_options_backtester_cash_yield_toggle():
    close, iv, ex = _path(n=200)
    entries = pd.Series(False, index=close.index)
    entries.iloc[::40] = True
    f = pd.DataFrame({"close": close})
    on = OptionsSwingBacktester(SPEC, EXITS, cash_yield=True).run(
        {"SPY": f}, {"SPY": iv}, {"SPY": entries}, {"SPY": ex})
    off = OptionsSwingBacktester(SPEC, EXITS).run(
        {"SPY": f}, {"SPY": iv}, {"SPY": entries}, {"SPY": ex})
    assert on.equity.iloc[-1] > off.equity.iloc[-1]
    assert len(on.trades) == len(off.trades)


# ----------------------------------------------------------------------
# portfolio building blocks
# ----------------------------------------------------------------------

def test_faber_signal_uses_completed_months_only():
    from backtesting.portfolio import faber_signal

    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    close = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    sig = faber_signal(close)
    assert not sig.loc["2019-06"].any()          # < 10 completed months
    assert sig.loc["2021-06"].all()               # steady uptrend
    # truncating the future must not change past values
    cut = faber_signal(close.loc[:"2020-12-31"])
    assert (sig.loc[:"2020-12-31"] == cut).all()


def test_trend_core_holds_asset_or_cash():
    from backtesting.portfolio import trend_core_returns

    idx = pd.bdate_range("2023-01-02", periods=6)
    px = pd.Series([100, 101, 102, 103, 104, 105.0], index=idx)
    sig = pd.Series([False, True, True, False, False, False], index=idx)
    r = trend_core_returns(px, sig, switch_cost_bps=0, cash_yield=False)
    # signal at t-1 decides exposure for day t
    assert r.iloc[1] == 0.0
    assert r.iloc[2] == pytest.approx(102 / 101 - 1)
    assert r.iloc[3] == pytest.approx(103 / 102 - 1)
    assert r.iloc[4] == 0.0
    costly = trend_core_returns(px, sig, switch_cost_bps=10, cash_yield=False)
    assert costly.iloc[2] == pytest.approx(r.iloc[2] - 0.001)


def test_blend_weights():
    from backtesting.portfolio import blend

    idx = pd.bdate_range("2023-01-02", periods=3)
    a = pd.Series([0.0, 0.10, 0.0], index=idx)
    b = pd.Series([0.0, -0.10, 0.0], index=idx)
    eq = blend({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    assert eq.iloc[-1] == pytest.approx(100_000)
    with pytest.raises(ValueError):
        blend({"a": a, "b": b}, {"a": 0.7, "b": 0.7})


def test_top_up_appends_and_rescales():
    base = pd.DataFrame({c: [100.0, 101.0] for c in md.COLUMNS},
                        index=pd.to_datetime(["2026-09-21", "2026-09-22"]))
    newer = pd.DataFrame({c: [202.0, 204.0, 206.0] for c in md.COLUMNS},
                         index=pd.to_datetime(["2026-09-22", "2026-09-23", "2026-09-24"]))
    out = md.top_up(base, newer)
    assert len(out) == 4 and out.index.is_monotonic_increasing
    # overlap 2026-09-22: 101 vs 202 -> newer rows scaled by 0.5
    assert out.Close.iloc[-1] == pytest.approx(103.0)
    assert md.top_up(base, newer.iloc[:1]).equals(base)


def test_stale_mirror_gets_yfinance_top_up(tmp_path):
    stale = pd.DataFrame({c: [100.0] for c in md.COLUMNS},
                         index=[pd.Timestamp.now().normalize() - pd.Timedelta(days=20)])
    fresh = pd.DataFrame({c: [100.0, 105.0] for c in md.COLUMNS},
                         index=[stale.index[0], pd.Timestamp.now().normalize()])
    with patch.object(md, "_http_get", return_value=b"DATE,OPEN,HIGH,LOW,CLOSE\n"), \
         patch.object(md, "_normalise_vix", return_value=stale), \
         patch.object(md, "_fetch_yfinance", return_value=fresh):
        df = md.load_daily("VIX", cache_dir=tmp_path, refresh=True, allow_partial=True)
    assert len(df) == 2 and df.Close.iloc[-1] == 105.0


# ----------------------------------------------------------------------
# code-review fixes: data splicing, unfinished bars, staleness
# ----------------------------------------------------------------------

def test_top_up_drops_nan_rows_and_rescales_adjclose_separately():
    base = pd.DataFrame({"Open": [100.0], "High": [100.0], "Low": [100.0], "Close": [100.0],
                         "AdjClose": [98.0], "Volume": [1.0]},
                        index=pd.to_datetime(["2026-09-22"]))
    newer = pd.DataFrame({"Open": [200.0, 210.0, np.nan], "High": [200.0, 210.0, np.nan],
                          "Low": [200.0, 210.0, np.nan], "Close": [200.0, 210.0, np.nan],
                          "AdjClose": [200.0, 210.0, np.nan], "Volume": [1.0, 1.0, 1.0]},
                         index=pd.to_datetime(["2026-09-22", "2026-09-23", "2026-09-24"]))
    out = md.top_up(base, newer)
    assert len(out) == 2 and not out.Close.isna().any()
    assert out.Close.iloc[-1] == pytest.approx(105.0)      # 210 * 100/200
    assert out.AdjClose.iloc[-1] == pytest.approx(102.9)   # 210 * 98/200
    # AdjClose return across the splice equals the source's own return
    assert out.AdjClose.pct_change().iloc[-1] == pytest.approx(210 / 200 - 1)


def test_drop_unfinished_and_business_days():
    idx = pd.to_datetime(["2026-10-05", "2026-10-06"])
    df = pd.DataFrame({c: [1.0, 2.0] for c in md.COLUMNS}, index=idx)
    before = pd.Timestamp("2026-10-06 15:45", tz=md.MARKET_TZ)
    after = pd.Timestamp("2026-10-06 16:30", tz=md.MARKET_TZ)
    assert len(md.drop_unfinished(df, now=before)) == 1
    assert len(md.drop_unfinished(df, now=after)) == 2
    assert md.business_days_old(pd.Timestamp("2026-10-02"), now=before) == 2  # Fri -> Tue
    assert md.business_days_old(pd.Timestamp("2026-10-06"), now=before) == 0


def test_unfinished_bar_never_cached(tmp_path):
    today = pd.Timestamp.now(tz=md.MARKET_TZ).tz_localize(None).normalize()
    df = pd.DataFrame({c: [1.0, 2.0] for c in md.COLUMNS},
                      index=[today - pd.Timedelta(days=1), today])
    chop = lambda d, now=None: d.iloc[:-1]  # noqa: E731  (pretend the session is open)
    empty = pd.DataFrame(columns=md.COLUMNS)
    with patch.object(md, "_http_get", return_value=b"DATE,OPEN,HIGH,LOW,CLOSE\n"), \
         patch.object(md, "_normalise_vix", return_value=df), \
         patch.object(md, "_fetch_yfinance", return_value=empty), \
         patch.object(md, "drop_unfinished", side_effect=chop):
        live = md.load_daily("VIX", cache_dir=tmp_path, refresh=True, allow_partial=True)
        final = md.load_daily("VIX", cache_dir=tmp_path, refresh=True)
    assert len(live) == 2 and len(final) == 1
    cached = pd.read_csv(tmp_path / "VIX.csv", index_col=0, parse_dates=True)
    assert len(cached) == 1


# ----------------------------------------------------------------------
# code-review fixes: engine parity and robustness
# ----------------------------------------------------------------------

def _engine_inputs(n=60, drift=-0.01):
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series(100 * np.exp(drift * np.arange(n)), index=idx)
    return idx, close


def test_engine_honours_stop_loss_like_backtester():
    from backtesting.core_overlay_engine import CoreOverlayEngine, EngineState

    idx, close = _engine_inputs()
    exits = ExitRules(min_hold=3, max_hold=10, profit_target=None, stop_loss=-0.3,
                      signal_exit=False)
    eng = CoreOverlayEngine(SPEC, exits, core_weight=0.0)
    st = EngineState(cash=100_000)
    for i, d in enumerate(idx):
        eng.step(st, d, 100.0, False, {"SPY": close.iloc[i]}, {"SPY": 0.16},
                 {"SPY": i == 0}, {"SPY": False})
    assert st.trades and st.trades[0]["exit_reason"] == "stop"
    assert -0.6 < st.trades[0]["ret_on_risk"] <= -0.3


def test_engine_rejects_invalid_core_price():
    from backtesting.core_overlay_engine import CoreOverlayEngine, EngineState

    eng = CoreOverlayEngine(SPEC, EXITS)
    with pytest.raises(ValueError):
        eng.step(EngineState(cash=1.0), pd.Timestamp("2024-01-02"), float("nan"), True,
                 {}, {}, {}, {})


def test_engine_keeps_last_mark_when_quote_missing():
    from backtesting.core_overlay_engine import CoreOverlayEngine, EngineState

    idx = pd.bdate_range("2024-01-02", periods=4)
    eng = CoreOverlayEngine(SPEC, ExitRules(min_hold=3, max_hold=10, profit_target=None,
                                            signal_exit=False), core_weight=0.0)
    st = EngineState(cash=100_000)
    px = [100.0, 104.0, float("nan"), 104.0]
    for i, d in enumerate(idx):
        eng.step(st, d, 100.0, False, {"SPY": px[i]}, {"SPY": 0.16}, {"SPY": i == 0},
                 {"SPY": False})
    eq = [v for _, v in st.equity]
    assert eq[2] == pytest.approx(eq[1], rel=1e-3)  # no jump back to cost on the gap day
