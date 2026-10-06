"""Tests for the options swing strategy, its backtester, and data helpers."""

import numpy as np
import pandas as pd
import pytest

from backtesting.market_data import _normalise_vix, split_factor
from backtesting.options_backtest import (
    ExitRules,
    OptionsSwingBacktester,
    StructureSpec,
    build_legs,
    monte_carlo_equity,
)
from backtesting.options_pricing import SkewModel
from trading.options_swing import (
    SwingConfig,
    build_ticket,
    confidence_score,
    confidence_tier,
    entry_signal,
    next_expiry,
)
from trading.regime_mcmc import RegimeMCMC, compute_states


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _feature_row(**kw) -> pd.DataFrame:
    base = dict(close=100.0, above200=True, pullback_atr=1.6, rsi2=8.0, ibs=0.5, bb_z=-1.0,
                weekly_up=True, monthly_up=True, mtf_score=2.0, rv20=0.12, sma5=101.0)
    base.update(kw)
    return pd.DataFrame([base], index=[pd.Timestamp("2024-03-05")])


def _synthetic_features(n=300, drift=0.002):
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = pd.Series(100 * np.exp(drift * np.arange(n)), index=idx)
    f = pd.DataFrame({"close": close, "sma5": close.rolling(5).mean().bfill()})
    iv = pd.Series(0.15, index=idx)
    return f, iv


# ----------------------------------------------------------------------
# strategy rules
# ----------------------------------------------------------------------

class TestEntryRules:
    cfg = SwingConfig()

    def test_classic_dip_in_uptrend_fires(self):
        assert entry_signal(_feature_row(), self.cfg).iloc[0]

    def test_downtrend_never_fires(self):
        assert not entry_signal(_feature_row(above200=False), self.cfg).iloc[0]

    def test_small_pullback_never_fires(self):
        assert not entry_signal(_feature_row(pullback_atr=1.0), self.cfg).iloc[0]

    def test_needs_a_trigger(self):
        row = _feature_row(rsi2=40.0, ibs=0.6, bb_z=-0.5, pullback_atr=1.7)
        assert not entry_signal(row, self.cfg).iloc[0]
        assert entry_signal(row.assign(pullback_atr=2.1), self.cfg).iloc[0]
        assert entry_signal(row.assign(ibs=0.1, rsi2=14.0), self.cfg).iloc[0]
        assert entry_signal(row.assign(bb_z=-2.5), self.cfg).iloc[0]

    def test_weekly_trend_option(self):
        cfg = SwingConfig(require_weekly_trend=True)
        assert not entry_signal(_feature_row(weekly_up=False), cfg).iloc[0]


class TestConfidence:
    def test_score_levels_and_tiers(self):
        cfg = SwingConfig()
        assert confidence_score(_feature_row()).iloc[0] == 100
        assert confidence_score(_feature_row(rv20=0.25)).iloc[0] == 70
        assert confidence_score(_feature_row(rv20=0.25, mtf_score=0)).iloc[0] == 40
        assert confidence_tier(100, cfg) == "HIGH"
        assert confidence_tier(70, cfg) == "MEDIUM"
        assert confidence_tier(40, cfg) == "LOW"


def test_next_expiry_is_a_friday_at_least_dte_out():
    for d in pd.bdate_range("2024-01-01", periods=10):
        e = next_expiry(d, 21)
        assert e.dayofweek == 4
        assert 21 <= (e - d).days < 28


def test_ticket_default_is_itm_call():
    f = _feature_row(rsi2=4.0, ibs=0.1, vix=18.0, vix_ratio10=1.2)
    t = build_ticket("SPY", f, 0.18, SwingConfig())
    assert t.short_strike is None and t.max_profit is None
    assert t.long_strike < t.spot  # in the money
    assert 0.72 <= t.long_delta <= 0.88
    assert t.shares_position_frac == pytest.approx(0.33)
    assert t.est_debit > t.spot - t.long_strike  # intrinsic plus some time value
    assert t.take_profit_value == pytest.approx(1.6 * t.est_debit, abs=0.02)


def test_ticket_call_debit_spread_option():
    f = _feature_row(rsi2=4.0, ibs=0.1, vix=18.0, vix_ratio10=1.2)
    cfg = SwingConfig(structure_kind="call_debit_spread", dte=21, long_delta=0.55)
    t = build_ticket("SPY", f, 0.18, cfg)
    assert t.long_strike < t.short_strike
    assert 0 < t.est_debit < t.short_strike - t.long_strike
    assert t.max_profit == pytest.approx(t.short_strike - t.long_strike - t.est_debit, abs=0.02)
    assert 0.45 <= t.long_delta <= 0.65


def test_unknown_structure_rejected():
    with pytest.raises(ValueError):
        SwingConfig(structure_kind="iron_condor").structure()


def test_config_loads_from_yaml():
    cfg = SwingConfig.from_yaml("config/default.yaml")
    assert cfg.symbols == ("SPY", "QQQ", "IWM")
    assert cfg.min_hold == 3
    assert cfg.structure().kind == "long_call"
    assert cfg.long_delta == 0.80 and cfg.dte == 30
    assert cfg.max_hold == 10 and cfg.risk_per_trade == 0.06


# ----------------------------------------------------------------------
# backtester
# ----------------------------------------------------------------------

class TestBacktester:
    def test_legs_for_each_structure(self):
        skew = SkewModel()
        for kind in ("long_call", "call_debit_spread", "put_credit_spread"):
            spec = StructureSpec(kind, dte=14)
            bull = build_legs(spec, +1, 100.0, 14 / 365, 0.2, 0.0, skew, True)
            bear = build_legs(spec, -1, 100.0, 14 / 365, 0.2, 0.0, skew, True)
            assert all(leg.strike > 0 for leg in bull + bear)
        cds = build_legs(StructureSpec("call_debit_spread"), +1, 100, 0.05, 0.2, 0, skew, True)
        assert cds[0].qty == 1 and cds[1].qty == -1 and cds[0].strike < cds[1].strike

    def test_rising_market_makes_money_with_debit_spreads(self):
        f, iv = _synthetic_features()
        entries = pd.Series(False, index=f.index)
        entries.iloc[::15] = True
        bt = OptionsSwingBacktester(StructureSpec("call_debit_spread", dte=21),
                                    ExitRules(min_hold=3, max_hold=5, signal_exit=False,
                                              profit_target=None), risk_per_trade=0.05)
        res = bt.run({"SPY": f}, {"SPY": iv}, {"SPY": entries})
        st = res.stats()
        assert st["trades"] > 5
        assert st["avg_ret"] > 0
        assert res.equity.iloc[-1] > res.equity.iloc[0]
        assert all(3 <= t.days_held <= 5 for t in res.trades)

    def test_falling_market_loses_money_with_debit_spreads(self):
        f, iv = _synthetic_features(drift=-0.003)
        entries = pd.Series(False, index=f.index)
        entries.iloc[::15] = True
        bt = OptionsSwingBacktester(StructureSpec("call_debit_spread", dte=21),
                                    ExitRules(min_hold=5, max_hold=5, signal_exit=False,
                                              profit_target=None))
        res = bt.run({"SPY": f}, {"SPY": iv}, {"SPY": entries})
        assert res.stats()["avg_ret"] < 0

    def test_max_concurrent_respected(self):
        f, iv = _synthetic_features()
        feats = {s: f for s in ("SPY", "QQQ", "IWM")}
        ivs = {s: iv for s in feats}
        entries = {s: pd.Series(True, index=f.index) for s in feats}
        bt = OptionsSwingBacktester(StructureSpec("call_debit_spread"),
                                    ExitRules(min_hold=10, max_hold=10, signal_exit=False,
                                              profit_target=None), max_concurrent=2)
        res = bt.run(feats, ivs, entries)
        tf = res.trade_frame()
        for d in f.index[20:200:7]:
            open_n = ((tf.entry_date <= d) & (tf.exit_date > d)).sum()
            assert open_n <= 2

    def test_split_scale_keeps_returns_continuous(self):
        # Same economic path, but quoted 10x higher before a "split" date:
        # the price_scale only changes the strike grid and $ minimum ticks.
        f, iv = _synthetic_features()
        entries = pd.Series(False, index=f.index)
        entries.iloc[::20] = True
        scale = pd.Series(np.where(f.index < f.index[150], 10.0, 1.0), index=f.index)
        bt = OptionsSwingBacktester(StructureSpec("call_debit_spread"), ExitRules(),
                                    etfs=())
        res = bt.run({"X": f}, {"X": iv}, {"X": entries}, price_scale={"X": scale})
        assert all(abs(t.ret_on_risk) < 3 for t in res.trades)

    def test_monte_carlo_equity(self):
        rets = np.array([0.3, -1.0, 0.2, 0.25, 0.1])
        mc = monte_carlo_equity(rets, trades_per_year=20, risk_per_trade=0.05, n_sims=2000)
        assert mc["p05_return"] <= mc["median_return"] <= mc["p95_return"]
        assert -1 < mc["p95_max_dd"] <= mc["median_max_dd"] <= 0
        assert 0 <= mc["prob_loss"] <= 1


# ----------------------------------------------------------------------
# data helpers and regime MCMC
# ----------------------------------------------------------------------

def test_split_factor_unadjusts():
    idx = pd.to_datetime(["2024-06-07", "2024-06-10"])
    f = split_factor(idx, "NVDA")
    assert f.tolist() == [10.0, 1.0]
    assert split_factor(idx, "SPY").tolist() == [1.0, 1.0]


def test_vix_normalisation():
    raw = pd.DataFrame({"DATE": ["2024-01-02", "2024-01-03"], "OPEN": [13, 14],
                        "HIGH": [14, 15], "LOW": [12, 13], "CLOSE": [13.5, 14.5]})
    df = _normalise_vix(raw)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    assert df.Close.iloc[-1] == 14.5


class TestRegimeMCMC:
    def _close(self, n=900, seed=3):
        rng = np.random.default_rng(seed)
        return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
                         index=pd.bdate_range("2018-01-01", periods=n))

    def test_states_are_causal(self):
        c = self._close()
        full = compute_states(c)
        cut = compute_states(c.iloc[:600])
        assert (full.iloc[:600] == cut).all()

    def test_forecast_uses_only_past(self):
        c = self._close()
        m1 = RegimeMCMC(horizon=3, n_paths=500, seed=1)
        m2 = RegimeMCMC(horizon=3, n_paths=500, seed=1)
        a = m1.forecast_latest(c.iloc[:700])
        # appending future data must not change a forecast made at bar 699
        logret = np.log(c).diff().fillna(0).to_numpy()
        states = compute_states(c).to_numpy()
        b = m2.forecast_at(logret, states, 699)
        assert a is not None and b is not None
        assert a.p_up == pytest.approx(b.p_up)
        assert 0 <= a.p_up <= 1 and a.q05 <= a.median_return <= a.q95
