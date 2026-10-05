"""Tests for backtesting/options_pricing.py"""

import math

import numpy as np
import pandas as pd
import pytest

from backtesting.options_pricing import (
    CostModel,
    Leg,
    SkewModel,
    atm_iv_series,
    bs_delta,
    bs_price,
    close_value,
    open_position,
    position_return,
    profit_fraction,
    strike_for_delta,
)


class TestBlackScholes:
    def test_put_call_parity(self):
        s, k, t, v, r = 100.0, 95.0, 30 / 365, 0.25, 0.03
        c = bs_price(s, k, t, v, r, True)
        p = bs_price(s, k, t, v, r, False)
        assert c - p == pytest.approx(s - k * math.exp(-r * t), abs=1e-9)

    def test_expiry_is_intrinsic(self):
        assert bs_price(105, 100, 0.0, 0.2, 0.0, True) == pytest.approx(5.0)
        assert bs_price(95, 100, 0.0, 0.2, 0.0, True) == 0.0
        assert bs_price(95, 100, 0.0, 0.2, 0.0, False) == pytest.approx(5.0)

    def test_atm_approximation(self):
        # ATM call ~ 0.4 * S * sigma * sqrt(T) when r = 0
        price = bs_price(100, 100, 30 / 365, 0.20, 0.0, True)
        assert price == pytest.approx(0.4 * 100 * 0.20 * math.sqrt(30 / 365), rel=0.02)

    def test_delta_bounds_and_sign(self):
        assert 0.0 < bs_delta(100, 100, 0.1, 0.2, 0.0, True) < 1.0
        assert -1.0 < bs_delta(100, 100, 0.1, 0.2, 0.0, False) < 0.0

    @pytest.mark.parametrize("is_call", [True, False])
    @pytest.mark.parametrize("target", [0.20, 0.30, 0.50])
    def test_strike_for_delta_inverts_delta(self, is_call, target):
        k = strike_for_delta(500, target, 14 / 365, 0.18, 0.04, is_call, increment=0.01)
        assert abs(bs_delta(500, k, 14 / 365, 0.18, 0.04, is_call)) == pytest.approx(target, abs=0.01)


class TestSkew:
    def test_otm_puts_richer_otm_calls_cheaper(self):
        sk = SkewModel()
        atm = sk.iv(0.15, 100, 100, 30 / 365)
        assert atm == pytest.approx(0.15)
        assert sk.iv(0.15, 100, 95, 30 / 365) > atm
        assert sk.iv(0.15, 100, 105, 30 / 365) < atm

    def test_bounded(self):
        sk = SkewModel()
        assert sk.iv(0.15, 100, 10, 30 / 365) <= 0.15 * sk.cap_mult
        assert sk.iv(0.15, 100, 1000, 30 / 365) >= 0.15 * sk.floor_mult


class TestPositions:
    def setup_method(self):
        self.date = pd.Timestamp("2024-01-02")
        self.skew = SkewModel()
        self.costs = CostModel.etf()

    def test_debit_spread_risk_and_profit(self):
        legs = [Leg(True, 100, +1), Leg(True, 105, -1)]
        pos = open_position(legs, 100, self.date, 21, 0.2, self.skew, self.costs)
        assert pos.entry_value > 0
        assert pos.max_loss == pytest.approx(pos.entry_value + pos.entry_cost)
        assert pos.max_profit == pytest.approx(5 - pos.max_loss)

    def test_credit_spread_risk_and_profit(self):
        legs = [Leg(False, 95, -1), Leg(False, 90, +1)]
        pos = open_position(legs, 100, self.date, 14, 0.2, self.skew, self.costs)
        assert pos.entry_value < 0
        credit = -pos.entry_value - pos.entry_cost
        assert pos.max_profit == pytest.approx(credit)
        assert pos.max_loss == pytest.approx(5 - credit)

    def test_debit_spread_bounded_outcomes_at_expiry(self):
        legs = [Leg(True, 100, +1), Leg(True, 105, -1)]
        pos = open_position(legs, 100, self.date, 7, 0.2, self.skew, self.costs)
        expiry = pos.expiry
        win = close_value(pos, 200, expiry, 0.2, self.skew, self.costs)
        lose = close_value(pos, 50, expiry, 0.2, self.skew, self.costs)
        assert position_return(pos, lose) == pytest.approx(-1.0)
        assert profit_fraction(pos, win) == pytest.approx(1.0)

    def test_round_trip_costs_lose_money_if_nothing_moves(self):
        legs = [Leg(True, 100, +1), Leg(True, 105, -1)]
        pos = open_position(legs, 100, self.date, 21, 0.2, self.skew, self.costs)
        immediate = close_value(pos, 100, self.date, 0.2, self.skew, self.costs)
        assert position_return(pos, immediate) < 0

    def test_cost_model_minimum_tick(self):
        c = CostModel(min_half_spread=0.01, pct_half_spread=0.01, commission_per_contract=0.0)
        assert c.leg_cost(0.20) == pytest.approx(0.01)
        assert c.leg_cost(5.00) == pytest.approx(0.05)


def _frame(close: np.ndarray, start="2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({"Close": close}, index=idx)


class TestAtmIvSeries:
    def test_scales_with_vix_and_floors_at_realized(self):
        rng = np.random.default_rng(0)
        n = 400
        spy = _frame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
        vix = pd.DataFrame({"Close": np.full(n, 20.0)}, index=spy.index)
        iv = atm_iv_series(spy, vix, spy)
        # SPY vs itself -> ratio 1 -> 0.9 * 20% = 18%, unless realized vol is higher
        rv10 = np.log(spy.Close).diff().rolling(10).std() * math.sqrt(252)
        expected = np.maximum(0.18, rv10.fillna(0))
        assert np.allclose(iv.iloc[300:], expected.iloc[300:])

    def test_stale_vix_falls_back_to_realized(self):
        rng = np.random.default_rng(1)
        n = 300
        spy = _frame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
        vix = pd.DataFrame({"Close": np.full(n - 10, 20.0)}, index=spy.index[: n - 10])
        iv = atm_iv_series(spy, vix, spy)
        assert not iv.iloc[-1:].isna().any()
        rv20 = np.log(spy.Close).diff().rolling(20).std() * math.sqrt(252)
        # beyond the 3-day forward-fill window the estimate is realized-vol based
        assert iv.iloc[-1] == pytest.approx(max(1.2 * rv20.iloc[-1],
                                                 (np.log(spy.Close).diff().rolling(10).std()
                                                  * math.sqrt(252)).iloc[-1]))
