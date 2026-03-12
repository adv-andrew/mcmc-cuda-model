"""Tests for MCMCIndicator signal generation."""

import math
import numpy as np
import pandas as pd
import pytest

from trading.indicator import MCMCIndicator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "ticker",
    "timestamp",
    "timeframe",
    "slope_degrees",
    "direction",
    "band_width_pct",
    "regime",
    "regime_confidence",
    "signal_strength",
    "suggested_action",
    "forecast_median",
    "forecast_p5",
    "forecast_p95",
    "current_price",
}

VALID_DIRECTIONS = {"BULLISH", "BEARISH", "NEUTRAL"}
VALID_ACTIONS = {"BUY", "SELL", "HOLD"}


def _make_price_df(prices, col="close"):
    return pd.DataFrame({col: prices})


def _flat_prices(n=60, base=100.0):
    return np.full(n, base)


def _uptrend_prices(n=60, start=100.0, pct_per_step=0.005):
    return np.array([start * (1 + pct_per_step) ** i for i in range(n)])


def _downtrend_prices(n=60, start=100.0, pct_per_step=0.005):
    return np.array([start * (1 - pct_per_step) ** i for i in range(n)])


def _indicator(n_simulations=500, **kwargs):
    """Return a fast MCMCIndicator suitable for unit tests."""
    return MCMCIndicator(
        n_simulations=n_simulations,
        n_steps=10,
        enable_gpu=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestGenerateSignalOutputKeys:
    def test_all_expected_keys_present(self):
        ind = _indicator()
        df = _make_price_df(_uptrend_prices())
        result = ind.generate_signal("AAPL", df, "1h")
        assert EXPECTED_KEYS == set(result.keys())

    def test_direction_valid(self):
        ind = _indicator()
        df = _make_price_df(_uptrend_prices())
        result = ind.generate_signal("TEST", df, "5m")
        assert result["direction"] in VALID_DIRECTIONS

    def test_action_valid(self):
        ind = _indicator()
        df = _make_price_df(_uptrend_prices())
        result = ind.generate_signal("TEST", df, "5m")
        assert result["suggested_action"] in VALID_ACTIONS

    def test_ticker_and_timeframe_passthrough(self):
        ind = _indicator()
        df = _make_price_df(_uptrend_prices())
        result = ind.generate_signal("TSLA", df, "15m")
        assert result["ticker"] == "TSLA"
        assert result["timeframe"] == "15m"

    def test_signal_strength_between_0_and_1(self):
        ind = _indicator()
        df = _make_price_df(_uptrend_prices())
        result = ind.generate_signal("X", df, "1d")
        assert 0.0 <= result["signal_strength"] <= 1.0

    def test_regime_confidence_between_0_and_1(self):
        ind = _indicator()
        df = _make_price_df(_uptrend_prices())
        result = ind.generate_signal("X", df, "1d")
        assert 0.0 <= result["regime_confidence"] <= 1.0

    def test_regime_is_0_1_or_2(self):
        ind = _indicator()
        df = _make_price_df(_uptrend_prices())
        result = ind.generate_signal("X", df, "1d")
        assert result["regime"] in (0, 1, 2)

    def test_forecast_p5_le_median_le_p95(self):
        ind = _indicator(n_simulations=2000)
        df = _make_price_df(_uptrend_prices())
        result = ind.generate_signal("X", df, "1d")
        assert result["forecast_p5"] <= result["forecast_median"] <= result["forecast_p95"]

    def test_current_price_matches_last_close(self):
        ind = _indicator()
        prices = _uptrend_prices()
        df = _make_price_df(prices)
        result = ind.generate_signal("X", df, "1d")
        assert result["current_price"] == pytest.approx(prices[-1], rel=1e-4)


# ---------------------------------------------------------------------------
# Slope direction
# ---------------------------------------------------------------------------


class TestSlopeDirection:
    def test_uptrend_produces_positive_slope(self):
        """Strong uptrend should yield a positive slope_degrees."""
        ind = _indicator(n_simulations=2000)
        # Seed for reproducibility
        np.random.seed(42)
        df = _make_price_df(_uptrend_prices(n=60, pct_per_step=0.01))
        result = ind.generate_signal("UP", df, "1h")
        # Slope should be positive; the median forecast will be above current price
        assert result["slope_degrees"] > 0, (
            f"Expected positive slope for uptrend, got {result['slope_degrees']}"
        )

    def test_downtrend_produces_negative_slope(self):
        """Strong downtrend should yield a negative slope_degrees."""
        ind = _indicator(n_simulations=2000)
        np.random.seed(42)
        df = _make_price_df(_downtrend_prices(n=60, pct_per_step=0.01))
        result = ind.generate_signal("DOWN", df, "1h")
        assert result["slope_degrees"] < 0, (
            f"Expected negative slope for downtrend, got {result['slope_degrees']}"
        )

    def test_bullish_direction_for_large_positive_slope(self):
        ind = MCMCIndicator(slope_threshold=5.0, n_simulations=500, n_steps=10, enable_gpu=False)
        # Force a deterministic positive slope by patching forecast > current
        slope = ind._calculate_slope(100.0, 110.0, _uptrend_prices())
        direction = ind._slope_to_direction(slope)
        assert direction == "BULLISH"

    def test_bearish_direction_for_large_negative_slope(self):
        ind = MCMCIndicator(slope_threshold=5.0, n_simulations=500, n_steps=10, enable_gpu=False)
        slope = ind._calculate_slope(100.0, 90.0, _downtrend_prices())
        direction = ind._slope_to_direction(slope)
        assert direction == "BEARISH"

    def test_neutral_direction_near_zero_slope(self):
        ind = MCMCIndicator(slope_threshold=10.0, n_simulations=500, n_steps=10, enable_gpu=False)
        slope = ind._calculate_slope(100.0, 100.001, _flat_prices())
        direction = ind._slope_to_direction(slope)
        assert direction == "NEUTRAL"


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------


class TestRegimeDetection:
    def test_regime_returns_tuple(self):
        ind = _indicator()
        regime, confidence = ind._detect_regime(_uptrend_prices())
        assert isinstance(regime, int)
        assert isinstance(confidence, float)

    def test_bull_regime_for_strong_uptrend(self):
        """Consistent uptrend should land in regime 2 (bull)."""
        ind = _indicator()
        prices = _uptrend_prices(n=120, pct_per_step=0.02)
        regime, _ = ind._detect_regime(prices)
        # The last return is the most positive; quantile classification puts it in regime 2
        assert regime == 2

    def test_bear_regime_for_strong_downtrend(self):
        """Consistent downtrend should land in regime 0 (bear)."""
        ind = _indicator()
        prices = _downtrend_prices(n=120, pct_per_step=0.02)
        regime, _ = ind._detect_regime(prices)
        assert regime == 0

    def test_confidence_high_for_consistent_trend(self):
        """A long consistent trend should produce high regime confidence."""
        ind = _indicator()
        prices = _uptrend_prices(n=120, pct_per_step=0.01)
        _, confidence = ind._detect_regime(prices)
        assert confidence >= 0.5

    def test_short_price_series_returns_default(self):
        ind = _indicator()
        regime, confidence = ind._detect_regime(np.array([100.0, 101.0]))
        assert regime == 1
        assert confidence == 0.5


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------


class TestBuildTransitionMatrix:
    def test_shape(self):
        ind = _indicator()
        states = np.array([0, 1, 2, 1, 0, 2])
        matrix = ind._build_transition_matrix(states, n_states=3)
        assert matrix.shape == (3, 3)

    def test_rows_sum_to_one(self):
        ind = _indicator()
        states = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        matrix = ind._build_transition_matrix(states, n_states=3)
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(3), atol=1e-9)


# ---------------------------------------------------------------------------
# Signal strength
# ---------------------------------------------------------------------------


class TestCalculateSignalStrength:
    def test_high_strength_for_strong_signal(self):
        ind = _indicator()
        strength = ind._calculate_signal_strength(
            slope=45.0, band_width=0.01, regime_confidence=0.9
        )
        assert strength > 0.5

    def test_low_strength_for_wide_band(self):
        ind = _indicator()
        strength = ind._calculate_signal_strength(
            slope=45.0, band_width=0.10, regime_confidence=0.5
        )
        ind2 = _indicator()
        strength_narrow = ind2._calculate_signal_strength(
            slope=45.0, band_width=0.01, regime_confidence=0.5
        )
        assert strength <= strength_narrow

    def test_bounded_between_0_and_1(self):
        ind = _indicator()
        for slope in [-90, -45, 0, 45, 90, 200]:
            for bw in [0.0, 0.05, 0.2]:
                for rc in [0.0, 0.5, 1.0]:
                    s = ind._calculate_signal_strength(slope, bw, rc)
                    assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Bootstrap fallback
# ---------------------------------------------------------------------------


class TestBootstrapForecast:
    def test_shape(self):
        ind = _indicator()
        prices = _uptrend_prices()
        returns = np.diff(np.log(prices))
        paths = ind._bootstrap_forecast(prices, returns)
        assert paths.shape == (ind.n_simulations, ind.n_steps + 1)

    def test_first_column_is_current_price(self):
        ind = _indicator()
        prices = _uptrend_prices()
        returns = np.diff(np.log(prices))
        paths = ind._bootstrap_forecast(prices, returns)
        np.testing.assert_allclose(paths[:, 0], prices[-1])


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_loads_default_yaml(self, tmp_path):
        cfg = {
            "mcmc": {"n_simulations": 1000, "n_steps": 10, "n_regimes": 3},
            "signal": {
                "slope_threshold": 8.0,
                "band_width_max": 0.04,
                "regime_confidence_min": 0.55,
                "mtf_weight": 0.25,
                "regime_weight": 0.25,
            },
        }
        import yaml

        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump(cfg))

        ind = MCMCIndicator.from_config(str(cfg_file))
        assert ind.n_simulations == 1000
        assert ind.slope_threshold == 8.0
        assert ind.regime_weight == 0.25

    def test_missing_sections_use_defaults(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("{}")

        ind = MCMCIndicator.from_config(str(cfg_file))
        assert ind.n_simulations == 25000
        assert ind.slope_threshold == 10.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_too_few_bars_raises(self):
        ind = _indicator()
        df = _make_price_df([100.0, 101.0])
        with pytest.raises(ValueError, match="at least 5"):
            ind.generate_signal("X", df, "1h")

    def test_uppercase_close_column(self):
        ind = _indicator()
        df = _make_price_df(_uptrend_prices(), col="Close")
        result = ind.generate_signal("X", df, "1h")
        assert "slope_degrees" in result

    def test_flat_prices_do_not_crash(self):
        ind = _indicator()
        df = _make_price_df(_flat_prices())
        result = ind.generate_signal("FLAT", df, "1d")
        assert result["direction"] == "NEUTRAL"
