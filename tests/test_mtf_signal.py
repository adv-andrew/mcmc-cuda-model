"""Tests for MCMCIndicator.generate_mtf_signal (multi-timeframe)."""

import numpy as np
import pandas as pd
import pytest

from trading.indicator import MCMCIndicator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_df(prices, col="close"):
    return pd.DataFrame({col: prices})


def _uptrend(n=60, pct=0.005):
    return np.array([100.0 * (1 + pct) ** i for i in range(n)])


def _downtrend(n=60, pct=0.005):
    return np.array([100.0 * (1 - pct) ** i for i in range(n)])


def _flat(n=60):
    return np.full(n, 100.0)


def _indicator():
    return MCMCIndicator(n_simulations=300, n_steps=10, enable_gpu=False)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestMTFSignalStructure:
    def test_required_keys_present(self):
        ind = _indicator()
        tf_data = {
            "1h": _make_price_df(_uptrend()),
            "4h": _make_price_df(_uptrend()),
        }
        result = ind.generate_mtf_signal("AAPL", tf_data)
        assert "mtf_alignment" in result
        assert "mtf_score" in result
        assert "signal_strength" in result

    def test_mtf_alignment_keys_match_timeframes(self):
        ind = _indicator()
        tf_data = {
            "1h": _make_price_df(_uptrend()),
            "4h": _make_price_df(_uptrend()),
            "1d": _make_price_df(_uptrend()),
        }
        result = ind.generate_mtf_signal("AAPL", tf_data)
        assert set(result["mtf_alignment"].keys()) == {"1h", "4h", "1d"}

    def test_mtf_score_bounded(self):
        ind = _indicator()
        tf_data = {
            "1h": _make_price_df(_uptrend()),
            "4h": _make_price_df(_uptrend()),
        }
        result = ind.generate_mtf_signal("AAPL", tf_data)
        assert 1 <= result["mtf_score"] <= len(tf_data)

    def test_signal_strength_bounded(self):
        ind = _indicator()
        tf_data = {
            "1h": _make_price_df(_uptrend()),
            "4h": _make_price_df(_downtrend()),
        }
        result = ind.generate_mtf_signal("AAPL", tf_data)
        assert 0.0 <= result["signal_strength"] <= 1.0

    def test_ticker_passthrough(self):
        ind = _indicator()
        tf_data = {"1h": _make_price_df(_uptrend())}
        result = ind.generate_mtf_signal("TSLA", tf_data)
        assert result["ticker"] == "TSLA"

    def test_single_timeframe_works(self):
        ind = _indicator()
        tf_data = {"1h": _make_price_df(_uptrend())}
        result = ind.generate_mtf_signal("X", tf_data)
        assert result["mtf_score"] == 1

    def test_empty_timeframe_data_raises(self):
        ind = _indicator()
        with pytest.raises(ValueError, match="must not be empty"):
            ind.generate_mtf_signal("X", {})


# ---------------------------------------------------------------------------
# Score logic
# ---------------------------------------------------------------------------


class TestMTFScore:
    def test_full_agreement_score_equals_n_timeframes(self):
        """All timeframes in the same trend should give max mtf_score."""
        ind = _indicator()
        np.random.seed(0)
        prices = _uptrend(n=60, pct=0.02)
        tf_data = {
            "1h": _make_price_df(prices),
            "4h": _make_price_df(prices),
            "1d": _make_price_df(prices),
        }
        result = ind.generate_mtf_signal("X", tf_data)
        # All three should agree on the same direction
        assert result["mtf_score"] == 3

    def test_alignment_directions_are_valid(self):
        ind = _indicator()
        tf_data = {
            "1h": _make_price_df(_uptrend()),
            "4h": _make_price_df(_downtrend()),
        }
        result = ind.generate_mtf_signal("X", tf_data)
        valid = {"BULLISH", "BEARISH", "NEUTRAL"}
        for direction in result["mtf_alignment"].values():
            assert direction in valid
