"""Tests for SignalCombiner."""

import pytest

from trading.signal_combiner import SignalCombiner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mcmc(action="BUY", strength=0.8):
    return {"suggested_action": action, "signal_strength": strength}


def _llm(action="BUY", confidence=0.7):
    return {"action": action, "confidence": confidence}


def _combiner(**kwargs):
    return SignalCombiner(**kwargs)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestCombineOutputStructure:
    def test_required_keys_present(self):
        c = _combiner()
        result = c.combine(_mcmc())
        assert set(result.keys()) == {"action", "position_scale", "reason"}

    def test_action_is_valid(self):
        c = _combiner()
        result = c.combine(_mcmc())
        assert result["action"] in {"BUY", "SELL", "HOLD"}

    def test_position_scale_between_0_and_1(self):
        c = _combiner()
        result = c.combine(_mcmc())
        assert 0.0 <= result["position_scale"] <= 1.0

    def test_reason_is_string(self):
        c = _combiner()
        result = c.combine(_mcmc())
        assert isinstance(result["reason"], str)


# ---------------------------------------------------------------------------
# MCMC-only rules
# ---------------------------------------------------------------------------


class TestMCMCOnly:
    def test_mcmc_only_returns_half_scale(self):
        c = _combiner(mcmc_only_scale=0.5)
        result = c.combine(_mcmc("BUY", strength=0.9))
        assert result["action"] == "BUY"
        assert result["position_scale"] == pytest.approx(0.5)

    def test_mcmc_sell_only_returns_half_scale(self):
        c = _combiner(mcmc_only_scale=0.5)
        result = c.combine(_mcmc("SELL", strength=0.85))
        assert result["action"] == "SELL"
        assert result["position_scale"] == pytest.approx(0.5)

    def test_weak_mcmc_returns_hold(self):
        c = _combiner(min_mcmc_strength=0.6)
        result = c.combine(_mcmc("BUY", strength=0.3))
        assert result["action"] == "HOLD"
        assert result["position_scale"] == pytest.approx(0.0)

    def test_mcmc_hold_action_returns_hold(self):
        c = _combiner()
        result = c.combine(_mcmc("HOLD", strength=0.9))
        assert result["action"] == "HOLD"

    def test_custom_mcmc_only_scale(self):
        c = _combiner(mcmc_only_scale=0.4)
        result = c.combine(_mcmc("BUY", strength=0.9))
        assert result["position_scale"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# MCMC + LLM agreement
# ---------------------------------------------------------------------------


class TestAgreement:
    def test_agreement_returns_full_scale(self):
        c = _combiner(agreement_scale=1.0)
        result = c.combine(_mcmc("BUY", strength=0.9), _llm("BUY", confidence=0.8))
        assert result["action"] == "BUY"
        assert result["position_scale"] == pytest.approx(1.0)

    def test_sell_agreement_full_scale(self):
        c = _combiner(agreement_scale=1.0)
        result = c.combine(_mcmc("SELL", strength=0.9), _llm("SELL", confidence=0.8))
        assert result["action"] == "SELL"
        assert result["position_scale"] == pytest.approx(1.0)

    def test_custom_agreement_scale(self):
        c = _combiner(agreement_scale=0.75)
        result = c.combine(_mcmc("BUY", strength=0.9), _llm("BUY", confidence=0.8))
        assert result["position_scale"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# MCMC + LLM disagreement
# ---------------------------------------------------------------------------


class TestDisagreement:
    def test_disagreement_returns_hold_zero_scale(self):
        c = _combiner()
        result = c.combine(_mcmc("BUY", strength=0.9), _llm("SELL", confidence=0.8))
        assert result["action"] == "HOLD"
        assert result["position_scale"] == pytest.approx(0.0)

    def test_llm_hold_vs_mcmc_buy_is_disagreement(self):
        c = _combiner()
        result = c.combine(_mcmc("BUY", strength=0.9), _llm("HOLD", confidence=0.8))
        assert result["action"] == "HOLD"


# ---------------------------------------------------------------------------
# LLM confidence gate
# ---------------------------------------------------------------------------


class TestLLMConfidenceGate:
    def test_low_llm_confidence_falls_back_to_mcmc_only(self):
        c = _combiner(min_llm_confidence=0.5, mcmc_only_scale=0.5)
        result = c.combine(
            _mcmc("BUY", strength=0.9),
            _llm("BUY", confidence=0.3),
        )
        assert result["action"] == "BUY"
        assert result["position_scale"] == pytest.approx(0.5)

    def test_exact_threshold_passes(self):
        c = _combiner(min_llm_confidence=0.5, agreement_scale=1.0)
        result = c.combine(
            _mcmc("BUY", strength=0.9),
            _llm("BUY", confidence=0.5),
        )
        assert result["action"] == "BUY"
        assert result["position_scale"] == pytest.approx(1.0)
