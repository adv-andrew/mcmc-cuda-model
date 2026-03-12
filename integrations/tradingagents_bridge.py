"""TradingAgents integration stub with budget tracking."""

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LLM_USAGE_PATH = "data/results/llm_usage.json"


class TradingAgentsBridge:
    """
    Stub integration with TradingAgents LLM validation layer.

    Tracks daily budget and call count. validate_signal agrees with MCMC
    signals (stub implementation — no real LLM calls are made).
    """

    def __init__(
        self,
        daily_budget_usd: float = 2.0,
        max_calls_per_day: int = 3,
        signal_threshold: float = 0.75,
        usage_path: str = LLM_USAGE_PATH,
    ) -> None:
        self.daily_budget_usd = daily_budget_usd
        self.max_calls_per_day = max_calls_per_day
        self.signal_threshold = signal_threshold
        self.usage_path = usage_path

        self._usage = self._load_usage()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_call(self) -> bool:
        """Return True if budget and call-count allow another LLM call today."""
        today = self._today()
        day_usage = self._usage.get(today, {"calls": 0, "cost_usd": 0.0})
        return (
            day_usage["calls"] < self.max_calls_per_day
            and day_usage["cost_usd"] < self.daily_budget_usd
        )

    def should_validate(self, signal_strength: float) -> bool:
        """Return True when signal is strong enough to warrant LLM validation."""
        return signal_strength >= self.signal_threshold

    def validate_signal(
        self,
        ticker: str,
        mcmc_signal: dict,
        price_data=None,
    ) -> dict:
        """
        Validate an MCMC signal using TradingAgents (stub).

        This stub always agrees with the MCMC signal to allow development and
        testing without real LLM API keys or costs.

        Args:
            ticker: Ticker symbol.
            mcmc_signal: Signal dict from MCMCIndicator.generate_signal().
            price_data: Optional DataFrame of price history (unused in stub).

        Returns:
            Validation result dict with keys:
                validated (bool), confidence (float), action (str),
                reasoning (str), agreed_with_mcmc (bool).
        """
        if not self.can_call():
            logger.warning(
                "TradingAgentsBridge: daily budget/call limit reached; skipping validation."
            )
            return {
                "validated": False,
                "confidence": 0.0,
                "action": mcmc_signal.get("suggested_action", "HOLD"),
                "reasoning": "Budget limit reached — validation skipped.",
                "agreed_with_mcmc": True,
            }

        # Stub: agree with MCMC at slightly reduced confidence
        mcmc_action = mcmc_signal.get("suggested_action", "HOLD")
        mcmc_strength = mcmc_signal.get("signal_strength", 0.0)
        stub_confidence = min(mcmc_strength * 0.95, 1.0)

        result = {
            "validated": True,
            "confidence": stub_confidence,
            "action": mcmc_action,
            "reasoning": (
                f"[STUB] TradingAgents agrees with MCMC signal '{mcmc_action}' "
                f"for {ticker} (strength={mcmc_strength:.3f})."
            ),
            "agreed_with_mcmc": True,
        }

        # Estimate cost per call (~$0.01 stub cost)
        estimated_cost = 0.01
        self._record_call(cost_usd=estimated_cost)
        logger.info(
            "TradingAgentsBridge: validated %s signal for %s (confidence=%.3f)",
            mcmc_action,
            ticker,
            stub_confidence,
        )
        return result

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def _today(self) -> str:
        return str(date.today())

    def _load_usage(self) -> dict:
        path = Path(self.usage_path)
        if path.exists():
            try:
                with open(path, "r") as fh:
                    return json.load(fh)
            except Exception as exc:
                logger.warning("Failed to load LLM usage log: %s", exc)
        return {}

    def _save_usage(self) -> None:
        path = Path(self.usage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as fh:
                json.dump(self._usage, fh, indent=2)
        except Exception as exc:
            logger.warning("Failed to save LLM usage log: %s", exc)

    def _record_call(self, cost_usd: float = 0.01) -> None:
        today = self._today()
        if today not in self._usage:
            self._usage[today] = {"calls": 0, "cost_usd": 0.0}
        self._usage[today]["calls"] += 1
        self._usage[today]["cost_usd"] += cost_usd
        self._save_usage()
