"""SignalCombiner — fuses MCMC indicator signals with optional LLM signals."""


class SignalCombiner:
    """Combine an MCMC signal with an optional LLM signal into a trade decision.

    Parameters
    ----------
    mcmc_only_scale:
        Position scale when only the MCMC signal is available (default 0.5).
    agreement_scale:
        Position scale when MCMC and LLM agree (default 1.0).
    min_mcmc_strength:
        Minimum ``signal_strength`` required for the MCMC signal to be acted
        upon (default 0.6).
    min_llm_confidence:
        Minimum confidence required from the LLM signal (default 0.5).
    """

    ACTION_BUY = "BUY"
    ACTION_SELL = "SELL"
    ACTION_HOLD = "HOLD"

    def __init__(
        self,
        mcmc_only_scale: float = 0.5,
        agreement_scale: float = 1.0,
        min_mcmc_strength: float = 0.6,
        min_llm_confidence: float = 0.5,
    ) -> None:
        self.mcmc_only_scale = mcmc_only_scale
        self.agreement_scale = agreement_scale
        self.min_mcmc_strength = min_mcmc_strength
        self.min_llm_confidence = min_llm_confidence

    def combine(self, mcmc_signal: dict, llm_signal: dict | None = None) -> dict:
        """Produce a combined trade decision.

        Parameters
        ----------
        mcmc_signal:
            Output of ``MCMCIndicator.generate_signal`` (or
            ``generate_mtf_signal``).  Must contain ``suggested_action`` and
            ``signal_strength``.
        llm_signal:
            Optional dict with at least ``action`` (BUY/SELL/HOLD) and
            ``confidence`` (0-1 float).  Pass ``None`` to use MCMC alone.

        Returns
        -------
        dict with keys:
            - ``action``: BUY | SELL | HOLD
            - ``position_scale``: 0.0 – 1.0 multiplier for position sizing
            - ``reason``: human-readable explanation string
        """
        mcmc_action = mcmc_signal.get("suggested_action", self.ACTION_HOLD)
        mcmc_strength = float(mcmc_signal.get("signal_strength", 0.0))

        # MCMC strength gate
        if mcmc_strength < self.min_mcmc_strength or mcmc_action == self.ACTION_HOLD:
            return {
                "action": self.ACTION_HOLD,
                "position_scale": 0.0,
                "reason": (
                    f"MCMC signal too weak or HOLD "
                    f"(strength={mcmc_strength:.3f}, "
                    f"min={self.min_mcmc_strength:.3f})"
                ),
            }

        # No LLM signal — use MCMC alone at reduced scale
        if llm_signal is None:
            return {
                "action": mcmc_action,
                "position_scale": self.mcmc_only_scale,
                "reason": (
                    f"MCMC only: {mcmc_action} with strength {mcmc_strength:.3f} "
                    f"(scale={self.mcmc_only_scale})"
                ),
            }

        llm_action = llm_signal.get("action", self.ACTION_HOLD)
        llm_confidence = float(llm_signal.get("confidence", 0.0))

        # LLM confidence gate
        if llm_confidence < self.min_llm_confidence:
            return {
                "action": mcmc_action,
                "position_scale": self.mcmc_only_scale,
                "reason": (
                    f"LLM confidence too low ({llm_confidence:.3f} < "
                    f"{self.min_llm_confidence:.3f}); using MCMC only"
                ),
            }

        # Agreement check
        if llm_action == mcmc_action:
            return {
                "action": mcmc_action,
                "position_scale": self.agreement_scale,
                "reason": (
                    f"MCMC+LLM agree: {mcmc_action} "
                    f"(mcmc_strength={mcmc_strength:.3f}, "
                    f"llm_confidence={llm_confidence:.3f}, "
                    f"scale={self.agreement_scale})"
                ),
            }

        # Disagreement — hold
        return {
            "action": self.ACTION_HOLD,
            "position_scale": 0.0,
            "reason": (
                f"MCMC ({mcmc_action}) and LLM ({llm_action}) disagree; HOLD"
            ),
        }
