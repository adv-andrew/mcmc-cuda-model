"""OpenAI GPT integration for validating MCMC trading signals."""

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LLM_USAGE_PATH = "data/results/llm_usage.json"


class OpenAIValidator:
    """
    Validates MCMC trading signals using OpenAI GPT-4.

    Sends price context and MCMC signal to GPT for a second opinion.
    Can agree, disagree, or suggest modifications to the signal.
    """

    def __init__(
        self,
        daily_budget_usd: float = 2.0,
        max_calls_per_day: int = 100,
        model: str = "gpt-4o-mini",
        usage_path: str = LLM_USAGE_PATH,
    ) -> None:
        self.daily_budget_usd = daily_budget_usd
        self.max_calls_per_day = max_calls_per_day
        self.model = model
        self.usage_path = usage_path
        self._usage = self._load_usage()

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment")

        self._client = None

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def can_call(self) -> bool:
        """Return True if budget and call-count allow another LLM call today."""
        today = self._today()
        day_usage = self._usage.get(today, {"calls": 0, "cost_usd": 0.0})
        return (
            day_usage["calls"] < self.max_calls_per_day
            and day_usage["cost_usd"] < self.daily_budget_usd
        )

    def validate_signal(
        self,
        ticker: str,
        mcmc_signal: dict,
        price_data: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Validate an MCMC signal using OpenAI GPT.

        Args:
            ticker: Ticker symbol.
            mcmc_signal: Signal dict from MCMCIndicator.generate_signal().
            price_data: DataFrame of recent price history.

        Returns:
            Validation result dict with keys:
                validated (bool), confidence (float), action (str),
                reasoning (str), agreed_with_mcmc (bool).
        """
        if not self.can_call():
            logger.warning("OpenAI daily budget/call limit reached; skipping validation.")
            return {
                "validated": False,
                "confidence": 0.0,
                "action": mcmc_signal.get("suggested_action", "HOLD"),
                "reasoning": "Budget limit reached — validation skipped.",
                "agreed_with_mcmc": True,
            }

        if not self.api_key:
            logger.warning("No OpenAI API key configured; using MCMC signal as-is.")
            return {
                "validated": False,
                "confidence": mcmc_signal.get("signal_strength", 0.0),
                "action": mcmc_signal.get("suggested_action", "HOLD"),
                "reasoning": "No API key — falling back to MCMC only.",
                "agreed_with_mcmc": True,
            }

        # Build context for GPT
        price_context = self._build_price_context(price_data) if price_data is not None else "No price data provided."

        prompt = self._build_prompt(ticker, mcmc_signal, price_context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300,
            )

            # Parse response
            content = response.choices[0].message.content
            result = self._parse_response(content, mcmc_signal)

            # Track usage
            usage = response.usage
            cost = self._estimate_cost(usage.prompt_tokens, usage.completion_tokens)
            self._record_call(cost_usd=cost)

            logger.info(
                "OpenAI validated %s: %s (confidence=%.2f, agreed=%s, cost=$%.4f)",
                ticker, result["action"], result["confidence"],
                result["agreed_with_mcmc"], cost
            )

            return result

        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            return {
                "validated": False,
                "confidence": mcmc_signal.get("signal_strength", 0.0),
                "action": mcmc_signal.get("suggested_action", "HOLD"),
                "reasoning": f"API error: {exc}",
                "agreed_with_mcmc": True,
            }

    def _system_prompt(self) -> str:
        return """You validate MCMC trading signals. The MCMC model runs 25,000 Monte Carlo simulations and has backtested at +6% returns.

Respond with JSON only:
{"action": "BUY"|"SELL"|"HOLD", "confidence": 0.0-1.0, "reasoning": "one sentence"}

RULES:
1. DEFAULT: Match the MCMC suggested_action. Only override if you have STRONG evidence.
2. OVERRIDE ONLY when you see CLEAR reversals: price dropped >8% then bounced >3% (V-reversal), or triple tested same level.
3. If MCMC says BUY/SELL with strength >0.7, your confidence should be at least 0.65.
4. Never say HOLD just because "uncertain" - the MCMC already accounts for uncertainty.

You are a CONFIRMER, not a skeptic. Trust the math. Override rate should be <15%."""

    def _build_prompt(self, ticker: str, mcmc_signal: dict, price_context: str) -> str:
        return f"""Validate this MCMC trading signal for {ticker}:

MCMC Signal:
- Direction: {mcmc_signal.get('direction', 'UNKNOWN')}
- Suggested Action: {mcmc_signal.get('suggested_action', 'HOLD')}
- Signal Strength: {mcmc_signal.get('signal_strength', 0):.3f}
- Slope (degrees): {mcmc_signal.get('slope_degrees', 0):.1f}
- Regime: {mcmc_signal.get('regime', 'UNKNOWN')} (0=bear, 1=neutral, 2=bull)
- Regime Confidence: {mcmc_signal.get('regime_confidence', 0):.2f}
- Current Price: ${mcmc_signal.get('current_price', 0):.2f}
- 30-day Forecast Median: ${mcmc_signal.get('forecast_median', 0):.2f}
- Forecast Range: ${mcmc_signal.get('forecast_p5', 0):.2f} - ${mcmc_signal.get('forecast_p95', 0):.2f}

Recent Price Action:
{price_context}

Respond with JSON only."""

    def _build_price_context(self, df: pd.DataFrame, n_days: int = 10) -> str:
        """Build a summary of recent price action."""
        if df is None or df.empty:
            return "No data"

        df = df.tail(n_days)

        lines = []
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)[:10]
            close = row.get('Close', row.get('close', 0))
            high = row.get('High', row.get('high', close))
            low = row.get('Low', row.get('low', close))
            lines.append(f"{date_str}: ${close:.2f} (H:{high:.2f} L:{low:.2f})")

        # Add summary stats
        closes = df['Close'] if 'Close' in df.columns else df['close']
        pct_change = ((closes.iloc[-1] / closes.iloc[0]) - 1) * 100
        volatility = closes.pct_change().std() * 100

        summary = f"\n{n_days}-day change: {pct_change:+.1f}%, Daily volatility: {volatility:.1f}%"

        return "\n".join(lines) + summary

    def _parse_response(self, content: str, mcmc_signal: dict) -> dict:
        """Parse GPT response JSON."""
        try:
            # Try to extract JSON from response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content)

            action = data.get("action", "HOLD").upper()
            if action not in ["BUY", "SELL", "HOLD"]:
                action = "HOLD"

            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            reasoning = data.get("reasoning", "No reasoning provided.")

            mcmc_action = mcmc_signal.get("suggested_action", "HOLD")
            agreed = (action == mcmc_action)

            return {
                "validated": True,
                "confidence": confidence,
                "action": action,
                "reasoning": reasoning,
                "agreed_with_mcmc": agreed,
            }

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Failed to parse GPT response: %s", exc)
            return {
                "validated": False,
                "confidence": mcmc_signal.get("signal_strength", 0.0),
                "action": mcmc_signal.get("suggested_action", "HOLD"),
                "reasoning": f"Parse error: {content[:100]}",
                "agreed_with_mcmc": True,
            }

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on model pricing."""
        # GPT-4o-mini pricing (as of 2024)
        if "gpt-4o-mini" in self.model:
            return (prompt_tokens * 0.00015 + completion_tokens * 0.0006) / 1000
        # GPT-4o pricing
        elif "gpt-4o" in self.model:
            return (prompt_tokens * 0.005 + completion_tokens * 0.015) / 1000
        # Default estimate
        return 0.001

    def _today(self) -> str:
        return str(date.today())

    def _load_usage(self) -> dict:
        path = Path(self.usage_path)
        if path.exists():
            try:
                with open(path, "r") as fh:
                    return json.load(fh)
            except Exception:
                pass
        return {}

    def _save_usage(self) -> None:
        path = Path(self.usage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as fh:
                json.dump(self._usage, fh, indent=2)
        except Exception as exc:
            logger.warning("Failed to save LLM usage log: %s", exc)

    def _record_call(self, cost_usd: float = 0.001) -> None:
        today = self._today()
        if today not in self._usage:
            self._usage[today] = {"calls": 0, "cost_usd": 0.0}
        self._usage[today]["calls"] += 1
        self._usage[today]["cost_usd"] += cost_usd
        self._save_usage()
