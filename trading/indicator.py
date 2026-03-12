"""MCMC Indicator Core - Signal generation using Monte Carlo simulation with regime detection."""

import math
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Attempt GPU (CuPy) import; fall back to NumPy
try:
    import cupy as cp
    _GPU_AVAILABLE = True
    logger.info("CuPy available — GPU acceleration enabled.")
except ImportError:
    cp = None
    _GPU_AVAILABLE = False
    logger.info("CuPy not available — using NumPy (CPU).")


class MCMCIndicator:
    """
    MCMC-based trading indicator.

    Runs Monte Carlo price-path simulations to produce a directional signal
    (slope in degrees), regime classification, and a composite signal strength.
    """

    DIRECTION_BULLISH = "BULLISH"
    DIRECTION_BEARISH = "BEARISH"
    DIRECTION_NEUTRAL = "NEUTRAL"

    ACTION_BUY = "BUY"
    ACTION_SELL = "SELL"
    ACTION_HOLD = "HOLD"

    def __init__(
        self,
        n_simulations: int = 25000,
        n_steps: int = 30,
        n_regimes: int = 3,
        slope_threshold: float = 10.0,
        band_width_max: float = 0.05,
        regime_confidence_min: float = 0.6,
        mtf_weight: float = 0.3,
        regime_weight: float = 0.3,
        enable_gpu: bool = True,
    ) -> None:
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.n_regimes = n_regimes
        self.slope_threshold = slope_threshold
        self.band_width_max = band_width_max
        self.regime_confidence_min = regime_confidence_min
        self.mtf_weight = mtf_weight
        self.regime_weight = regime_weight
        self.enable_gpu = enable_gpu and _GPU_AVAILABLE

        # Choose array backend
        self._xp = cp if self.enable_gpu else np

    # ------------------------------------------------------------------
    # Class method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str) -> "MCMCIndicator":
        """Load an MCMCIndicator from a YAML configuration file.

        The YAML is expected to follow the layout of config/default.yaml:
          mcmc:
            n_simulations, n_steps, n_regimes
          signal:
            slope_threshold, band_width_max, regime_confidence_min,
            mtf_weight, regime_weight
        """
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        mcmc_cfg = cfg.get("mcmc", {})
        signal_cfg = cfg.get("signal", {})

        return cls(
            n_simulations=mcmc_cfg.get("n_simulations", 25000),
            n_steps=mcmc_cfg.get("n_steps", 30),
            n_regimes=mcmc_cfg.get("n_regimes", 3),
            slope_threshold=signal_cfg.get("slope_threshold", 10.0),
            band_width_max=signal_cfg.get("band_width_max", 0.05),
            regime_confidence_min=signal_cfg.get("regime_confidence_min", 0.6),
            mtf_weight=signal_cfg.get("mtf_weight", 0.3),
            regime_weight=signal_cfg.get("regime_weight", 0.3),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        timeframe: str,
    ) -> dict:
        """Generate a trading signal from historical price data.

        Parameters
        ----------
        ticker:
            Instrument symbol, e.g. ``"AAPL"``.
        price_data:
            DataFrame with at least a ``close`` column (case-insensitive).
            The most-recent row is treated as the current bar.
        timeframe:
            String label such as ``"1m"``, ``"5m"``, ``"1h"``, etc.

        Returns
        -------
        dict
            Keys: ticker, timestamp, timeframe, slope_degrees, direction,
            band_width_pct, regime, regime_confidence, signal_strength,
            suggested_action, forecast_median, forecast_p5, forecast_p95,
            current_price.
        """
        prices = self._extract_prices(price_data)
        if len(prices) < 5:
            raise ValueError(f"Need at least 5 price bars; got {len(prices)}.")

        current_price = float(prices[-1])
        returns = np.diff(np.log(prices))

        # --- run simulation -----------------------------------------------
        try:
            paths = self._run_mcmc(prices)
        except Exception as exc:  # pragma: no cover
            logger.warning("MCMC simulation failed (%s); using bootstrap.", exc)
            paths = self._bootstrap_forecast(prices, returns)

        final_prices = paths[:, -1]

        forecast_median = float(np.percentile(final_prices, 50))
        forecast_p5 = float(np.percentile(final_prices, 5))
        forecast_p95 = float(np.percentile(final_prices, 95))

        # --- derived metrics -----------------------------------------------
        slope_degrees = self._calculate_slope(current_price, forecast_median, prices)
        band_width_pct = (forecast_p95 - forecast_p5) / current_price

        regime, regime_confidence = self._detect_regime(prices)

        direction = self._slope_to_direction(slope_degrees)
        signal_strength = self._calculate_signal_strength(
            slope_degrees, band_width_pct, regime_confidence
        )
        suggested_action = self._strength_to_action(
            direction, signal_strength, regime, regime_confidence
        )

        return {
            "ticker": ticker,
            "timestamp": datetime.utcnow().isoformat(),
            "timeframe": timeframe,
            "slope_degrees": round(slope_degrees, 4),
            "direction": direction,
            "band_width_pct": round(band_width_pct, 6),
            "regime": regime,
            "regime_confidence": round(regime_confidence, 4),
            "signal_strength": round(signal_strength, 4),
            "suggested_action": suggested_action,
            "forecast_median": round(forecast_median, 6),
            "forecast_p5": round(forecast_p5, 6),
            "forecast_p95": round(forecast_p95, 6),
            "current_price": round(current_price, 6),
        }

    # ------------------------------------------------------------------
    # Internal simulation helpers
    # ------------------------------------------------------------------

    def _run_mcmc(self, prices: np.ndarray) -> np.ndarray:
        """Run Monte Carlo GBM simulation.

        Uses log-return statistics estimated from ``prices`` to simulate
        ``n_simulations`` price paths of length ``n_steps``.

        Returns
        -------
        np.ndarray, shape (n_simulations, n_steps + 1)
        """
        xp = self._xp
        returns = np.diff(np.log(prices))

        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        if sigma == 0.0:
            sigma = 1e-8

        current_price = float(prices[-1])

        # GBM drift adjustment
        drift = mu - 0.5 * sigma ** 2

        if self.enable_gpu:
            rand_matrix = xp.random.normal(0.0, 1.0, (self.n_simulations, self.n_steps))
            log_returns = drift + sigma * rand_matrix
            # Cumulative sum to build log-price paths, then exponentiate
            log_paths = xp.cumsum(log_returns, axis=1)
            price_paths = current_price * xp.exp(log_paths)
            # Prepend current price column
            start_col = xp.full((self.n_simulations, 1), current_price)
            full_paths = xp.concatenate([start_col, price_paths], axis=1)
            return xp.asnumpy(full_paths)
        else:
            rand_matrix = np.random.normal(0.0, 1.0, (self.n_simulations, self.n_steps))
            log_returns = drift + sigma * rand_matrix
            log_paths = np.cumsum(log_returns, axis=1)
            price_paths = current_price * np.exp(log_paths)
            start_col = np.full((self.n_simulations, 1), current_price)
            return np.concatenate([start_col, price_paths], axis=1)

    def _bootstrap_forecast(
        self, prices: np.ndarray, returns: np.ndarray
    ) -> np.ndarray:
        """Fallback bootstrap forecast when MCMC fails.

        Resamples historical returns with replacement to build price paths.

        Returns
        -------
        np.ndarray, shape (n_simulations, n_steps + 1)
        """
        if len(returns) == 0:
            returns = np.array([0.0])

        current_price = float(prices[-1])
        indices = np.random.randint(0, len(returns), size=(self.n_simulations, self.n_steps))
        sampled_returns = returns[indices]
        log_paths = np.cumsum(sampled_returns, axis=1)
        price_paths = current_price * np.exp(log_paths)
        start_col = np.full((self.n_simulations, 1), current_price)
        return np.concatenate([start_col, price_paths], axis=1)

    # ------------------------------------------------------------------
    # Regime detection helpers
    # ------------------------------------------------------------------

    def _infer_regimes(self, returns: np.ndarray) -> np.ndarray:
        """Classify each return into a regime state via quantile thresholds.

        With ``n_regimes=3`` the thresholds are the 33rd and 67th percentiles,
        giving states 0 (bear), 1 (neutral), 2 (bull).

        Returns
        -------
        np.ndarray of int, same length as ``returns``.
        """
        if len(returns) == 0:
            return np.array([], dtype=int)

        quantiles = np.linspace(0, 100, self.n_regimes + 1)[1:-1]
        thresholds = np.percentile(returns, quantiles)

        # If all thresholds are identical (degenerate distribution, e.g. flat
        # prices or perfectly uniform returns) assign a single regime based
        # on the sign of the mean return: negative → bear (0), positive → bull
        # (n_regimes-1), zero → neutral (middle).
        if np.all(thresholds == thresholds[0]):
            mean_ret = float(np.mean(returns))
            if mean_ret > 1e-10:
                regime_label = self.n_regimes - 1  # bull
            elif mean_ret < -1e-10:
                regime_label = 0  # bear
            else:
                regime_label = self.n_regimes // 2  # neutral
            return np.full(len(returns), regime_label, dtype=int)

        states = np.zeros(len(returns), dtype=int)
        for i, threshold in enumerate(thresholds):
            states[returns > threshold] = i + 1

        return states

    def _build_transition_matrix(
        self, states: np.ndarray, n_states: int
    ) -> np.ndarray:
        """Estimate a Markov transition matrix from a sequence of states.

        Parameters
        ----------
        states:
            Integer array of state labels in [0, n_states).
        n_states:
            Number of distinct states.

        Returns
        -------
        np.ndarray, shape (n_states, n_states), row-normalised probabilities.
        """
        matrix = np.zeros((n_states, n_states), dtype=float)
        for from_state, to_state in zip(states[:-1], states[1:]):
            matrix[from_state, to_state] += 1.0

        # Row-normalise; uniform fallback for unseen rows
        row_sums = matrix.sum(axis=1, keepdims=True)
        zero_rows = (row_sums == 0).flatten()
        row_sums[zero_rows] = 1.0
        matrix[zero_rows] = 1.0 / n_states
        matrix /= row_sums
        return matrix

    def _detect_regime(self, prices: np.ndarray) -> tuple:
        """Detect the current market regime and confidence.

        Uses the last observed state from quantile-based regime inference.
        Confidence is derived from how extreme the most-recent returns are
        relative to historical quantile boundaries.

        Returns
        -------
        (regime: int, confidence: float)
            regime  — 0=bear, 1=neutral, 2=bull
            confidence — 0-1 float
        """
        if len(prices) < 3:
            return 1, 0.5  # neutral default

        returns = np.diff(np.log(prices))
        states = self._infer_regimes(returns)

        if len(states) == 0:
            return 1, 0.5

        current_regime = int(states[-1])

        # Confidence: fraction of recent window in the same state
        window = min(len(states), max(5, len(states) // 4))
        recent_states = states[-window:]
        confidence = float(np.mean(recent_states == current_regime))

        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        return current_regime, confidence

    # ------------------------------------------------------------------
    # Signal computation helpers
    # ------------------------------------------------------------------

    def _calculate_slope(
        self,
        current: float,
        forecast: float,
        prices: np.ndarray,
    ) -> float:
        """Compute the forecast slope in degrees, normalised by volatility.

        The raw percentage move from ``current`` to ``forecast`` is divided by
        recent price volatility so that a 1-sigma move maps to ~45 degrees,
        then converted to degrees via ``math.atan``.

        Returns
        -------
        float — degrees in [-90, 90].
        """
        if current <= 0:
            return 0.0

        returns = np.diff(np.log(prices)) if len(prices) > 1 else np.array([0.0])
        raw_vol = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0

        # When prices are effectively flat (zero variance), any forecast
        # difference is simulation noise — return 0 to avoid a spurious signal.
        if raw_vol == 0.0:
            return 0.0

        vol = raw_vol
        pct_move = (forecast - current) / current
        normalised = pct_move / vol
        return math.degrees(math.atan(normalised))

    def _calculate_signal_strength(
        self,
        slope: float,
        band_width: float,
        regime_confidence: float,
    ) -> float:
        """Compute a composite signal strength in [0, 1].

        Components:
        - Slope strength: |slope| / 90  (capped at 1)
        - Uncertainty penalty: 1 - min(band_width / band_width_max, 1)
        - Regime confidence weighted by ``regime_weight``
        """
        slope_strength = min(abs(slope) / 90.0, 1.0)
        uncertainty_factor = 1.0 - min(band_width / max(self.band_width_max, 1e-8), 1.0)
        regime_factor = regime_confidence

        # Weighted composite
        slope_wt = 1.0 - self.regime_weight
        strength = (
            slope_wt * slope_strength * uncertainty_factor
            + self.regime_weight * regime_factor
        )
        return max(0.0, min(1.0, float(strength)))

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _slope_to_direction(self, slope_degrees: float) -> str:
        if slope_degrees >= self.slope_threshold:
            return self.DIRECTION_BULLISH
        if slope_degrees <= -self.slope_threshold:
            return self.DIRECTION_BEARISH
        return self.DIRECTION_NEUTRAL

    def _strength_to_action(
        self,
        direction: str,
        signal_strength: float,
        regime: int,
        regime_confidence: float,
    ) -> str:
        """Map direction + strength + regime to a suggested action."""
        if regime_confidence < self.regime_confidence_min:
            return self.ACTION_HOLD

        if direction == self.DIRECTION_BULLISH and regime == 2:
            return self.ACTION_BUY
        if direction == self.DIRECTION_BEARISH and regime == 0:
            return self.ACTION_SELL
        return self.ACTION_HOLD

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def generate_mtf_signal(
        self,
        ticker: str,
        timeframe_data: dict,
    ) -> dict:
        """Generate a multi-timeframe signal by aggregating per-timeframe signals.

        Parameters
        ----------
        ticker:
            Instrument symbol.
        timeframe_data:
            Mapping of timeframe label -> DataFrame with a ``close`` column.

        Returns
        -------
        dict
            All keys from ``generate_signal`` for the first (or shortest)
            timeframe, plus:
            - ``mtf_alignment``: dict mapping each timeframe to its direction.
            - ``mtf_score``: count of timeframes agreeing with the dominant
              direction (range 0..n_timeframes).
            - ``signal_strength``: original value adjusted upward when all
              timeframes align, downward when they disagree.
        """
        if not timeframe_data:
            raise ValueError("timeframe_data must not be empty.")

        per_tf: dict = {}
        for tf, df in timeframe_data.items():
            per_tf[tf] = self.generate_signal(ticker, df, tf)

        # Alignment map: timeframe -> direction
        mtf_alignment = {tf: sig["direction"] for tf, sig in per_tf.items()}

        # Count occurrences of each direction
        direction_counts: dict = {}
        for direction in mtf_alignment.values():
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

        dominant_direction = max(direction_counts, key=lambda d: direction_counts[d])
        mtf_score = direction_counts[dominant_direction]
        n_timeframes = len(timeframe_data)

        # Use the signal from the first timeframe as the base
        first_tf = next(iter(timeframe_data))
        base_signal = dict(per_tf[first_tf])

        # Adjust signal_strength based on alignment ratio
        alignment_ratio = mtf_score / n_timeframes  # 1.0 = full agreement
        raw_strength = base_signal["signal_strength"]
        if alignment_ratio == 1.0:
            adjusted_strength = min(1.0, raw_strength * 1.2)
        elif alignment_ratio >= 0.5:
            adjusted_strength = raw_strength * alignment_ratio
        else:
            adjusted_strength = raw_strength * 0.5 * alignment_ratio

        base_signal["signal_strength"] = round(adjusted_strength, 4)
        base_signal["mtf_alignment"] = mtf_alignment
        base_signal["mtf_score"] = mtf_score
        return base_signal

    @staticmethod
    def _extract_prices(price_data: pd.DataFrame) -> np.ndarray:
        """Extract a 1-D numpy close-price array from a DataFrame."""
        col_map = {c.lower(): c for c in price_data.columns}
        if "close" in col_map:
            return price_data[col_map["close"]].dropna().to_numpy(dtype=float)
        # Fallback: use the first numeric column
        for col in price_data.columns:
            if pd.api.types.is_numeric_dtype(price_data[col]):
                return price_data[col].dropna().to_numpy(dtype=float)
        raise ValueError("price_data must contain a numeric 'close' column.")
