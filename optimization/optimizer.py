"""Optuna-based parameter optimizer for the MCMC Trading System."""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """
    Bayesian hyper-parameter search using Optuna's TPE sampler.

    Parameters
    ----------
    data:
        Mapping of ticker -> DataFrame passed through to the fitness function.
    initial_cash:
        Starting capital used in each trial simulation.
    min_trades:
        Minimum trades required; fewer incurs a fitness penalty.
    """

    # Search-space definition: name -> (type, low, high[, step])
    _SEARCH_SPACE = {
        "slope_threshold":    ("float", 5.0,    20.0,  None),
        "n_regimes":          ("int",   2,       4,     None),
        "n_simulations":      ("int",   10_000,  50_000, 10_000),
        "signal_strength_min":("float", 0.5,    0.9,   None),
        "position_size_pct":  ("float", 0.05,   0.25,  None),
        "stop_loss_pct":      ("float", 0.01,   0.05,  None),
        "take_profit_pct":    ("float", 0.02,   0.10,  None),
    }

    def __init__(
        self,
        data: dict,
        initial_cash: float = 100_000.0,
        min_trades: int = 100,
    ) -> None:
        self.data = data
        self.initial_cash = initial_cash
        self.min_trades = min_trades

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        n_trials: int = 500,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
    ) -> tuple[dict, float]:
        """
        Run the Optuna optimization study.

        Parameters
        ----------
        n_trials:
            Maximum number of trials to evaluate.
        timeout:
            Wall-clock time limit in seconds (``None`` = no limit).
        n_jobs:
            Number of parallel jobs (passed to ``study.optimize``).

        Returns
        -------
        (best_params, best_value)
            The parameter dict and its corresponding fitness score.
        """
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "optuna is required for ParameterOptimizer. "
                "Install it with: pip install optuna"
            ) from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=False,
        )

        best_params = study.best_trial.params
        best_value = study.best_value
        logger.info(
            "Optimization complete: best_value=%.4f, best_params=%s",
            best_value,
            best_params,
        )
        return best_params, best_value

    def save_best_params(
        self,
        params: dict,
        value: float,
        path: str = "config/best_params.json",
    ) -> None:
        """
        Persist the best parameters to a JSON file.

        Parameters
        ----------
        params:
            Parameter dict to save.
        value:
            The corresponding fitness value (stored for reference).
        path:
            Output file path (directories are created if needed).
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {"params": params, "fitness_value": value}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Best params saved to %s", path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _objective(self, trial) -> float:
        """Optuna objective: sample parameters and evaluate fitness."""
        from optimization.fitness import calculate_fitness

        params: dict = {}
        for name, spec in self._SEARCH_SPACE.items():
            kind = spec[0]
            low, high = spec[1], spec[2]
            step = spec[3]
            if kind == "float":
                params[name] = trial.suggest_float(name, low, high)
            elif kind == "int":
                if step is not None:
                    params[name] = trial.suggest_int(name, int(low), int(high), step=int(step))
                else:
                    params[name] = trial.suggest_int(name, int(low), int(high))

        return calculate_fitness(
            data=self.data,
            params=params,
            initial_cash=self.initial_cash,
            min_trades=self.min_trades,
        )
