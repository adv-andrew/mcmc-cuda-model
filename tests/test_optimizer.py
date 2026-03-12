"""Minimal tests for optimization.fitness and optimization.optimizer."""

import numpy as np
import pandas as pd
import pytest


def _make_data(n: int = 60) -> dict:
    np.random.seed(1)
    prices = 100.0 * np.cumprod(1 + np.random.normal(0.001, 0.01, n))
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame(
        {"Close": prices, "Open": prices, "High": prices, "Low": prices, "Volume": 1e6},
        index=idx,
    )
    return {"OPT": df}


# ---------------------------------------------------------------------------
# fitness.py
# ---------------------------------------------------------------------------

class TestCalculateFitness:
    def test_returns_float(self):
        from optimization.fitness import calculate_fitness

        params = {
            "slope_threshold": 10.0,
            "n_regimes": 3,
            "n_simulations": 10_000,
            "signal_strength_min": 0.6,
            "position_size_pct": 0.10,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
        }
        score = calculate_fitness(_make_data(), params, initial_cash=10_000, min_trades=5)
        assert isinstance(score, float)

    def test_penalty_for_zero_trades(self):
        """With a very high signal_strength_min no trades should fire -> penalty applied."""
        from optimization.fitness import calculate_fitness

        params = {
            "slope_threshold": 10.0,
            "n_regimes": 3,
            "n_simulations": 10_000,
            "signal_strength_min": 0.99,  # nearly impossible to satisfy
            "position_size_pct": 0.10,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
        }
        score = calculate_fitness(_make_data(), params, initial_cash=10_000, min_trades=100)
        # The trade-shortfall penalty is -50 when n_trades == 0
        assert score <= -49.0


# ---------------------------------------------------------------------------
# optimizer.py
# ---------------------------------------------------------------------------

class TestParameterOptimizerInit:
    def test_init_stores_attributes(self):
        from optimization.optimizer import ParameterOptimizer

        opt = ParameterOptimizer(data=_make_data(), initial_cash=20_000, min_trades=10)
        assert opt.initial_cash == 20_000
        assert opt.min_trades == 10

    def test_search_space_keys_present(self):
        from optimization.optimizer import ParameterOptimizer

        opt = ParameterOptimizer(data=_make_data())
        expected_keys = {
            "slope_threshold",
            "n_regimes",
            "n_simulations",
            "signal_strength_min",
            "position_size_pct",
            "stop_loss_pct",
            "take_profit_pct",
        }
        assert expected_keys == set(opt._SEARCH_SPACE.keys())


class TestParameterOptimizerOptimize:
    def test_optimize_returns_params_and_value(self):
        pytest.importorskip("optuna")
        from optimization.optimizer import ParameterOptimizer

        opt = ParameterOptimizer(data=_make_data(n=80), initial_cash=10_000, min_trades=5)
        best_params, best_value = opt.optimize(n_trials=3, n_jobs=1)
        assert isinstance(best_params, dict)
        assert isinstance(best_value, float)
        assert set(best_params.keys()) == set(opt._SEARCH_SPACE.keys())

    def test_save_best_params(self, tmp_path):
        from optimization.optimizer import ParameterOptimizer

        opt = ParameterOptimizer(data=_make_data())
        params = {"slope_threshold": 12.0, "n_regimes": 3}
        out_path = str(tmp_path / "params.json")
        opt.save_best_params(params, value=5.0, path=out_path)

        import json
        with open(out_path) as fh:
            payload = json.load(fh)
        assert payload["params"] == params
        assert payload["fitness_value"] == 5.0
