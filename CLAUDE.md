# MCMC Trading System - Project Conventions

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Architecture

The project is organized into modular components:

- **trading/**: Signal generation using MCMC indicators
- **brokers/**: Broker interface and implementations
- **backtesting/**: GPU-accelerated backtesting engine
- **optimization/**: Parameter optimization with Optuna
- **integrations/**: External system integrations (LLM agents, data feeds)
- **config/**: Configuration files (default.yaml, tickers.yaml)
- **data/**: Cache and results directories
- **tests/**: Unit and integration tests

## Conventions

### Configuration
- All configuration lives in `config/default.yaml` and `config/tickers.yaml`
- Use YAML for human-readable configs
- Environment variables can override config values

### Code Style
- Follow PEP 8
- Use type hints for function signatures
- Add docstrings to modules and public functions
- Use descriptive variable names

### Testing
- Place tests in the `tests/` directory
- Use pytest for test execution
- Aim for >80% code coverage on core modules

### Commits
- Use conventional commits: feat:, fix:, refactor:, test:, docs:
- Keep commits focused and atomic
- Reference tasks/issues in commit messages when applicable

## Key Configuration Parameters

See `config/default.yaml` for:
- MCMC simulation parameters (n_simulations, n_steps, n_regimes)
- Signal generation thresholds
- Risk management settings
- Backtest parameters
- LLM integration budget and limits
