"""The paper ledger must reproduce the single-account backtest exactly."""

import numpy as np
import pandas as pd
import pytest

from backtesting.market_data import split_factor
from backtesting.options_pricing import atm_iv_series
from backtesting.portfolio import faber_signal, simulate_core_overlay
from trading.features import build_features
from trading.options_swing import SwingConfig, confidence_score, entry_signal, exit_signal
from trading.paper_ledger import PaperLedger


def _ohlc(seed: int, n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0006, 0.009, n)
    ret[rng.choice(np.arange(300, n), 25, replace=False)] -= 0.025  # sharp dips
    close = 100 * np.exp(np.cumsum(ret))
    idx = pd.bdate_range("2020-01-01", periods=n)
    high = close * (1 + rng.uniform(0, 0.006, n))
    low = close * (1 - rng.uniform(0, 0.006, n))
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                         "AdjClose": close, "Volume": 1e6}, index=idx)


@pytest.fixture(scope="module")
def market():
    data = {s: _ohlc(i) for i, s in enumerate(["SPY", "QQQ", "IWM"])}
    idx = data["SPY"].index
    vix = pd.DataFrame({"Close": 18 + 4 * np.sin(np.arange(len(idx)) / 40.0)}, index=idx)
    for c in ("Open", "High", "Low", "AdjClose"):
        vix[c] = vix["Close"]
    vix["Volume"] = 0.0
    data["VIX"] = vix
    return data


def _backtest(data, cfg, start):
    syms = list(cfg.symbols)
    f = {s: build_features(data[s], data["VIX"]) for s in syms}
    iv = {s: atm_iv_series(data[s], data["VIX"], data["SPY"]) for s in syms}
    eq, tr = simulate_core_overlay(
        data["SPY"]["Close"], faber_signal(data["SPY"]["Close"], cfg.faber_months), f, iv,
        {s: entry_signal(f[s], cfg) for s in syms}, {s: exit_signal(f[s]) for s in syms},
        cfg.structure(), cfg.exits(), cfg.core_weight, cfg.risk_per_trade, cfg.max_concurrent,
        {s: confidence_score(f[s]) + f[s].pullback_atr for s in syms},
        {s: split_factor(data[s].index, s) for s in syms}, start=start)
    return eq, tr


def test_ledger_replay_matches_backtest_across_save_and_reload(market, tmp_path):
    cfg = SwingConfig()
    start = "2021-03-01"
    eq_bt, tr_bt = _backtest(market, cfg, start)
    assert len(tr_bt) >= 5, "synthetic market should generate trades"

    path = tmp_path / "ledger.json"
    mid = market["SPY"].index[600]
    PaperLedger(path, cfg, whole_contracts=False).update(market, start=start,
                                                         up_to=str(mid.date()))
    resumed = PaperLedger(path, cfg)  # fresh object from JSON; sizing mode persisted
    resumed.update(market)

    eq_l = pd.Series({pd.Timestamp(d): v for d, v in resumed.state.equity})
    assert eq_l.index.equals(eq_bt.index)
    assert np.allclose(eq_l.to_numpy(), eq_bt.to_numpy(), rtol=0, atol=1e-6)
    closed = pd.DataFrame(resumed.state.trades)
    common = closed[closed.entry_date < str(eq_bt.index[-1].date())]
    assert len(common) == len(tr_bt)
    assert np.allclose(common.ret_on_risk.to_numpy(), tr_bt.ret_on_risk.to_numpy())


def test_update_is_idempotent(market, tmp_path):
    ledger = PaperLedger(tmp_path / "l.json", SwingConfig())
    ledger.update(market, start="2022-06-01")
    n = len(ledger.state.equity)
    assert ledger.update(market) == []
    assert len(ledger.state.equity) == n


def test_fill_tracking_and_report(market, tmp_path):
    ledger = PaperLedger(tmp_path / "l.json", SwingConfig())
    ledger.update(market, start="2021-03-01")
    tr = ledger.state.trades[0]
    ledger.record_fill(tr["id"], tr["model_debit"] * 1.05)
    slip = ledger.fill_slippage()
    assert slip["n"] == 1 and slip["mean"] == pytest.approx(0.05)
    text = ledger.report()
    assert "WARNING" in text and "Equity $" in text
    with pytest.raises(KeyError):
        ledger.record_fill("NOPE-2020-01-01", 1.0)


def test_whole_contract_ledger_buys_integers(market, tmp_path):
    ledger = PaperLedger(tmp_path / "w.json", SwingConfig(), initial=250_000)
    ledger.update(market, start="2021-03-01")
    assert ledger.meta["whole_contracts"] is True
    assert ledger.state.trades, "should have traded"
    for p in ledger.state.positions:
        assert abs(p["contracts"] - round(p["contracts"])) < 1e-9
    for t in ledger.state.trades:
        assert t["ret_on_risk"] >= -1.0


def test_contracts_for():
    from trading.options_swing import contracts_for

    cfg = SwingConfig()  # 6% risk per trade
    assert contracts_for(20.0, 100_000, cfg) == 3      # $6,000 / $2,000
    assert contracts_for(46.0, 100_000, cfg) == 1      # $4,600 fits under $6,000
    assert contracts_for(46.0, 60_000, cfg) == 1       # $4,600 <= 1.5 x $3,600
    assert contracts_for(46.0, 25_000, cfg) == 0       # $4,600 > 1.5 x $1,500 -> skip


def test_ledger_keeps_its_config_and_warns(market, tmp_path):
    from dataclasses import replace

    path = tmp_path / "c.json"
    PaperLedger(path, SwingConfig(), whole_contracts=False).update(market, start="2022-01-03")
    changed = replace(SwingConfig(), risk_per_trade=0.10)
    kept = PaperLedger(path, changed)
    assert kept.cfg.risk_per_trade == SwingConfig().risk_per_trade
    assert kept.config_warning and "NOTE:" in kept.report()
    adopted = PaperLedger(path, changed, adopt_config=True)
    assert adopted.cfg.risk_per_trade == 0.10 and adopted.config_warning is None


def test_old_ledger_without_sizing_flag_stays_fractional(market, tmp_path):
    import json

    path = tmp_path / "old.json"
    PaperLedger(path, SwingConfig(), whole_contracts=False).update(market, start="2022-01-03")
    raw = json.loads(path.read_text())
    raw["meta"].pop("whole_contracts")
    path.write_text(json.dumps(raw))
    assert PaperLedger(path, SwingConfig()).engine.whole_contracts is False
