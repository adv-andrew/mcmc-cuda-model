"""
Paper-trading ledger for portfolio mode, driven by the backtest engine.

Every update replays the trading days since the last run through
:class:`backtesting.core_overlay_engine.CoreOverlayEngine`, the same engine
the single-account backtest uses, and saves the state to JSON. You can then
record the price you could really have paid for each option
(``record_fill``). The ledger's most important output is the gap between
those real fills and the model's estimated debit: the bias audit showed the
edge disappears if real fills run ~4-5% above the model.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.core_overlay_engine import CoreOverlayEngine, EngineState
from backtesting.market_data import split_factor
from backtesting.options_pricing import atm_iv_series
from backtesting.portfolio import faber_signal
from trading.features import build_features
from trading.options_swing import SwingConfig, confidence_score, entry_signal, exit_signal

DEFAULT_LEDGER = Path("data/paper/ledger.json")


class PaperLedger:
    def __init__(self, path: Path | str = DEFAULT_LEDGER, cfg: Optional[SwingConfig] = None,
                 initial: float = 100_000.0) -> None:
        self.path = Path(path)
        self.cfg = cfg or SwingConfig.from_yaml()
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.state = EngineState.from_dict(raw["state"])
            self.meta = raw["meta"]
        else:
            self.state = EngineState(cash=float(initial))
            self.meta = {"initial": float(initial), "created": None,
                         "config": {k: (list(v) if isinstance(v, tuple) else v)
                                    for k, v in asdict(self.cfg).items()}}
        self.engine = CoreOverlayEngine(
            self.cfg.structure(), self.cfg.exits(), self.cfg.core_weight,
            self.cfg.risk_per_trade, self.cfg.max_concurrent)

    # ------------------------------------------------------------------
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"meta": self.meta, "state": self.state.to_dict()}, indent=1))
        tmp.replace(self.path)

    def update(self, data: Dict[str, pd.DataFrame], start: Optional[str] = None,
               up_to: Optional[str] = None) -> List[str]:
        """Process every trading day after the last processed one.

        ``data`` maps the core symbol, the strategy symbols and ``"VIX"`` to
        daily OHLCV frames with full history (indicators need ~1 year).
        A new ledger starts at ``start`` (default: the latest bar).
        """
        cfg = self.cfg
        syms = list(cfg.symbols)
        vix = data["VIX"]
        feats = {s: build_features(data[s], vix) for s in syms}
        ivs = {s: atm_iv_series(data[s], vix, data["SPY"]) for s in syms}
        ent = {s: entry_signal(feats[s], cfg) for s in syms}
        ext = {s: exit_signal(feats[s]) for s in syms}
        prio = {s: confidence_score(feats[s]) + feats[s].pullback_atr for s in syms}
        scale = {s: split_factor(data[s].index, s) for s in syms}
        core_close = data[cfg.core_symbol]["Close"]
        core_in = faber_signal(core_close, cfg.faber_months).shift(1, fill_value=False)
        # align to the core's calendar exactly as the backtest wrapper does
        cal = core_close.index
        close_a = {s: feats[s]["close"].reindex(cal) for s in syms}
        iv_a = {s: ivs[s].reindex(cal).ffill() for s in syms}
        ent_a = {s: ent[s].reindex(cal).fillna(False) for s in syms}
        ext_a = {s: ext[s].reindex(cal).fillna(False) for s in syms}
        prio_a = {s: prio[s].reindex(cal).fillna(0.0) for s in syms}
        scale_a = {s: scale[s].reindex(cal).ffill().bfill() for s in syms}

        dates = core_close.index
        if up_to:
            dates = dates[dates <= pd.Timestamp(up_to)]
        if self.state.last_date:
            dates = dates[dates > pd.Timestamp(self.state.last_date)]
        else:
            first = pd.Timestamp(start) if start else dates[-1]
            dates = dates[dates >= first]
            self.meta["created"] = str(dates[0].date()) if len(dates) else None

        events: List[str] = []
        for d in dates:
            n_trades, open_ids = len(self.state.trades), {p["id"] for p in self.state.positions}
            units_before = self.state.core_units
            self.engine.step(
                self.state, d, float(core_close.loc[d]), bool(core_in.loc[d]),
                {s: float(close_a[s].loc[d]) for s in syms},
                {s: float(iv_a[s].loc[d]) for s in syms},
                {s: bool(ent_a[s].loc[d]) for s in syms},
                {s: bool(ext_a[s].loc[d]) for s in syms},
                {s: float(prio_a[s].loc[d]) for s in syms},
                {s: float(scale_a[s].loc[d]) for s in syms})
            day = str(d.date())
            if abs(self.state.core_units - units_before) > 1e-9:
                delta = self.state.core_units - units_before
                events.append(f"{day} CORE {'BUY' if delta > 0 else 'SELL'} {abs(delta):.2f} "
                              f"{cfg.core_symbol} @ {core_close.loc[d]:.2f}")
            for t in self.state.trades[n_trades:]:
                events.append(f"{day} CLOSE {t['id']} after {t['days_held']}d ({t['exit_reason']}): "
                              f"{t['ret_on_risk']:+.1%}, P&L ${t['pnl']:+,.0f}")
            for p in self.state.positions:
                if p["id"] not in open_ids:
                    k = p["pos"]["legs"][0][1]
                    events.append(f"{day} OPEN {p['id']}: buy {p['contracts']:.2f} x {p['symbol']} "
                                  f"{p['pos']['expiry']} {k:g} call, est. debit "
                                  f"${p['model_debit']:.2f}/sh (risk ${p['risk']:,.0f})")
        self.save()
        return events

    def record_fill(self, position_id: str, actual_debit: float) -> None:
        """Record the per-share price you actually paid (or could have paid)."""
        for coll in (self.state.positions, self.state.trades):
            for p in coll:
                if p["id"] == position_id:
                    p["actual_debit"] = float(actual_debit)
                    self.save()
                    return
        raise KeyError(position_id)

    # ------------------------------------------------------------------
    def fill_slippage(self) -> Optional[dict]:
        rows = [p for p in self.state.positions + self.state.trades if p.get("actual_debit")]
        if not rows:
            return None
        gaps = np.array([p["actual_debit"] / p["model_debit"] - 1.0 for p in rows])
        return {"n": len(rows), "mean": float(gaps.mean()), "max": float(gaps.max())}

    def report(self) -> str:
        st = self.state
        if not st.equity:
            return "Ledger is empty - run an update first."
        eq = pd.Series({pd.Timestamp(d): v for d, v in st.equity})
        initial = self.meta["initial"]
        lines = [
            f"Paper ledger {self.path}  (since {self.meta.get('created')}, last update {st.last_date})",
            f"Equity ${eq.iloc[-1]:,.0f}  ({eq.iloc[-1] / initial - 1:+.1%} vs ${initial:,.0f} start)"
            f"  max drawdown {(eq / eq.cummax() - 1).min():+.1%}",
            f"Cash ${st.cash:,.0f}  |  core {st.core_units:.2f} {self.cfg.core_symbol}  |  "
            f"open options {len(st.positions)}",
        ]
        for p in st.positions:
            lines.append(f"   open {p['id']}: {p['contracts']:.2f} x {p['pos']['expiry']} "
                         f"{p['pos']['legs'][0][1]:g} call, held {p['days_held']}d, "
                         f"value ${p['last_value']:,.0f} (cost ${p['risk']:,.0f})")
        if st.trades:
            tf = pd.DataFrame(st.trades)
            lines.append(f"Closed option trades: {len(tf)}  win {(tf.ret_on_risk > 0).mean():.0%}"
                         f"  avg {tf.ret_on_risk.mean():+.1%}  total P&L ${tf.pnl.sum():+,.0f}")
            lines.append("   (backtest expectation: ~63% win, ~+4.5% avg return per trade)")
        slip = self.fill_slippage()
        if slip:
            verdict = ("OK" if slip["mean"] <= 0.02 else
                       "WARNING: above the ~2% the edge can absorb")
            lines.append(f"Real fills vs model debit: {slip['n']} recorded, average "
                         f"{slip['mean']:+.1%} (worst {slip['max']:+.1%}) - {verdict}")
        else:
            lines.append("No real fills recorded yet - use `fill <id> <price>` to track slippage.")
        return "\n".join(lines)
