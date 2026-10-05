"""
Options Swing Strategy: buy the dip in an uptrend with in-the-money calls.

Research summary (see docs/OPTIONS_SWING_STRATEGY.md and scripts/research_*):

- On SPY/QQQ/IWM, short-term oversold readings *inside* a long-term uptrend
  are followed by above-average 3-5 day returns, in-sample (2011-2019) and
  out-of-sample (2020-2026). Momentum-continuation and bearish setups are
  not, which is why the old trend-following MCMC slope had negative skill.
- Of the option structures tested with realistic pricing (VIX-based IV,
  skew, spreads, commissions), a single in-the-money call (~0.70 delta,
  ~30 DTE) is the most robust: it was profitable on untouched 1991-2009
  S&P 500 data and barely affected by the skew assumption. The 0.55/0.30
  call debit spread scored higher on 2011-2026 but its edge depended on how
  the short call is priced and faded on 1991-2009 (scripts/research_bias_checks.py).
  Credit spreads win more often but lose money after costs on 3-5 day holds.
- A pullback of at least 1.5 ATR from the 5-day high is the confirmation
  filter with the strongest in-sample evidence. Multi-timeframe alignment
  (21-day / weekly / monthly up while the 5-day move is down) and a calm
  volatility regime raise the confidence score.

Rules
-----
Entry (at the close, signal computed a few minutes before the bell):
    close > SMA(200)  AND  pullback >= 1.5 ATR from the 5-day high  AND
    any of: RSI(2) < 10 | (IBS < 0.25 and RSI(2) < 15) | close < lower
    Bollinger(20, 2) | pullback >= 2 ATR
Position:
    Buy a ~0.70-delta call, nearest Friday expiry >= 30 calendar days out
    (optionally a 0.55/0.30 call debit spread via ``structure_kind``).
Exit (checked at each close):
    +60% gain (spreads: of max profit)  |  close > SMA(5) after >= 3 trading days held  |
    7 trading days held  |  expiry
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

from backtesting.options_backtest import ExitRules, StructureSpec, build_legs
from backtesting.options_pricing import (
    CostModel,
    SkewModel,
    bs_delta,
    open_position,
    risk_free_rate,
)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


@dataclass
class SwingConfig:
    symbols: tuple = ("SPY", "QQQ", "IWM")
    # entry
    rsi2_max: float = 10.0
    ibs_max: float = 0.25
    ibs_rsi2_max: float = 15.0
    bb_z_max: float = -2.0
    min_pullback_atr: float = 1.5
    deep_pullback_atr: float = 2.0
    require_weekly_trend: bool = False
    # structure: "long_call" (default, most robust) or "call_debit_spread"
    structure_kind: str = "long_call"
    dte: int = 30
    long_delta: float = 0.70
    short_delta: float = 0.30  # only used by call_debit_spread
    # exits
    min_hold: int = 3
    max_hold: int = 7
    profit_target: float = 0.60
    # portfolio
    risk_per_trade: float = 0.05
    max_concurrent: int = 2
    # confidence tiers (score is 40 / 70 / 100)
    high_confidence: int = 100
    medium_confidence: int = 70

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "SwingConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            raw = (yaml.safe_load(fh) or {}).get("options_swing", {}) or {}
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        if "symbols" in known:
            known["symbols"] = tuple(known["symbols"])
        return cls(**known)

    def structure(self) -> StructureSpec:
        if self.structure_kind not in ("long_call", "call_debit_spread"):
            raise ValueError(f"unsupported structure_kind {self.structure_kind!r}")
        return StructureSpec(self.structure_kind, dte=self.dte,
                             long_delta=self.long_delta, short_delta=self.short_delta)

    def exits(self) -> ExitRules:
        return ExitRules(min_hold=self.min_hold, max_hold=self.max_hold,
                         profit_target=self.profit_target, stop_loss=None, signal_exit=True)


# ----------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------

def dip_trigger(f: pd.DataFrame, cfg: SwingConfig) -> pd.Series:
    """Any of the oversold triggers (before trend / size confirmation)."""
    return (
        (f.rsi2 < cfg.rsi2_max)
        | ((f.ibs < cfg.ibs_max) & (f.rsi2 < cfg.ibs_rsi2_max))
        | (f.bb_z < cfg.bb_z_max)
        | (f.pullback_atr >= cfg.deep_pullback_atr)
    )


def entry_signal(f: pd.DataFrame, cfg: SwingConfig) -> pd.Series:
    """Boolean entry series: dip trigger, in an uptrend, of meaningful size."""
    sig = f.above200 & (f.pullback_atr >= cfg.min_pullback_atr) & dip_trigger(f, cfg)
    if cfg.require_weekly_trend:
        sig &= f.weekly_up
    return sig.fillna(False).astype(bool)


def exit_signal(f: pd.DataFrame) -> pd.Series:
    """Connors-style recovery exit: close back above the 5-day SMA."""
    return (f.close > f.sma5).fillna(False)


def confidence_score(f: pd.DataFrame) -> pd.Series:
    """0-100 score for a qualifying setup.

    Only two conditions separated better from worse strategy trades with the
    same sign in *both* 2011-2019 and 2020-2026 (most others flipped sign,
    i.e. were noise):

    =================================================================  ======
    component                                                          points
    =================================================================  ======
    base (setup qualifies)                                                 40
    timeframes aligned: 21-day, weekly and monthly trend all up while      30
    the 5-day move is down (``mtf_score >= 2``) - a pullback *within*
    every higher timeframe
    calm regime: 20-day realized vol < 15%                                 30
    =================================================================  ======

    Because these were identified with both periods in view, treat the tier
    statistics as descriptive, not as independent out-of-sample evidence.
    """
    s = pd.Series(40.0, index=f.index)
    s += np.where(f.mtf_score >= 2, 30, 0)
    s += np.where(f.rv20 < 0.15, 30, 0)
    return s.clip(0, 100)


def confidence_tier(score: float, cfg: SwingConfig) -> str:
    if score >= cfg.high_confidence:
        return "HIGH"
    if score >= cfg.medium_confidence:
        return "MEDIUM"
    return "LOW"


# ----------------------------------------------------------------------
# Trade ticket for the latest bar
# ----------------------------------------------------------------------

@dataclass
class TradeTicket:
    symbol: str
    date: str
    spot: float
    confidence: int
    tier: str
    long_strike: float
    short_strike: Optional[float]  # None for a single call
    expiry: str
    est_debit: float
    max_profit: Optional[float]    # None (unbounded) for a single call
    take_profit_value: float
    atm_iv: float
    long_delta: float
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def next_expiry(date: pd.Timestamp, dte: int) -> pd.Timestamp:
    """Nearest Friday on or after ``date + dte`` calendar days (weekly expiries)."""
    target = pd.Timestamp(date) + pd.Timedelta(days=dte)
    return target + pd.Timedelta(days=(4 - target.dayofweek) % 7)


def build_ticket(
    symbol: str, f: pd.DataFrame, atm_iv: float, cfg: SwingConfig,
    skew: Optional[SkewModel] = None,
) -> TradeTicket:
    """Concrete option order (single call or call spread) for the latest row of ``f``."""
    skew = skew or SkewModel()
    row = f.iloc[-1]
    date = f.index[-1]
    spot = float(row.close)
    expiry = next_expiry(date, cfg.dte)
    dte = (expiry - date).days
    t = dte / 365.0
    rate = risk_free_rate(date)
    legs = build_legs(cfg.structure(), +1, spot, t, atm_iv, rate, skew, is_etf=True)
    pos = open_position(legs, spot, date, dte, atm_iv, skew, CostModel.etf())
    score = int(confidence_score(f.iloc[[-1]]).iloc[0])
    reasons = []
    if row.rsi2 < 10:
        reasons.append(f"RSI(2) {row.rsi2:.1f}")
    if row.ibs < cfg.ibs_max:
        reasons.append(f"IBS {row.ibs:.2f}")
    if row.bb_z < cfg.bb_z_max:
        reasons.append(f"Bollinger z {row.bb_z:.2f}")
    reasons.append(f"pullback {row.pullback_atr:.2f} ATR")
    reasons.append("timeframes aligned (21d/weekly/monthly up)" if row.mtf_score >= 2
                   else "higher timeframes mixed")
    reasons.append(f"realized vol {row.rv20:.0%}" + (" (calm)" if row.rv20 < 0.15 else ""))
    if "vix" in f and not math.isnan(row.get("vix", float("nan"))):
        reasons.append(f"VIX {row.vix:.1f} ({row.vix_ratio10:.2f}x 10d avg)")
    return TradeTicket(
        symbol=symbol,
        date=str(date.date()),
        spot=round(spot, 2),
        confidence=score,
        tier=confidence_tier(score, cfg),
        long_strike=float(legs[0].strike),
        short_strike=float(legs[1].strike) if len(legs) > 1 else None,
        expiry=str(expiry.date()),
        est_debit=round(pos.max_loss, 2),
        max_profit=None if math.isinf(pos.max_profit) else round(pos.max_profit, 2),
        take_profit_value=round(pos.max_loss + cfg.profit_target * (
            pos.max_loss if math.isinf(pos.max_profit) else pos.max_profit), 2),
        atm_iv=round(atm_iv, 4),
        long_delta=round(bs_delta(spot, legs[0].strike, t, atm_iv, rate, True), 2),
        reasons=reasons,
    )


def scan(
    features: Dict[str, pd.DataFrame], atm_iv: Dict[str, pd.Series], cfg: SwingConfig
) -> Dict[str, dict]:
    """Evaluate the latest bar of every symbol; return status and tickets."""
    out: Dict[str, dict] = {}
    for sym, f in features.items():
        row = f.iloc[-1]
        sig = bool(entry_signal(f.iloc[-260:], cfg).iloc[-1])
        status = {
            "date": str(f.index[-1].date()),
            "close": float(row.close),
            "signal": sig,
            "above200": bool(row.above200),
            "rsi2": float(row.rsi2),
            "ibs": float(row.ibs),
            "pullback_atr": float(row.pullback_atr),
            "weekly_up": bool(row.weekly_up),
            "mtf_score": float(row.mtf_score),
            "exit_signal_if_held": bool(row.close > row.sma5),
        }
        if sig:
            status["ticket"] = build_ticket(sym, f, float(atm_iv[sym].iloc[-1]), cfg).to_dict()
        out[sym] = status
    return out
