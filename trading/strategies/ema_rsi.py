"""
trading/strategies/ema_rsi.py — Stratégie EMA Momentum + RSI Filter.

Logique d'entrée :
  BUY  : EMA 20 > EMA 50 > EMA 200, RSI entre 45 et 65,
          MACD hist > 0, volume > moyenne 20 bougies
  SELL : EMA 20 < EMA 50 < EMA 200, RSI entre 35 et 55,
          MACD hist < 0, volume > moyenne 20 bougies

Sortie :
  Signal(action, confidence, sl, tp, size)
  SL/TP calculés dynamiquement via ATR.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from analysis.technical.indicators import TechnicalIndicators
from trading.strategies.base import BaseStrategy, Signal

if TYPE_CHECKING:
    from config.settings import Settings


class EmaRsiStrategy(BaseStrategy):
    """Stratégie 1 — EMA Momentum + RSI Filter (signal de bougie en bougie)."""

    # Paramètres EMA
    EMA_FAST   = 20
    EMA_MID    = 50
    EMA_SLOW   = 200

    # Filtre RSI
    RSI_BUY_LOW  = 45
    RSI_BUY_HIGH = 65
    RSI_SELL_LOW  = 35
    RSI_SELL_HIGH = 55

    # Nombre minimal de bougies avant de générer un signal
    WARMUP_BARS = 210

    def __init__(self, settings: "Settings", symbol: str = "BTC/USDT") -> None:
        super().__init__(settings)
        self.symbol = symbol
        self._buffer: deque[dict] = deque(maxlen=self.WARMUP_BARS + 10)

    @property
    def name(self) -> str:
        return "EmaRsiStrategy"

    def on_bar(self, features: dict) -> Signal | None:
        """Traite une nouvelle bougie et retourne un signal si les conditions sont réunies."""
        self._buffer.append(features)
        if len(self._buffer) < self.WARMUP_BARS:
            return None

        df = pd.DataFrame(list(self._buffer))
        df = TechnicalIndicators.add_all(df)

        last = df.iloc[-1]
        if pd.isna(last.get("ema_200") or float("nan")):
            return None

        action = self._evaluate(last)
        if action == "HOLD":
            return None

        atr = last.get("atr_14", 0) or 0
        close = float(last["close"])
        risk_cfg = self.settings.risk
        stops = TechnicalIndicators.atr_stops(
            close, atr,
            sl_mult=risk_cfg.atr_sl_multiplier,
            tp_mult=risk_cfg.atr_tp_multiplier,
        )

        confidence = self._confidence(last)
        sl = stops["stop_loss_long"]   if action == "BUY" else stops["stop_loss_short"]
        tp = stops["take_profit_long"] if action == "BUY" else stops["take_profit_short"]

        logger.debug(f"EmaRsiStrategy: {action} @ {close:.2f}  SL={sl:.2f}  TP={tp:.2f}")

        return Signal(
            action=action,
            symbol=self.symbol,
            confidence=confidence,
            stop_loss=sl,
            take_profit=tp,
            strategy=self.name,
            metadata={
                "rsi":      float(last.get("rsi_14", 0) or 0),
                "macd_hist": float(last.get("macd_hist", 0) or 0),
                "ema_20":   float(last.get("ema_20", 0) or 0),
                "ema_50":   float(last.get("ema_50", 0) or 0),
                "ema_200":  float(last.get("ema_200", 0) or 0),
                "atr":      float(atr),
            },
        )

    def _evaluate(self, row) -> str:
        close   = float(row.get("close", 0))
        ema_20  = float(row.get("ema_20", 0) or 0)
        ema_50  = float(row.get("ema_50", 0) or 0)
        ema_200 = float(row.get("ema_200", 0) or 0)
        rsi     = float(row.get("rsi_14", 50) or 50)
        macd_h  = float(row.get("macd_hist", 0) or 0)
        vol_r   = float(row.get("volume_ratio", 1.0) or 1.0)

        # BUY conditions
        ema_bull = close > ema_20 > ema_50 > ema_200
        rsi_ok_buy = self.RSI_BUY_LOW < rsi < self.RSI_BUY_HIGH
        if ema_bull and rsi_ok_buy and macd_h > 0 and vol_r > 1.0:
            return "BUY"

        # SELL conditions
        ema_bear = close < ema_20 < ema_50 < ema_200
        rsi_ok_sell = self.RSI_SELL_LOW < rsi < self.RSI_SELL_HIGH
        if ema_bear and rsi_ok_sell and macd_h < 0 and vol_r > 1.0:
            return "SELL"

        return "HOLD"

    def _confidence(self, row) -> float:
        """Score de confiance [0, 1] basé sur la force des conditions."""
        score = 0.0
        rsi    = float(row.get("rsi_14", 50) or 50)
        macd_h = float(row.get("macd_hist", 0) or 0)
        vol_r  = float(row.get("volume_ratio", 1.0) or 1.0)

        # RSI dans la zone idéale
        if 50 < rsi < 60 or 40 < rsi < 50:
            score += 0.3

        # MACD fort
        if abs(macd_h) > 0:
            score += 0.3

        # Volume élevé
        if vol_r > 1.5:
            score += 0.2
        elif vol_r > 1.2:
            score += 0.1

        # Bonus EMA spacing
        ema_20  = float(row.get("ema_20", 0) or 0)
        ema_50  = float(row.get("ema_50", 0) or 0)
        if ema_20 != 0 and abs(ema_20 - ema_50) / (ema_50 + 1e-8) > 0.01:
            score += 0.2

        return min(1.0, score)
