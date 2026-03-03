"""
analysis/technical/price_action.py — Détection de structure de marché Price Action.

Modules :
  - SwingDetector      : repère les Swing Highs / Swing Lows via argrelextrema.
  - MarketStructure    : classe la structure (HH/HL/LH/LL) et détecte BOS / CHOCH.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


# ---------------------------------------------------------------------------
# Enumerations et types
# ---------------------------------------------------------------------------

class Trend(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"


class StructureEvent(str, Enum):
    BOS_UP   = "BOS_UP"    # Break of Structure haussier
    BOS_DOWN = "BOS_DOWN"  # Break of Structure baissier
    CHOCH_UP   = "CHOCH_UP"    # Change of Character (retournement → haussier)
    CHOCH_DOWN = "CHOCH_DOWN"  # Change of Character (retournement → baissier)


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: Literal["high", "low"]


@dataclass
class StructureAnalysis:
    """Résultat complet d'une analyse de structure de marché."""
    swing_highs: list[SwingPoint]
    swing_lows: list[SwingPoint]
    trend: Trend
    last_higher_high: float | None
    last_lower_low: float | None
    last_higher_low: float | None
    last_lower_high: float | None
    recent_events: list[StructureEvent] = field(default_factory=list)

    @property
    def is_bullish(self) -> bool:
        return self.trend == Trend.BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.trend == Trend.BEARISH


# ---------------------------------------------------------------------------
# SwingDetector
# ---------------------------------------------------------------------------

class SwingDetector:
    """
    Détecte les Swing Highs et Swing Lows via scipy.signal.argrelextrema.

    Args:
        order : nombre de bougies de chaque côté pour valider un pivot (défaut 5).
    """

    def __init__(self, order: int = 5) -> None:
        self.order = order

    def detect(self, df: pd.DataFrame) -> tuple[list[SwingPoint], list[SwingPoint]]:
        """
        Analyse un DataFrame OHLCV et retourne (swing_highs, swing_lows).

        Le DataFrame doit avoir au minimum les colonnes 'high' et 'low'.
        """
        highs = df["high"].values
        lows  = df["low"].values

        high_idx = argrelextrema(highs, np.greater_equal, order=self.order)[0]
        low_idx  = argrelextrema(lows,  np.less_equal,    order=self.order)[0]

        swing_highs = [SwingPoint(index=int(i), price=float(highs[i]), kind="high") for i in high_idx]
        swing_lows  = [SwingPoint(index=int(i), price=float(lows[i]),  kind="low")  for i in low_idx]

        return swing_highs, swing_lows


# ---------------------------------------------------------------------------
# MarketStructure
# ---------------------------------------------------------------------------

class MarketStructure:
    """
    Analyse la structure de marché à partir des swing points.

    Logique :
      - Bullish  : HH (Higher High) + HL (Higher Low) confirmés.
      - Bearish  : LH (Lower High)  + LL (Lower Low)  confirmés.
      - Ranging  : alternance sans tendance nette.
      - BOS      : cassure du dernier swing extrême dans la direction de tendance.
      - CHOCH    : cassure dans la direction opposée à la tendance courante.
    """

    def __init__(self, swing_detector: SwingDetector | None = None) -> None:
        self._detector = swing_detector or SwingDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> StructureAnalysis:
        """
        Lance l'analyse complète sur un DataFrame OHLCV.

        Returns:
            StructureAnalysis avec trend, swing points et événements récents.
        """
        swing_highs, swing_lows = self._detector.detect(df)

        trend = self._classify_trend(swing_highs, swing_lows)
        events = self._detect_events(swing_highs, swing_lows, trend, df)

        lhh = self._last_higher_high(swing_highs)
        lll = self._last_lower_low(swing_lows)
        lhl = self._last_higher_low(swing_lows)
        llh = self._last_lower_high(swing_highs)

        return StructureAnalysis(
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            trend=trend,
            last_higher_high=lhh,
            last_lower_low=lll,
            last_higher_low=lhl,
            last_lower_high=llh,
            recent_events=events,
        )

    # ------------------------------------------------------------------
    # Trend classification
    # ------------------------------------------------------------------

    def _classify_trend(
        self,
        swing_highs: list[SwingPoint],
        swing_lows: list[SwingPoint],
        min_pivots: int = 2,
    ) -> Trend:
        if len(swing_highs) < min_pivots or len(swing_lows) < min_pivots:
            return Trend.RANGING

        recent_highs = sorted(swing_highs[-min_pivots:], key=lambda s: s.index)
        recent_lows  = sorted(swing_lows[-min_pivots:],  key=lambda s: s.index)

        hh = recent_highs[-1].price > recent_highs[0].price  # Higher High
        hl = recent_lows[-1].price  > recent_lows[0].price   # Higher Low
        lh = recent_highs[-1].price < recent_highs[0].price  # Lower High
        ll = recent_lows[-1].price  < recent_lows[0].price   # Lower Low

        if hh and hl:
            return Trend.BULLISH
        if lh and ll:
            return Trend.BEARISH
        return Trend.RANGING

    # ------------------------------------------------------------------
    # BOS / CHOCH detection
    # ------------------------------------------------------------------

    def _detect_events(
        self,
        swing_highs: list[SwingPoint],
        swing_lows: list[SwingPoint],
        trend: Trend,
        df: pd.DataFrame,
    ) -> list[StructureEvent]:
        events: list[StructureEvent] = []
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return events

        last_close = float(df["close"].iloc[-1])
        prev_high  = swing_highs[-2].price
        last_high  = swing_highs[-1].price
        prev_low   = swing_lows[-2].price
        last_low   = swing_lows[-1].price

        if trend == Trend.BULLISH:
            # BOS_UP : cassure du dernier swing high → continuation haussière
            if last_close > last_high and last_high > prev_high:
                events.append(StructureEvent.BOS_UP)
            # CHOCH_DOWN : cassure du dernier swing low → retournement
            if last_close < last_low and last_low < prev_low:
                events.append(StructureEvent.CHOCH_DOWN)

        elif trend == Trend.BEARISH:
            # BOS_DOWN : cassure du dernier swing low → continuation baissière
            if last_close < last_low and last_low < prev_low:
                events.append(StructureEvent.BOS_DOWN)
            # CHOCH_UP : cassure du dernier swing high → retournement
            if last_close > last_high and last_high > prev_high:
                events.append(StructureEvent.CHOCH_UP)

        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _last_higher_high(self, highs: list[SwingPoint]) -> float | None:
        for i in range(len(highs) - 1, 0, -1):
            if highs[i].price > highs[i - 1].price:
                return highs[i].price
        return None

    def _last_lower_low(self, lows: list[SwingPoint]) -> float | None:
        for i in range(len(lows) - 1, 0, -1):
            if lows[i].price < lows[i - 1].price:
                return lows[i].price
        return None

    def _last_higher_low(self, lows: list[SwingPoint]) -> float | None:
        for i in range(len(lows) - 1, 0, -1):
            if lows[i].price > lows[i - 1].price:
                return lows[i].price
        return None

    def _last_lower_high(self, highs: list[SwingPoint]) -> float | None:
        for i in range(len(highs) - 1, 0, -1):
            if highs[i].price < highs[i - 1].price:
                return highs[i].price
        return None
