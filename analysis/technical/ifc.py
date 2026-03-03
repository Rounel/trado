"""
analysis/technical/ifc.py — Institutional Forex Concepts : détection de patterns de chandeliers.

Patterns détectés :
  - Pin Bar (Hammer / Shooting Star)
  - Engulfing (Bullish / Bearish)
  - Inside Bar
  - Doji
  - Marubozu

Chaque pattern est détecté sans dépendance externe (calcul pur NumPy/Pandas).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pandas as pd


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class PatternType(str, Enum):
    PIN_BAR_BULL    = "pin_bar_bullish"
    PIN_BAR_BEAR    = "pin_bar_bearish"
    ENGULFING_BULL  = "engulfing_bullish"
    ENGULFING_BEAR  = "engulfing_bearish"
    INSIDE_BAR      = "inside_bar"
    DOJI            = "doji"
    MARUBOZU_BULL   = "marubozu_bullish"
    MARUBOZU_BEAR   = "marubozu_bearish"


@dataclass
class CandlePattern:
    """Pattern détecté sur une bougie ou une paire de bougies."""
    pattern: PatternType
    direction: Literal["bullish", "bearish", "neutral"]
    strength: float     # [0, 1] : qualité du pattern
    index: int          # index dans le DataFrame original

    @property
    def is_bullish(self) -> bool:
        return self.direction == "bullish"

    @property
    def is_bearish(self) -> bool:
        return self.direction == "bearish"


# ---------------------------------------------------------------------------
# Helpers de mesure
# ---------------------------------------------------------------------------

def _body(o: float, c: float) -> float:
    return abs(c - o)

def _wick_up(o: float, h: float, c: float) -> float:
    return h - max(o, c)

def _wick_down(o: float, l: float, c: float) -> float:
    return min(o, c) - l

def _range(h: float, l: float) -> float:
    return h - l + 1e-8


# ---------------------------------------------------------------------------
# CandlePatternDetector
# ---------------------------------------------------------------------------

class CandlePatternDetector:
    """
    Détecteur de patterns de chandeliers pour l'analyse IFC.

    Les seuils sont calibrés sur des données crypto (BTC/USDT 1h/4h) et forex.
    Ils peuvent être ajustés via le constructeur.

    Args:
        pin_bar_ratio       : ratio wick/range minimum pour un Pin Bar (défaut 0.6).
        engulfing_margin    : marge tolérée pour l'engulfing (défaut 0.0).
        doji_body_pct       : body max en % de range pour un Doji (défaut 0.1).
        marubozu_wick_pct   : wick max en % de range pour un Marubozu (défaut 0.05).
    """

    def __init__(
        self,
        pin_bar_ratio: float = 0.6,
        engulfing_margin: float = 0.0,
        doji_body_pct: float = 0.1,
        marubozu_wick_pct: float = 0.05,
    ) -> None:
        self.pin_bar_ratio = pin_bar_ratio
        self.engulfing_margin = engulfing_margin
        self.doji_body_pct = doji_body_pct
        self.marubozu_wick_pct = marubozu_wick_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_all(self, df: pd.DataFrame, lookback: int = 10) -> list[CandlePattern]:
        """
        Détecte tous les patterns sur les `lookback` dernières bougies.

        Returns:
            Liste de CandlePattern triée par index.
        """
        patterns: list[CandlePattern] = []
        subset = df.tail(lookback).reset_index(drop=True)
        n = len(subset)

        for i in range(n):
            o = float(subset["open"].iloc[i])
            h = float(subset["high"].iloc[i])
            l = float(subset["low"].iloc[i])
            c = float(subset["close"].iloc[i])

            # Single-candle patterns
            p = self._pin_bar(o, h, l, c, i)
            if p:
                patterns.append(p)

            p = self._doji(o, h, l, c, i)
            if p:
                patterns.append(p)

            p = self._marubozu(o, h, l, c, i)
            if p:
                patterns.append(p)

            # Two-candle patterns (nécessite i > 0)
            if i > 0:
                po = float(subset["open"].iloc[i - 1])
                ph = float(subset["high"].iloc[i - 1])
                pl = float(subset["low"].iloc[i - 1])
                pc = float(subset["close"].iloc[i - 1])

                p = self._engulfing(po, ph, pl, pc, o, h, l, c, i)
                if p:
                    patterns.append(p)

                p = self._inside_bar(ph, pl, h, l, i)
                if p:
                    patterns.append(p)

        return patterns

    def latest_pattern(self, df: pd.DataFrame) -> CandlePattern | None:
        """Retourne le dernier pattern détecté, ou None."""
        patterns = self.detect_all(df, lookback=5)
        if not patterns:
            return None
        return patterns[-1]

    def has_confirmation(
        self,
        df: pd.DataFrame,
        direction: Literal["bullish", "bearish"],
        lookback: int = 3,
    ) -> tuple[bool, CandlePattern | None]:
        """
        Vérifie si un pattern confirme la direction donnée sur les dernières bougies.

        Returns:
            (confirmed, best_pattern)
        """
        patterns = self.detect_all(df, lookback=lookback)
        matching = [p for p in patterns if p.direction == direction]
        if not matching:
            return False, None
        best = max(matching, key=lambda p: p.strength)
        return True, best

    # ------------------------------------------------------------------
    # Single-candle detectors
    # ------------------------------------------------------------------

    def _pin_bar(self, o: float, h: float, l: float, c: float, idx: int) -> CandlePattern | None:
        """
        Pin Bar (Hammer / Shooting Star).

        Bullish Pin Bar  : wick inférieur ≥ pin_bar_ratio × range, corps dans le tiers haut.
        Bearish Pin Bar  : wick supérieur ≥ pin_bar_ratio × range, corps dans le tiers bas.
        """
        rng      = _range(h, l)
        wick_lo  = _wick_down(o, l, c)
        wick_hi  = _wick_up(o, h, c)
        body     = _body(o, c)

        # Bullish Pin Bar (Hammer)
        if wick_lo >= self.pin_bar_ratio * rng and wick_lo >= 2 * body:
            strength = min(1.0, wick_lo / rng)
            return CandlePattern(PatternType.PIN_BAR_BULL, "bullish", strength, idx)

        # Bearish Pin Bar (Shooting Star)
        if wick_hi >= self.pin_bar_ratio * rng and wick_hi >= 2 * body:
            strength = min(1.0, wick_hi / rng)
            return CandlePattern(PatternType.PIN_BAR_BEAR, "bearish", strength, idx)

        return None

    def _doji(self, o: float, h: float, l: float, c: float, idx: int) -> CandlePattern | None:
        """Doji : corps très petit par rapport au range."""
        rng  = _range(h, l)
        body = _body(o, c)
        if body <= self.doji_body_pct * rng:
            strength = 1.0 - (body / (self.doji_body_pct * rng + 1e-8))
            return CandlePattern(PatternType.DOJI, "neutral", min(1.0, strength), idx)
        return None

    def _marubozu(self, o: float, h: float, l: float, c: float, idx: int) -> CandlePattern | None:
        """Marubozu : quasi absence de mèches, bougie de forte conviction."""
        rng      = _range(h, l)
        wick_lo  = _wick_down(o, l, c)
        wick_hi  = _wick_up(o, h, c)
        max_wick = self.marubozu_wick_pct * rng

        if wick_lo <= max_wick and wick_hi <= max_wick:
            body = _body(o, c)
            strength = min(1.0, body / rng)
            direction = "bullish" if c > o else "bearish"
            ptype = PatternType.MARUBOZU_BULL if c > o else PatternType.MARUBOZU_BEAR
            return CandlePattern(ptype, direction, strength, idx)
        return None

    # ------------------------------------------------------------------
    # Two-candle detectors
    # ------------------------------------------------------------------

    def _engulfing(
        self,
        po: float, ph: float, pl: float, pc: float,
        o: float,  h: float,  l: float,  c: float,
        idx: int,
    ) -> CandlePattern | None:
        """
        Engulfing Bullish / Bearish.

        Bullish : bougie précédente baissière, bougie actuelle haussière qui l'englobe.
        Bearish : bougie précédente haussière, bougie actuelle baissière qui l'englobe.
        """
        margin = self.engulfing_margin

        # Bullish Engulfing
        if pc < po and c > o:
            if o <= pl - margin and c >= ph + margin:
                prev_body = _body(po, pc)
                curr_body = _body(o, c)
                strength = min(1.0, curr_body / (prev_body + 1e-8))
                return CandlePattern(PatternType.ENGULFING_BULL, "bullish", strength, idx)

        # Bearish Engulfing
        if pc > po and c < o:
            if o >= ph + margin and c <= pl - margin:
                prev_body = _body(po, pc)
                curr_body = _body(o, c)
                strength = min(1.0, curr_body / (prev_body + 1e-8))
                return CandlePattern(PatternType.ENGULFING_BEAR, "bearish", strength, idx)

        return None

    def _inside_bar(
        self,
        ph: float, pl: float,
        h: float,  l: float,
        idx: int,
    ) -> CandlePattern | None:
        """
        Inside Bar : la bougie actuelle est entièrement dans la range de la précédente.
        Signal de compression → breakout attendu.
        """
        if h <= ph and l >= pl:
            rng_prev = _range(ph, pl)
            rng_curr = _range(h, l)
            strength = 1.0 - (rng_curr / (rng_prev + 1e-8))
            return CandlePattern(PatternType.INSIDE_BAR, "neutral", max(0.0, strength), idx)
        return None
