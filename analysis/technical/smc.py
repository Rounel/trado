"""
analysis/technical/smc.py — Smart Money Concepts (SMC).

Modules :
  - OrderBlock         : blocs d'ordres institutionnels (dernière bougie opposée avant un mouvement fort).
  - FairValueGap       : déséquilibre de prix sur 3 bougies (gap non comblé).
  - LiquidityZones     : hauts/bas égaux qui concentrent les stops.
  - StopHuntDetector   : pic de volume anormal autour des niveaux ronds ou swings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Order Blocks
# ---------------------------------------------------------------------------

@dataclass
class OrderBlock:
    """Zone d'ordre institutionnel (rectangle de prix)."""
    kind: Literal["bullish", "bearish"]
    top: float
    bottom: float
    origin_index: int        # index de la bougie de référence
    strength: float          # [0, 1] basé sur l'amplitude du mouvement suivant


class OrderBlockDetector:
    """
    Détecte les Order Blocks (OB) selon la définition SMC :

    Bullish OB  : dernière bougie baissière avant un mouvement impulsionnel haussier.
    Bearish OB  : dernière bougie haussière avant un mouvement impulsionnel baissier.

    Args:
        impulse_factor : ratio minimum de variation pour qualifier un mouvement d'impulsionnel.
        lookback       : fenêtre de recherche (bougies).
    """

    def __init__(self, impulse_factor: float = 0.002, lookback: int = 50) -> None:
        self.impulse_factor = impulse_factor
        self.lookback = lookback

    def detect(self, df: pd.DataFrame) -> list[OrderBlock]:
        """Retourne la liste des OB actifs (non mitigés) sur la fenêtre lookback."""
        blocks: list[OrderBlock] = []
        subset = df.tail(self.lookback).reset_index(drop=True)
        n = len(subset)

        for i in range(1, n - 1):
            body_prev = subset["close"].iloc[i] - subset["open"].iloc[i]
            move_next = subset["close"].iloc[i + 1] - subset["open"].iloc[i + 1]
            move_pct  = abs(move_next) / (subset["close"].iloc[i] + 1e-8)

            if move_pct < self.impulse_factor:
                continue

            # Bullish OB : bougie baissière suivie d'une impulsion haussière
            if body_prev < 0 and move_next > 0:
                top    = max(subset["open"].iloc[i], subset["close"].iloc[i])
                bottom = min(subset["open"].iloc[i], subset["close"].iloc[i])
                strength = min(1.0, move_pct / (self.impulse_factor * 5))
                if not self._is_mitigated(subset, i + 2, bottom, top, "bullish"):
                    blocks.append(OrderBlock("bullish", top, bottom, i, strength))

            # Bearish OB : bougie haussière suivie d'une impulsion baissière
            elif body_prev > 0 and move_next < 0:
                top    = max(subset["open"].iloc[i], subset["close"].iloc[i])
                bottom = min(subset["open"].iloc[i], subset["close"].iloc[i])
                strength = min(1.0, move_pct / (self.impulse_factor * 5))
                if not self._is_mitigated(subset, i + 2, bottom, top, "bearish"):
                    blocks.append(OrderBlock("bearish", top, bottom, i, strength))

        return blocks

    def _is_mitigated(
        self,
        df: pd.DataFrame,
        from_idx: int,
        bottom: float,
        top: float,
        kind: str,
    ) -> bool:
        """Un OB est mitigé si le prix est revenu au milieu de la zone."""
        mid = (top + bottom) / 2
        for i in range(from_idx, len(df)):
            if kind == "bullish" and df["low"].iloc[i] <= mid:
                return True
            if kind == "bearish" and df["high"].iloc[i] >= mid:
                return True
        return False

    def nearest(
        self,
        blocks: list[OrderBlock],
        price: float,
        kind: Literal["bullish", "bearish"],
        tolerance: float = 0.005,
    ) -> OrderBlock | None:
        """Retourne l'OB le plus proche du prix courant (dans la tolérance %)."""
        candidates = [
            b for b in blocks
            if b.kind == kind
            and abs((b.top + b.bottom) / 2 - price) / price <= tolerance
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda b: abs((b.top + b.bottom) / 2 - price))


# ---------------------------------------------------------------------------
# Fair Value Gap
# ---------------------------------------------------------------------------

@dataclass
class FairValueGap:
    """Imbalance de prix entre 3 bougies consécutives."""
    kind: Literal["bullish", "bearish"]
    top: float      # borne haute du gap
    bottom: float   # borne basse du gap
    origin_index: int
    filled: bool = False


class FVGDetector:
    """
    Détecte les Fair Value Gaps (FVG) / Imbalances.

    FVG Bullish : low[i+2] > high[i]  →  gap entre candle i et candle i+2.
    FVG Bearish : high[i+2] < low[i]  →  gap dans l'autre sens.
    """

    def detect(self, df: pd.DataFrame, lookback: int = 50) -> list[FairValueGap]:
        gaps: list[FairValueGap] = []
        subset = df.tail(lookback).reset_index(drop=True)
        n = len(subset)

        for i in range(n - 2):
            high_i = subset["high"].iloc[i]
            low_i  = subset["low"].iloc[i]
            low_i2  = subset["low"].iloc[i + 2]
            high_i2 = subset["high"].iloc[i + 2]

            # Bullish FVG
            if low_i2 > high_i:
                filled = self._is_filled(subset, i + 3, high_i, low_i2, "bullish")
                gaps.append(FairValueGap("bullish", low_i2, high_i, i + 1, filled))

            # Bearish FVG
            elif high_i2 < low_i:
                filled = self._is_filled(subset, i + 3, high_i2, low_i, "bearish")
                gaps.append(FairValueGap("bearish", low_i, high_i2, i + 1, filled))

        return [g for g in gaps if not g.filled]

    def _is_filled(
        self,
        df: pd.DataFrame,
        from_idx: int,
        bottom: float,
        top: float,
        kind: str,
    ) -> bool:
        for i in range(from_idx, len(df)):
            if kind == "bullish" and df["low"].iloc[i] <= bottom:
                return True
            if kind == "bearish" and df["high"].iloc[i] >= top:
                return True
        return False


# ---------------------------------------------------------------------------
# Liquidity Zones
# ---------------------------------------------------------------------------

@dataclass
class LiquidityZone:
    """Zone de liquidité = niveau où plusieurs swings se sont formés."""
    kind: Literal["high", "low"]
    price: float
    count: int      # nombre de touches


class LiquidityZoneDetector:
    """
    Repère les niveaux de liquidité (equal highs / equal lows).

    Equal High : ≥ 2 swing highs dans un écart de `tolerance` %.
    Equal Low  : idem pour les swing lows.
    """

    def __init__(self, tolerance: float = 0.0015) -> None:
        self.tolerance = tolerance

    def detect(
        self,
        swing_highs: list,  # list[SwingPoint] importé depuis price_action
        swing_lows: list,
    ) -> list[LiquidityZone]:
        zones: list[LiquidityZone] = []
        zones.extend(self._cluster(swing_highs, "high"))
        zones.extend(self._cluster(swing_lows, "low"))
        return sorted(zones, key=lambda z: z.count, reverse=True)

    def _cluster(self, points: list, kind: str) -> list[LiquidityZone]:
        if not points:
            return []
        prices = np.array([p.price for p in points])
        used = np.zeros(len(prices), dtype=bool)
        zones: list[LiquidityZone] = []

        for i in range(len(prices)):
            if used[i]:
                continue
            ref = prices[i]
            mask = np.abs(prices - ref) / ref <= self.tolerance
            count = int(mask.sum())
            if count >= 2:
                avg_price = float(prices[mask].mean())
                zones.append(LiquidityZone(kind=kind, price=avg_price, count=count))
                used |= mask

        return zones


# ---------------------------------------------------------------------------
# Stop Hunt Detector
# ---------------------------------------------------------------------------

@dataclass
class StopHuntEvent:
    index: int
    price: float
    direction: Literal["long_hunt", "short_hunt"]  # long_hunt = stops des longs raflés
    volume_spike_ratio: float   # volume / volume_moyen
    near_round: bool


class StopHuntDetector:
    """
    Détecte les manipulations institutionnelles de type « stop hunt » :
      - Pic de volume anormal (> spike_threshold × volume moyen).
      - Cassure temporaire d'un niveau de liquidité (swing ou chiffre rond).
      - Clôture de retour dans la range précédente.

    Args:
        spike_threshold : multiple du volume moyen pour qualifier un pic.
        vol_window      : fenêtre pour calculer le volume moyen.
        round_pct       : tolérance autour des chiffres ronds (%).
    """

    def __init__(
        self,
        spike_threshold: float = 2.5,
        vol_window: int = 20,
        round_pct: float = 0.001,
    ) -> None:
        self.spike_threshold = spike_threshold
        self.vol_window = vol_window
        self.round_pct = round_pct

    def detect(self, df: pd.DataFrame) -> list[StopHuntEvent]:
        events: list[StopHuntEvent] = []
        vol_mean = df["volume"].rolling(self.vol_window).mean()

        for i in range(self.vol_window, len(df)):
            avg_vol = vol_mean.iloc[i]
            if avg_vol <= 0:
                continue
            spike_ratio = df["volume"].iloc[i] / avg_vol
            if spike_ratio < self.spike_threshold:
                continue

            high = df["high"].iloc[i]
            low  = df["low"].iloc[i]
            close = df["close"].iloc[i]

            # Short hunt : wick bas prononcé puis clôture remonte
            wick_low  = (df["open"].iloc[i] - low) / (df["open"].iloc[i] + 1e-8)
            wick_high = (high - df["open"].iloc[i]) / (df["open"].iloc[i] + 1e-8)

            near_round = self._near_round_number(low) or self._near_round_number(high)

            if wick_low > 0.6 * ((high - low) / (df["open"].iloc[i] + 1e-8)):
                events.append(StopHuntEvent(i, low, "long_hunt", spike_ratio, near_round))
            elif wick_high > 0.6 * ((high - low) / (df["open"].iloc[i] + 1e-8)):
                events.append(StopHuntEvent(i, high, "short_hunt", spike_ratio, near_round))

        return events

    def _near_round_number(self, price: float, divisions: int = 100) -> bool:
        """True si le prix est à moins de round_pct% d'un chiffre rond."""
        rounded = round(price / divisions) * divisions
        return abs(price - rounded) / (rounded + 1e-8) <= self.round_pct
