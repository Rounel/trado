"""
analysis/technical/ote.py — Optimal Trade Entry (OTE) via retracement de Fibonacci.

Logique :
  - Calcule les niveaux de Fibonacci entre un Swing A (origin) et un Swing B (end).
  - La zone OTE correspond au retracement entre 61,8 % et 78,6 %.
  - Pour un trade LONG  : A = Swing Low, B = Swing High (prix retrace vers le bas).
  - Pour un trade SHORT : A = Swing High, B = Swing Low  (prix retrace vers le haut).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Niveaux de Fibonacci standard
# ---------------------------------------------------------------------------

FIB_LEVELS: dict[str, float] = {
    "0.0":   0.000,
    "23.6":  0.236,
    "38.2":  0.382,
    "50.0":  0.500,
    "61.8":  0.618,
    "70.5":  0.705,
    "78.6":  0.786,
    "88.6":  0.886,
    "100.0": 1.000,
}

# Bornes de la zone OTE
OTE_LOWER = 0.618
OTE_UPPER = 0.786


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FibonacciGrid:
    """Grille de niveaux Fibonacci entre le point A et le point B."""
    swing_a: float          # prix origine (Swing Low pour long, Swing High pour short)
    swing_b: float          # prix extrême (Swing High pour long, Swing Low pour short)
    direction: Literal["long", "short"]
    levels: dict[str, float]  # label → prix absolu

    @property
    def ote_low(self) -> float:
        """Borne basse de la zone OTE (61,8 %)."""
        return self.levels["61.8"]

    @property
    def ote_high(self) -> float:
        """Borne haute de la zone OTE (78,6 %)."""
        return self.levels["78.6"]

    @property
    def equilibrium(self) -> float:
        """Niveau 50 % = équilibre de la range."""
        return self.levels["50.0"]


@dataclass
class OTESignal:
    """Résultat de l'évaluation OTE pour une bougie."""
    in_zone: bool
    direction: Literal["long", "short"]
    grid: FibonacciGrid
    current_price: float
    zone_strength: float    # [0, 1] : proximité avec le niveau 70,5 % idéal

    @property
    def zone_label(self) -> str:
        fib = (self.current_price - self.grid.swing_a) / (self.grid.swing_b - self.grid.swing_a + 1e-8)
        for label, level in sorted(FIB_LEVELS.items(), key=lambda x: abs(x[1] - abs(fib))):
            return label
        return "unknown"


# ---------------------------------------------------------------------------
# FibonacciCalculator
# ---------------------------------------------------------------------------

class FibonacciCalculator:
    """
    Calcule la grille de retracement de Fibonacci entre deux swing points.

    Pour un retracement LONG  :
        Niveau 0 % = Swing High (B), Niveau 100 % = Swing Low (A).
        → prix retrace depuis B vers A.

    Pour un retracement SHORT :
        Niveau 0 % = Swing Low (B), Niveau 100 % = Swing High (A).
        → prix retrace depuis B vers A.
    """

    @staticmethod
    def compute(
        swing_a: float,
        swing_b: float,
        direction: Literal["long", "short"],
    ) -> FibonacciGrid:
        """
        Construit la grille Fibonacci.

        Args:
            swing_a   : niveau de départ du mouvement impulsionnel.
            swing_b   : niveau d'arrivée du mouvement impulsionnel.
            direction : "long" → A est un bas, B est un haut.
                        "short" → A est un haut, B est un bas.

        Returns:
            FibonacciGrid avec tous les niveaux de prix.
        """
        rng = swing_b - swing_a  # positif pour long, négatif pour short

        levels: dict[str, float] = {}
        for label, ratio in FIB_LEVELS.items():
            # Le retracement part de B (extrême) vers A (origine)
            levels[label] = swing_b - ratio * rng

        return FibonacciGrid(
            swing_a=swing_a,
            swing_b=swing_b,
            direction=direction,
            levels=levels,
        )


# ---------------------------------------------------------------------------
# OTEEvaluator
# ---------------------------------------------------------------------------

class OTEEvaluator:
    """
    Évalue si le prix courant est dans la zone OTE (61,8 %–78,6 %).

    Utilise FibonacciCalculator en interne.
    """

    def __init__(self, calculator: FibonacciCalculator | None = None) -> None:
        self._calc = calculator or FibonacciCalculator()

    def evaluate(
        self,
        swing_a: float,
        swing_b: float,
        current_price: float,
        direction: Literal["long", "short"],
    ) -> OTESignal:
        """
        Vérifie si current_price se trouve dans la zone OTE.

        Returns:
            OTESignal avec in_zone, zone_strength et la grille Fibonacci.
        """
        grid = FibonacciCalculator.compute(swing_a, swing_b, direction)

        ote_lo = min(grid.ote_low, grid.ote_high)
        ote_hi = max(grid.ote_low, grid.ote_high)
        in_zone = ote_lo <= current_price <= ote_hi

        # Force = proximité du niveau idéal 70,5 %
        ideal = grid.levels["70.5"]
        zone_width = abs(ote_hi - ote_lo) + 1e-8
        dist_to_ideal = abs(current_price - ideal) / zone_width
        zone_strength = float(np.clip(1.0 - dist_to_ideal, 0.0, 1.0))

        return OTESignal(
            in_zone=in_zone,
            direction=direction,
            grid=grid,
            current_price=current_price,
            zone_strength=zone_strength if in_zone else 0.0,
        )

    def find_ote_setups(
        self,
        swing_highs: list,   # list[SwingPoint]
        swing_lows: list,    # list[SwingPoint]
        current_price: float,
        min_strength: float = 0.3,
    ) -> list[OTESignal]:
        """
        Cherche tous les setups OTE valides à partir des swings disponibles.

        Combine les deux derniers swings high/low pour construire les grilles.
        """
        signals: list[OTESignal] = []
        if len(swing_highs) < 1 or len(swing_lows) < 1:
            return signals

        last_high = swing_highs[-1].price
        last_low  = swing_lows[-1].price

        # Setup LONG : mouvement impulsionnel A(bas) → B(haut), retracement en cours
        if last_high > last_low:
            sig = self.evaluate(last_low, last_high, current_price, "long")
            if sig.in_zone and sig.zone_strength >= min_strength:
                signals.append(sig)

        # Setup SHORT : mouvement impulsionnel A(haut) → B(bas), retracement en cours
        if last_low < last_high:
            sig = self.evaluate(last_high, last_low, current_price, "short")
            if sig.in_zone and sig.zone_strength >= min_strength:
                signals.append(sig)

        return signals
