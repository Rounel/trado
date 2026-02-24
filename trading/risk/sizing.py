"""
trading/risk/sizing.py — Calcul de la taille de position.

Méthodes disponibles :
  - Kelly Criterion (fractionné : kelly_fraction)
  - ATR-based fixed risk (risque fixe en % du capital par trade)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings


class PositionSizer:
    """Calcule la taille optimale de chaque position."""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings

    def kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital: float,
        price: float,
    ) -> float:
        """
        Taille de position via Kelly Criterion (fractionnaire).

        Kelly = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
        La fraction est limitée par settings.risk.kelly_fraction.

        Retourne la quantité en unités de l'actif.
        """
        if avg_win <= 0 or price <= 0:
            return 0.0

        kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_pct = max(0.0, kelly_pct)

        # Limiter à la fraction configurée (prudence)
        applied_pct = min(kelly_pct, self.settings.risk.kelly_fraction)

        position_value = capital * applied_pct
        return position_value / price

    def atr_size(
        self,
        capital: float,
        price: float,
        atr: float,
        risk_pct: float = 0.01,
    ) -> float:
        """
        Taille de position basée sur le risque ATR.

        Risque par trade = capital * risk_pct
        Taille = risque_par_trade / (ATR * sl_multiplier)

        Retourne la quantité en unités de l'actif.
        """
        if atr <= 0 or price <= 0:
            return 0.0

        risk_amount = capital * risk_pct
        sl_distance = atr * self.settings.risk.atr_sl_multiplier
        quantity = risk_amount / sl_distance
        return quantity

    def fixed_pct_size(self, capital: float, price: float, pct: float = 0.02) -> float:
        """
        Taille fixe : engage `pct` % du capital par trade.
        Méthode la plus simple et la plus robuste pour débuter.
        """
        if price <= 0:
            return 0.0
        return (capital * pct) / price

    def max_quantity(self, capital: float, price: float) -> float:
        """Quantité maximale achetable avec le capital disponible (sans levier)."""
        if price <= 0:
            return 0.0
        return capital / price
