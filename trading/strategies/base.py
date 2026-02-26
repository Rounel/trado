"""
trading/strategies/base.py — Interface abstraite pour les stratégies de trading.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings


@dataclass
class Signal:
    """Signal de trading émis par une stratégie."""
    action:      str            # "BUY" | "SELL" | "HOLD"
    symbol:      str
    confidence:  float          # [0, 1]
    stop_loss:   float | None = None
    take_profit: float | None = None
    size:        float | None = None   # taille de position (en unités de l'actif)
    strategy:    str | None = None
    entry_price: float | None = None   # prix d'entrée réel (renseigné par l'engine)
    metadata:    dict = field(default_factory=dict)

    def is_actionable(self) -> bool:
        return self.action in ("BUY", "SELL")


class BaseStrategy(ABC):
    """Interface commune à toutes les stratégies TRADO."""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def on_bar(self, features: dict) -> Signal | None:
        """
        Appelé à chaque nouvelle bougie fermée.
        Retourne un Signal ou None si pas de signal.
        """
        ...

    @property
    def warmup_bars(self) -> int:
        """Nombre de bougies historiques nécessaires avant le premier signal."""
        return 0

    def on_tick(self, tick: dict) -> Signal | None:
        """Hook optionnel pour les stratégies tick-level."""
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
