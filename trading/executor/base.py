"""
trading/executor/base.py — Interface abstraite pour les brokers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from trading.strategies.base import Signal

if TYPE_CHECKING:
    from config.settings import Settings


class BrokerExecutor(ABC):
    """Interface commune à tous les exécuteurs de broker."""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings

    @abstractmethod
    async def submit(self, signal: Signal) -> dict:
        """
        Soumet un ordre au broker.
        Retourne un dict avec l'état de l'ordre (order_id, status, filled_qty, price...).
        """
        ...

    @abstractmethod
    async def cancel(self, order_id: str, symbol: str) -> bool:
        """Annule un ordre ouvert."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Retourne les positions ouvertes."""
        ...

    @abstractmethod
    async def get_balance(self) -> dict:
        """Retourne le solde du compte."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
