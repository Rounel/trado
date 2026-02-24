"""
models/base.py — Interface abstraite commune à tous les modèles TRADO.

Chaque modèle (LSTM, TFT, XGBoost, RL) doit implémenter cette interface
pour être utilisable dans l'ensemble et le backtest.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import pandas as pd


class Action(IntEnum):
    SELL = 0
    HOLD = 1
    BUY  = 2


@dataclass
class Prediction:
    """Résultat d'un modèle de prédiction."""
    action: Action
    confidence: float          # [0, 1]
    probabilities: dict[str, float]  # {"BUY": 0.7, "HOLD": 0.2, "SELL": 0.1}
    model_name: str
    metadata: dict[str, Any] | None = None


class BaseModel(ABC):
    """Interface abstraite pour tous les modèles de prédiction TRADO."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifiant unique du modèle."""
        ...

    @abstractmethod
    def fit(self, X_train: pd.DataFrame | np.ndarray, y_train: np.ndarray) -> None:
        """Entraîne le modèle."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> Prediction:
        """Génère une prédiction pour les features fournies."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Sauvegarde le modèle entraîné sur disque."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Charge un modèle sauvegardé depuis le disque."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
