"""
models/ensemble.py — Agrégation pondérée des prédictions de tous les modèles.

Stratégie : moyenne pondérée des probabilités, puis argmax.
Les poids peuvent être ajustés selon la performance historique de chaque modèle.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from models.base import BaseModel, Action, Prediction


class EnsembleModel:
    """Agrège les prédictions de plusieurs modèles via average pondérée."""

    DEFAULT_WEIGHTS = {
        "LSTM":   0.25,
        "TFT":    0.35,
        "XGBoost": 0.25,
        "PPO-RL": 0.15,
    }

    def __init__(
        self,
        models: Sequence[BaseModel],
        weights: dict[str, float] | None = None,
    ) -> None:
        self.models = list(models)
        self.weights = weights or self.DEFAULT_WEIGHTS

    def predict(self, X: np.ndarray) -> Prediction:
        """
        Génère une prédiction agrégée.
        X doit être compatible avec chaque modèle (format attendu par chacun).
        """
        aggregated = np.zeros(3)  # [SELL, HOLD, BUY]
        total_weight = 0.0
        individual: list[Prediction] = []

        for model in self.models:
            w = self.weights.get(model.name, 1.0)
            pred = model.predict(X)
            individual.append(pred)
            probs = np.array([
                pred.probabilities.get("SELL", 0.0),
                pred.probabilities.get("HOLD", 0.0),
                pred.probabilities.get("BUY", 0.0),
            ])
            aggregated += w * probs
            total_weight += w

        if total_weight > 0:
            aggregated /= total_weight

        action_idx = int(aggregated.argmax())
        return Prediction(
            action=Action(action_idx),
            confidence=float(aggregated[action_idx]),
            probabilities={
                "SELL": float(aggregated[0]),
                "HOLD": float(aggregated[1]),
                "BUY":  float(aggregated[2]),
            },
            model_name="Ensemble",
            metadata={"individual": [p.model_name for p in individual]},
        )

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Met à jour les poids dynamiquement (ex: après évaluation walk-forward)."""
        self.weights.update(new_weights)

    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """Ajoute un modèle à l'ensemble."""
        self.models.append(model)
        self.weights[model.name] = weight
