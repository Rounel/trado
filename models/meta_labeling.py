"""
models/meta_labeling.py — Meta-labeling pour filtrer les signaux via Probabilité de Profit (PoP).

Architecture :
  - Modèle secondaire (GradientBoostingClassifier de scikit-learn).
  - Reçoit en entrée les features des setups OTE/IFC/SMC générés par AdvancedStrategy.
  - Prédit la probabilité que le trade soit gagnant (PoP).
  - Seuil par défaut : PoP ≥ 0.55 → trade validé.

Features d'entrée (MetaFeatures) :
  - ote_strength        : force de la zone OTE [0, 1]
  - ob_strength         : qualité de l'Order Block [0, 1]
  - pattern_strength    : force du pattern IFC [0, 1]
  - volume_ratio        : volume actuel / volume moyen (20 bougies)
  - atr_normalized      : ATR / close
  - trend_aligned       : 1 si le trade est dans le sens de la structure, 0 sinon
  - fvg_present         : 1 si un FVG non comblé est dans la zone
  - liquidity_count     : nombre de touches sur la zone de liquidité la plus proche
  - orderbook_imbalance : (bid_vol - ask_vol) / (bid_vol + ask_vol) [-1, 1]
  - sentiment_score     : score Grok [-1, 1]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from models.base import Action, BaseModel, Prediction


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

@dataclass
class MetaFeatures:
    """Vecteur de features pour le meta-labeling."""
    ote_strength: float        = 0.0
    ob_strength: float         = 0.0
    pattern_strength: float    = 0.0
    volume_ratio: float        = 1.0
    atr_normalized: float      = 0.0
    trend_aligned: float       = 0.0
    fvg_present: float         = 0.0
    liquidity_count: float     = 0.0
    orderbook_imbalance: float = 0.0
    sentiment_score: float     = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.ote_strength,
            self.ob_strength,
            self.pattern_strength,
            self.volume_ratio,
            self.atr_normalized,
            self.trend_aligned,
            self.fvg_present,
            self.liquidity_count,
            self.orderbook_imbalance,
            self.sentiment_score,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "ote_strength", "ob_strength", "pattern_strength",
            "volume_ratio", "atr_normalized", "trend_aligned",
            "fvg_present", "liquidity_count", "orderbook_imbalance",
            "sentiment_score",
        ]


# ---------------------------------------------------------------------------
# MetaLabelingModel
# ---------------------------------------------------------------------------

class MetaLabelingModel(BaseModel):
    """
    Modèle de meta-labeling basé sur GradientBoostingClassifier.

    Rôle : calculer la Probabilité de Profit (PoP) pour chaque setup
    généré par l'AdvancedStrategy et filtrer les trades de faible qualité.

    Usage typique :
        model = MetaLabelingModel(pop_threshold=0.55)
        features = MetaFeatures(ote_strength=0.8, ob_strength=0.7, ...)
        pred = model.predict(features.to_array().reshape(1, -1))
        if pred.confidence >= model.pop_threshold:
            # Valider le signal
    """

    FEATURE_DIM = 10

    def __init__(
        self,
        pop_threshold: float = 0.55,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
    ) -> None:
        self.pop_threshold = pop_threshold
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
        )
        self._scaler = StandardScaler()
        self._fitted = False

    @property
    def name(self) -> str:
        return "MetaLabeling-GB"

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame | np.ndarray, y_train: np.ndarray) -> None:
        """
        Entraîne le meta-modèle.

        Args:
            X_train : matrice (n_samples, 10) de MetaFeatures.
            y_train : labels binaires 1 = trade gagnant, 0 = trade perdant.
        """
        X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y_train)
        self._fitted = True

    def predict(self, X: pd.DataFrame | np.ndarray) -> Prediction:
        """
        Prédit la PoP et retourne une Prediction standard.

        Si non entraîné, retourne une prédiction HOLD neutre (PoP = 0.5).
        """
        if not self._fitted:
            return self._default_prediction()

        arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        X_scaled = self._scaler.transform(arr)
        pop = float(self._model.predict_proba(X_scaled)[0, 1])

        if pop >= self.pop_threshold:
            action = Action.BUY
        elif (1.0 - pop) >= self.pop_threshold:
            action = Action.SELL
        else:
            action = Action.HOLD

        return Prediction(
            action=action,
            confidence=pop,
            probabilities={"BUY": pop, "HOLD": 1 - 2 * abs(pop - 0.5), "SELL": 1.0 - pop},
            model_name=self.name,
            metadata={"pop_threshold": self.pop_threshold, "fitted": self._fitted},
        )

    def is_valid(self, features: MetaFeatures) -> tuple[bool, float]:
        """
        Raccourci : retourne (True, pop) si le setup doit être tradé.

        Returns:
            (valid, pop_score)
        """
        pred = self.predict(features.to_array())
        return pred.confidence >= self.pop_threshold, pred.confidence

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"model": self._model, "scaler": self._scaler, "fitted": self._fitted}, path)

    def load(self, path: str) -> None:
        import joblib
        data = joblib.load(path)
        self._model   = data["model"]
        self._scaler  = data["scaler"]
        self._fitted  = data["fitted"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_prediction(self) -> Prediction:
        return Prediction(
            action=Action.HOLD,
            confidence=0.5,
            probabilities={"BUY": 0.33, "HOLD": 0.34, "SELL": 0.33},
            model_name=self.name,
            metadata={"fitted": False},
        )

    def feature_importances(self) -> dict[str, float] | None:
        """Retourne l'importance des features (après fit)."""
        if not self._fitted:
            return None
        return dict(zip(MetaFeatures.feature_names(), self._model.feature_importances_))
