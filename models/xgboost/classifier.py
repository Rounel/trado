"""
models/xgboost/classifier.py — Classificateur XGBoost BUY/SELL/HOLD.

XGBoost excelle sur les features tabulaires et est très rapide à l'inférence,
ce qui le rend idéal pour l'ensemble avec LSTM/TFT.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from models.base import BaseModel, Action, Prediction


class XGBoostClassifier(BaseModel):
    """Classificateur XGBoost pour BUY/HOLD/SELL."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        use_label_encoder: bool = False,
        random_state: int = 42,
    ) -> None:
        import xgboost as xgb  # type: ignore[import]

        self._clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            use_label_encoder=use_label_encoder,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        return "XGBoost"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        if isinstance(X_train, pd.DataFrame):
            self._feature_names = X_train.columns.tolist()
            X_arr = X_train.values
        else:
            X_arr = X_train

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._clf.fit(
            X_arr,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        logger.info(f"XGBoost entraîné — {len(y_train)} échantillons")

    def predict(self, X: pd.DataFrame | np.ndarray) -> Prediction:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        probs = self._clf.predict_proba(X_arr)[0]
        action_idx = int(probs.argmax())

        return Prediction(
            action=Action(action_idx),
            confidence=float(probs[action_idx]),
            probabilities={"SELL": float(probs[0]), "HOLD": float(probs[1]), "BUY": float(probs[2])},
            model_name=self.name,
        )

    def feature_importance(self) -> dict[str, float]:
        """Retourne l'importance des features (utile pour l'interprétabilité)."""
        scores = self._clf.feature_importances_
        if self._feature_names:
            return dict(zip(self._feature_names, scores))
        return {f"f{i}": float(s) for i, s in enumerate(scores)}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._clf, f)
        logger.info(f"XGBoost sauvegardé → {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._clf = pickle.load(f)
        logger.info(f"XGBoost chargé ← {path}")
