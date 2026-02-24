"""
data/pipeline/normalizer.py — Normalisation Min-Max et Z-score des features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Normalizer:
    """Normalise les features pour l'entraînement et l'inférence des modèles IA."""

    def __init__(self, method: str = "zscore") -> None:
        if method not in ("minmax", "zscore"):
            raise ValueError(f"Méthode inconnue : {method}. Choisir 'minmax' ou 'zscore'.")
        self.method = method
        self._scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
        self._fitted = False

    def fit(self, df: pd.DataFrame, columns: list[str] | None = None) -> "Normalizer":
        """Calcule les paramètres de normalisation sur les données d'entraînement."""
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        self._scaler.fit(df[cols].values)
        self._columns = cols
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique la normalisation (doit être appelé après fit)."""
        if not self._fitted:
            raise RuntimeError("Normalizer non encore fitté — appeler fit() d'abord.")
        df = df.copy()
        df[self._columns] = self._scaler.transform(df[self._columns].values)
        return df

    def fit_transform(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        """Fit + transform en une passe."""
        return self.fit(df, columns).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dénormalise les données (utile pour interpréter les prédictions)."""
        if not self._fitted:
            raise RuntimeError("Normalizer non encore fitté.")
        df = df.copy()
        df[self._columns] = self._scaler.inverse_transform(df[self._columns].values)
        return df

    def transform_array(self, arr: np.ndarray) -> np.ndarray:
        """Transforme un tableau numpy directement."""
        if not self._fitted:
            raise RuntimeError("Normalizer non encore fitté.")
        return self._scaler.transform(arr)
