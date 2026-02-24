"""
data/pipeline/features.py — Création des features à partir des bougies OHLCV.

Fenêtres glissantes : returns, rolling stats, indicateurs techniques légers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureBuilder:
    """Construit un vecteur de features à partir d'un DataFrame OHLCV."""

    WINDOWS = [5, 10, 20, 50]

    @classmethod
    def build(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les features au DataFrame OHLCV.

        Colonnes ajoutées : returns, log_returns, volatility_{w},
        price_vs_sma_{w}, volume_ratio_{w}, ...
        """
        df = df.copy()

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        for w in cls.WINDOWS:
            sma = df["close"].rolling(w).mean()
            std = df["close"].rolling(w).std()

            df[f"sma_{w}"] = sma
            df[f"price_vs_sma_{w}"] = (df["close"] - sma) / sma
            df[f"volatility_{w}"] = std / sma
            df[f"volume_ratio_{w}"] = df["volume"] / df["volume"].rolling(w).mean()
            df[f"return_{w}"] = df["close"].pct_change(w)

        # High-Low range normalisé
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]

        # Open-Close body
        df["body"] = (df["close"] - df["open"]) / df["open"]

        df.dropna(inplace=True)
        return df

    @classmethod
    def from_bar(cls, bar: dict) -> dict:
        """
        Crée un dict de features minimal depuis un seul bar.
        Utilisé dans la boucle de trading (streaming).
        """
        return {
            "symbol": bar.get("symbol"),
            "timeframe": bar.get("timeframe"),
            "close": bar["close"],
            "open": bar["open"],
            "high": bar["high"],
            "low": bar["low"],
            "volume": bar["volume"],
            "hl_range": (bar["high"] - bar["low"]) / bar["close"] if bar["close"] else 0.0,
            "body": (bar["close"] - bar["open"]) / bar["open"] if bar["open"] else 0.0,
        }

    @classmethod
    def get_feature_columns(cls) -> list[str]:
        """Retourne la liste des colonnes de features générées."""
        cols = ["returns", "log_returns", "hl_range", "body"]
        for w in cls.WINDOWS:
            cols += [
                f"sma_{w}", f"price_vs_sma_{w}",
                f"volatility_{w}", f"volume_ratio_{w}", f"return_{w}",
            ]
        return cols
