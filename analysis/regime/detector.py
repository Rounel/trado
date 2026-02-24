"""
analysis/regime/detector.py — Détection du régime de marché.

Classifie le marché en 4 régimes :
  TREND_UP   : tendance haussière confirmée (ADX fort + EMA alignées hausse)
  TREND_DOWN : tendance baissière confirmée (ADX fort + EMA alignées baisse)
  RANGE      : marché sans tendance, oscillant (ADX faible)
  VOLATILE   : volatilité anormalement élevée (ATR > 1.8× sa moyenne)
  UNKNOWN    : pas assez de données

Impact sur la fusion des signaux :
  - TREND   : l'IA (TFT/RL) est favorisée — elle capte mieux les tendances
  - RANGE   : le technique (RSI, BB) est favorisé — il génère des signaux de retour à la moyenne
  - VOLATILE: le sentiment est renforcé — les news pilotent les mouvements extrêmes
"""
from __future__ import annotations

from enum import Enum

import pandas as pd


class MarketRegime(str, Enum):
    TREND_UP   = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE      = "range"
    VOLATILE   = "volatile"
    UNKNOWN    = "unknown"

    def label(self) -> str:
        labels = {
            "trend_up":   "📈 Tendance haussière",
            "trend_down": "📉 Tendance baissière",
            "range":      "↔️  Range / Consolidation",
            "volatile":   "⚡ Volatilité extrême",
            "unknown":    "❓ Régime inconnu",
        }
        return labels.get(self.value, self.value)

    def is_trending(self) -> bool:
        return self in (MarketRegime.TREND_UP, MarketRegime.TREND_DOWN)


class RegimeDetector:
    """
    Détecte le régime de marché à partir d'un DataFrame d'indicateurs.

    Nécessite les colonnes : adx_14, atr_14, close, ema_20, ema_50, ema_200
    (produites par TechnicalIndicators.add_all)
    """

    # Seuils ADX
    ADX_STRONG = 25   # > 25 → tendance significative
    ADX_WEAK   = 20   # < 20 → absence de tendance (range)

    # Seuil volatilité : ATR courant vs moyenne mobile ATR
    ATR_VOL_MULT   = 1.8   # ATR > 1.8× sa moyenne → régime volatil
    ATR_LOOKBACK   = 50    # fenêtre pour la moyenne ATR

    # Nombre minimal de bougies pour une détection fiable
    MIN_BARS = 30

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """
        Classifie le régime sur la dernière bougie du DataFrame.

        Args:
            df : DataFrame avec indicateurs (au moins MIN_BARS lignes)

        Returns:
            MarketRegime
        """
        if len(df) < self.MIN_BARS:
            return MarketRegime.UNKNOWN

        last = df.iloc[-1]

        # ── 1. Volatilité anormale (priorité haute) ─────────────────────
        atr = float(last.get("atr_14") or 0)
        if atr > 0 and len(df) >= self.ATR_LOOKBACK:
            atr_mean = df["atr_14"].rolling(self.ATR_LOOKBACK).mean().iloc[-1]
            if atr_mean and atr > atr_mean * self.ATR_VOL_MULT:
                return MarketRegime.VOLATILE

        # ── 2. Force de la tendance via ADX ──────────────────────────────
        adx = float(last.get("adx_14") or 0)

        if adx >= self.ADX_STRONG:
            # Direction via alignement EMA
            close   = float(last.get("close")   or 0)
            ema_20  = float(last.get("ema_20")  or close)
            ema_50  = float(last.get("ema_50")  or close)

            if close > ema_20 and ema_20 > ema_50:
                return MarketRegime.TREND_UP
            elif close < ema_20 and ema_20 < ema_50:
                return MarketRegime.TREND_DOWN
            else:
                # ADX fort mais EMA mixtes → on regarde DM+ vs DM-
                dmp = float(last.get("dmp_14") or 0)
                dmn = float(last.get("dmn_14") or 0)
                return MarketRegime.TREND_UP if dmp > dmn else MarketRegime.TREND_DOWN

        if adx < self.ADX_WEAK:
            return MarketRegime.RANGE

        # Entre 20 et 25 : tendance faible → UNKNOWN
        return MarketRegime.UNKNOWN

    def detect_series(self, df: pd.DataFrame) -> pd.Series:
        """Détecte le régime sur chaque bougie du DataFrame (pour visualisation)."""
        regimes = []
        for i in range(len(df)):
            window = df.iloc[: i + 1]
            regimes.append(self.detect(window).value)
        return pd.Series(regimes, index=df.index)
