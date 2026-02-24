"""
analysis/aggregator/signal_fusion.py — Fusion pondérée et adaptative des signaux.

Combine :
  - Score technique (RSI, EMA, MACD, Bollinger)
  - Score IA (ensemble de modèles LSTM/TFT/XGBoost/RL)
  - Score sentiment (Grok / FinBERT)
  - Imbalance order book

Les poids s'adaptent dynamiquement au régime de marché détecté :
  - TREND   : IA renforcée (meilleure en tendance)
  - RANGE   : Technique renforcé (RSI/BB plus efficaces en range)
  - VOLATILE: Sentiment renforcé (les news pilotent les extrêmes)

→ Produit un score final [-1, +1] et une décision BUY / SELL / HOLD.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Decision(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class FusedSignal:
    """Résultat de la fusion des signaux."""
    decision:        Decision
    score:           float          # [-1, +1]
    confidence:      float          # [0, 1]
    tech_score:      float
    ai_score:        float
    sentiment_score: float
    ob_imbalance:    float
    regime:          str = "unknown"
    weights_used:    dict | None = None
    metadata:        dict | None = None


# Poids par régime de marché (tech, ai, sentiment, ob)
_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "trend_up":   {"tech": 0.25, "ai": 0.50, "sentiment": 0.15, "ob": 0.10},
    "trend_down": {"tech": 0.25, "ai": 0.50, "sentiment": 0.15, "ob": 0.10},
    "range":      {"tech": 0.50, "ai": 0.25, "sentiment": 0.15, "ob": 0.10},
    "volatile":   {"tech": 0.30, "ai": 0.20, "sentiment": 0.35, "ob": 0.15},
    "unknown":    {"tech": 0.35, "ai": 0.40, "sentiment": 0.15, "ob": 0.10},
}


class SignalFusion:
    """Agrège les signaux pour produire une décision finale adaptée au régime."""

    # Poids par défaut (régime unknown)
    W_TECH       = 0.35
    W_AI         = 0.40
    W_SENTIMENT  = 0.15
    W_ORDERBOOK  = 0.10

    # Seuils de décision
    BUY_THRESHOLD  =  0.20
    SELL_THRESHOLD = -0.20

    def fuse(
        self,
        tech_score:      float,
        ai_score:        float,
        sentiment_score: float,
        ob_imbalance:    float = 0.0,
        regime:          str   = "unknown",
    ) -> FusedSignal:
        """
        Calcule le score global et la décision finale.

        Args:
            tech_score      : [-1, +1] — signal technique
            ai_score        : [-1, +1] — signal IA (converti depuis probs BUY/SELL)
            sentiment_score : [-1, +1] — score Grok ou FinBERT
            ob_imbalance    : [-1, +1] — imbalance order book
            regime          : régime détecté (ajuste les poids dynamiquement)

        Returns:
            FusedSignal
        """
        # Poids adaptatifs selon le régime
        w = _REGIME_WEIGHTS.get(regime, _REGIME_WEIGHTS["unknown"])

        score = (
            w["tech"]      * tech_score
            + w["ai"]      * ai_score
            + w["sentiment"] * sentiment_score
            + w["ob"]      * ob_imbalance
        )
        score = max(-1.0, min(1.0, score))

        confidence = abs(score)

        if score >= self.BUY_THRESHOLD:
            decision = Decision.BUY
        elif score <= self.SELL_THRESHOLD:
            decision = Decision.SELL
        else:
            decision = Decision.HOLD

        return FusedSignal(
            decision=decision,
            score=score,
            confidence=confidence,
            tech_score=tech_score,
            ai_score=ai_score,
            sentiment_score=sentiment_score,
            ob_imbalance=ob_imbalance,
            regime=regime,
            weights_used=dict(w),
        )

    @staticmethod
    def tech_score_from_indicators(indicators: dict) -> float:
        """
        Calcule un score technique [-1, +1] depuis un dict d'indicateurs.

        Logique :
          - RSI < 35 → +0.5 (oversold = bullish), > 65 → -0.5
          - EMA alignées haussières → +0.3, baissières → -0.3
          - MACD hist > 0 → +0.2, < 0 → -0.2
          Normalise dans [-1, +1]
        """
        score = 0.0

        rsi = indicators.get("rsi_14", 50.0)
        if rsi < 35:
            score += 0.5
        elif rsi > 65:
            score -= 0.5
        elif rsi < 50:
            score += 0.1
        else:
            score -= 0.1

        close   = indicators.get("close", 0)
        ema_20  = indicators.get("ema_20", close)
        ema_50  = indicators.get("ema_50", close)
        ema_200 = indicators.get("ema_200", close)
        if close > ema_20 > ema_50 > ema_200:
            score += 0.3
        elif close < ema_20 < ema_50 < ema_200:
            score -= 0.3

        macd_hist = indicators.get("macd_hist", 0)
        score += 0.2 if macd_hist > 0 else (-0.2 if macd_hist < 0 else 0)

        return max(-1.0, min(1.0, score))

    @staticmethod
    def ai_score_from_prediction(prediction) -> float:
        """
        Convertit une Prediction de modèle IA en score [-1, +1].
        score = P(BUY) - P(SELL)
        """
        probs = prediction.probabilities
        return float(probs.get("BUY", 0.0) - probs.get("SELL", 0.0))
