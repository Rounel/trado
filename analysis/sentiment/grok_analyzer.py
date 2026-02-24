"""
analysis/sentiment/grok_analyzer.py — Scoring sentiment via xAI Grok.

Produit un score normalisé [-1, +1] pour alimenter le signal_fusion.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings


class GrokAnalyzer:
    """Analyse le sentiment de marché via Grok et retourne un score [-1, +1]."""

    def __init__(self, settings: "Settings") -> None:
        from data.collectors.grok import GrokCollector
        self._collector = GrokCollector(settings)
        self._cache: dict[str, float] = {}

    async def score(self, symbol: str, context: str = "") -> float:
        """
        Retourne le score de sentiment pour le symbole.

        Args:
            symbol  : ex. "BTC/USDT", "AAPL"
            context : actualité récente ou résumé de marché

        Returns:
            float dans [-1.0, +1.0]
        """
        sentiment = await self._collector.fetch_sentiment(symbol=symbol, context=context)
        logger.debug(f"GrokAnalyzer: {symbol} → score={sentiment:.3f}")
        return sentiment

    def score_to_label(self, score: float) -> str:
        """Convertit le score en label qualitatif."""
        if score >= 0.5:
            return "very_bullish"
        elif score >= 0.15:
            return "bullish"
        elif score <= -0.5:
            return "very_bearish"
        elif score <= -0.15:
            return "bearish"
        return "neutral"

    def is_adverse(self, score: float, threshold: float = -0.6) -> bool:
        """True si le sentiment est suffisamment négatif pour bloquer un BUY."""
        return score <= threshold

    def is_favorable(self, score: float, threshold: float = 0.4) -> bool:
        """True si le sentiment est suffisamment positif pour renforcer un BUY."""
        return score >= threshold
