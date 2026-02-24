"""
analysis/sentiment/finbert_analyzer.py — Sentiment financier via FinBERT.

FinBERT est un BERT pré-entraîné spécifiquement sur des textes financiers
(rapports annuels, actualités Bloomberg, Reuters).
Modèle : ProsusAI/finbert (HuggingFace)

Avantages vs Grok :
  - Spécialisé finance (vs généraliste)
  - 100% local, pas d'API externe, pas de coût par appel
  - Déterministe (même texte → même score)

Inconvénients :
  - ~440 MB à télécharger au premier lancement
  - Nécessite : `uv add transformers`
  - Moins adapté aux news très récentes (knowledge cutoff)

Fallback : retourne 0.0 si transformers non installé.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass


@lru_cache(maxsize=1)
def _load_pipeline():
    """Charge le pipeline FinBERT une seule fois (lazy, thread-safe via lru_cache)."""
    from transformers import pipeline  # type: ignore[import]
    logger.info("FinBERT: chargement du modèle ProsusAI/finbert…")
    pipe = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None,           # retourne toutes les classes avec leurs scores
        device=-1,            # CPU (-1), GPU (0)
    )
    logger.info("FinBERT: modèle chargé ✅")
    return pipe


class FinBERTAnalyzer:
    """
    Analyse le sentiment financier d'un texte via FinBERT.
    Retourne un score normalisé [-1, +1].
    """

    # Mapping label FinBERT → direction
    _LABEL_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

    def __init__(self, use_cache: bool = True) -> None:
        self._use_cache = use_cache
        self._cache: dict[str, float] = {}  # text hash → score
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Vérifie si transformers est installé."""
        if self._available is None:
            try:
                import transformers  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                logger.warning(
                    "FinBERT non disponible — installez avec : uv add transformers\n"
                    "Fallback sur score=0.0"
                )
        return self._available

    def score(self, text: str) -> float:
        """
        Analyse le sentiment d'un texte financier.

        Args:
            text : titre d'une news, résumé d'article, commentaire de marché

        Returns:
            float dans [-1.0, +1.0]
            +1.0 = très bullish, -1.0 = très bearish, 0.0 = neutre
        """
        if not text or not self.is_available():
            return 0.0

        # Cache
        if self._use_cache and text in self._cache:
            return self._cache[text]

        try:
            pipe = _load_pipeline()
            results = pipe(text[:512])  # FinBERT limité à 512 tokens

            # results = [[{"label": "positive", "score": 0.92}, ...]]
            if results and isinstance(results[0], list):
                label_scores = {r["label"]: r["score"] for r in results[0]}
            else:
                label_scores = {r["label"]: r["score"] for r in results}

            # Score pondéré : P(positive) - P(negative)
            pos = label_scores.get("positive", 0.0)
            neg = label_scores.get("negative", 0.0)
            score = float(pos - neg)
            score = max(-1.0, min(1.0, score))

            if self._use_cache:
                self._cache[text] = score
            return score

        except Exception as exc:
            logger.warning(f"FinBERT.score error: {exc}")
            return 0.0

    def score_batch(self, texts: list[str]) -> list[float]:
        """Analyse une liste de textes en batch (plus efficace pour GPU)."""
        if not texts or not self.is_available():
            return [0.0] * len(texts)

        try:
            pipe = _load_pipeline()
            truncated = [t[:512] for t in texts]
            all_results = pipe(truncated)

            scores = []
            for results in all_results:
                if isinstance(results, list):
                    label_scores = {r["label"]: r["score"] for r in results}
                else:
                    label_scores = {results["label"]: results["score"]}
                pos = label_scores.get("positive", 0.0)
                neg = label_scores.get("negative", 0.0)
                scores.append(max(-1.0, min(1.0, float(pos - neg))))
            return scores

        except Exception as exc:
            logger.warning(f"FinBERT.score_batch error: {exc}")
            return [0.0] * len(texts)

    def label(self, score: float) -> str:
        """Convertit un score en label qualitatif."""
        if score >= 0.5:   return "very_bullish"
        if score >= 0.15:  return "bullish"
        if score <= -0.5:  return "very_bearish"
        if score <= -0.15: return "bearish"
        return "neutral"
