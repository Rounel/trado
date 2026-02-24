"""
data/collectors/grok.py — Collecteur de sentiment via xAI Grok API.

Interroge Grok pour obtenir un score de sentiment [-1, +1] sur un actif
à partir de l'actualité récente.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings


SENTIMENT_PROMPT = """\
You are a financial sentiment analyst. Given the following context about {symbol},
provide a sentiment score between -1.0 (very bearish) and +1.0 (very bullish).
Respond ONLY with a JSON object: {{"score": <float>, "reasoning": "<one sentence>"}}.

Context: {context}
"""


class GrokCollector:
    """Collecte le sentiment de marché via xAI Grok."""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self._client = None

    async def fetch_sentiment(self, symbol: str, context: str = "") -> float:
        """
        Retourne un score de sentiment pour le symbole donné.
        context : news ou résumé de marché récent (optionnel).
        """
        if not self.settings.grok.api_key:
            logger.debug("GrokCollector: GROK_API_KEY non configuré, retour 0.0")
            return 0.0

        prompt = SENTIMENT_PROMPT.format(symbol=symbol, context=context or "No specific context.")
        try:
            import json
            import aiohttp

            headers = {
                "Authorization": f"Bearer {self.settings.grok.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.settings.grok.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 128,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.settings.grok.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            score = float(result.get("score", 0.0))
            return max(-1.0, min(1.0, score))

        except Exception as exc:
            logger.warning(f"GrokCollector.fetch_sentiment error: {exc}")
            return 0.0
