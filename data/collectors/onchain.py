"""
data/collectors/onchain.py — Données on-chain et Fear & Greed Index.

Sources :
- Fear & Greed Index : api.alternative.me (gratuit, sans auth)
- Données on-chain : placeholder pour intégration future (Glassnode, Nansen...)
"""
from __future__ import annotations

import aiohttp
from loguru import logger


class OnChainCollector:
    """Collecte le Fear & Greed Index et les données on-chain."""

    FEAR_GREED_URL = "https://api.alternative.me/fng/"

    async def fetch_fear_greed(self) -> dict:
        """
        Retourne le Fear & Greed Index actuel.

        Returns:
            {"value": int (0-100), "label": str, "classification": str}
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.FEAR_GREED_URL,
                    params={"limit": 1},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)

            entry = data["data"][0]
            value = int(entry["value"])
            return {
                "value": value,
                "label": entry.get("value_classification", "Unknown"),
                "normalized": value / 100.0,  # [0, 1]
            }
        except Exception as exc:
            logger.warning(f"OnChainCollector.fetch_fear_greed error: {exc}")
            return {"value": 50, "label": "Neutral", "normalized": 0.5}

    async def fetch_btc_dominance(self) -> float:
        """
        Retourne la dominance BTC (%) depuis CoinGecko (gratuit).
        Utile comme signal macro.
        """
        try:
            url = "https://api.coingecko.com/api/v3/global"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            return float(data["data"]["market_cap_percentage"].get("btc", 50.0))
        except Exception as exc:
            logger.warning(f"OnChainCollector.fetch_btc_dominance error: {exc}")
            return 50.0
