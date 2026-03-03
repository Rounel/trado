"""
data/collectors/news_feed.py — Flux de news financières temps réel.

Sources (par ordre de priorité) :
  1. NewsAPI  (NEWSAPI_KEY)     — actualité générale + finance + macro
  2. CryptoPanic (CRYPTOPANIC_KEY) — actualité crypto spécifique
  3. RSS fallback (sans clé)   — flux publics Reuters / MarketWatch

Retourne une liste de titres (str) prêts à être injectés dans le contexte Grok.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings


# ---------------------------------------------------------------------------
# Mapping symbole → termes de recherche
# ---------------------------------------------------------------------------

SYMBOL_QUERIES: dict[str, tuple[str, str | None]] = {
    # (newsapi_query, cryptopanic_currency_code)
    "BTC/USDT": ("Bitcoin BTC cryptocurrency", "BTC"),
    "ETH/USDT": ("Ethereum ETH cryptocurrency", "ETH"),
    "BNB/USDT": ("Binance BNB crypto",          "BNB"),
    "SOL/USDT": ("Solana SOL crypto",            "SOL"),
    "XRP/USDT": ("XRP Ripple crypto",            "XRP"),
    "AAPL":     ("Apple AAPL stock earnings",     None),
    "TSLA":     ("Tesla TSLA stock",              None),
    "NVDA":     ("NVIDIA NVDA semiconductor",     None),
    "AMZN":     ("Amazon AMZN earnings",          None),
    "MSFT":     ("Microsoft MSFT stock",          None),
    "EURUSD":   ("EUR USD euro dollar forex ECB Fed", None),
    "GBPUSD":   ("GBP USD pound dollar forex BoE", None),
    "USDJPY":   ("USD JPY yen dollar forex BoJ",  None),
}

# RSS publics (aucune clé requise)
_RSS_FEEDS: dict[str, list[str]] = {
    "default": [
        "https://feeds.content.dowjones.io/public/rss/mw_realestate",
        "https://feeds.reuters.com/reuters/businessNews",
    ],
    "crypto": [
        "https://cointelegraph.com/rss",
        "https://coindesk.com/arc/outboundfeeds/rss/",
    ],
}

_NEWSAPI_URL     = "https://newsapi.org/v2/everything"
_CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NewsItem:
    title: str
    source: str
    published_at: datetime
    url: str = ""

    def __str__(self) -> str:
        return self.title


# ---------------------------------------------------------------------------
# NewsFeedCollector
# ---------------------------------------------------------------------------

class NewsFeedCollector:
    """
    Collecte les titres d'actualité pour un symbole donné.

    Args:
        settings        : configuration TRADO (accès aux clés API).
        max_headlines   : nombre maximum de titres à retourner (défaut 5).
        cache_ttl_s     : durée du cache en secondes (défaut 300 = 5 min).
    """

    def __init__(
        self,
        settings: "Settings",
        max_headlines: int = 5,
        cache_ttl_s: int = 300,
    ) -> None:
        self.settings      = settings
        self.max_headlines = max_headlines
        self.cache_ttl_s   = cache_ttl_s
        self._cache: dict[str, tuple[list[NewsItem], float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_headlines(self, symbol: str) -> list[str]:
        """
        Retourne jusqu'à max_headlines titres pour le symbole donné.

        Essaie NewsAPI → CryptoPanic → RSS en cascade.
        Résultat mis en cache pendant cache_ttl_s secondes.
        """
        import time
        now = time.monotonic()
        cached = self._cache.get(symbol)
        if cached and now - cached[1] < self.cache_ttl_s:
            return [item.title for item in cached[0]]

        items = await self._fetch(symbol)
        items = items[: self.max_headlines]
        self._cache[symbol] = (items, now)
        logger.debug(f"NewsFeedCollector: {len(items)} titres pour {symbol}")
        return [item.title for item in items]

    # ------------------------------------------------------------------
    # Cascade de sources
    # ------------------------------------------------------------------

    async def _fetch(self, symbol: str) -> list[NewsItem]:
        query_info = SYMBOL_QUERIES.get(symbol.upper(), ("financial markets economy", None))
        newsapi_q, cp_currency = query_info

        # 1. NewsAPI
        newsapi_key = getattr(self.settings, "news", None)
        newsapi_key = getattr(newsapi_key, "newsapi_key", "") if newsapi_key else ""
        if newsapi_key:
            items = await self._fetch_newsapi(newsapi_q, newsapi_key)
            if items:
                return items

        # 2. CryptoPanic (crypto uniquement)
        cp_key = getattr(self.settings, "news", None)
        cp_key = getattr(cp_key, "cryptopanic_key", "") if cp_key else ""
        if cp_currency and cp_key:
            items = await self._fetch_cryptopanic(cp_currency, cp_key)
            if items:
                return items

        # 3. RSS fallback
        feed_type = "crypto" if cp_currency else "default"
        items = await self._fetch_rss(feed_type)
        return items

    # ------------------------------------------------------------------
    # NewsAPI
    # ------------------------------------------------------------------

    async def _fetch_newsapi(self, query: str, api_key: str) -> list[NewsItem]:
        try:
            import aiohttp
            params = {
                "q":        query,
                "sortBy":   "publishedAt",
                "pageSize": str(self.max_headlines),
                "language": "en",
                "apiKey":   api_key,
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    _NEWSAPI_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()

            items = []
            for art in data.get("articles", []):
                title = art.get("title") or ""
                if not title or title == "[Removed]":
                    continue
                items.append(NewsItem(
                    title=title,
                    source="newsapi",
                    published_at=datetime.now(timezone.utc),
                    url=art.get("url", ""),
                ))
            return items

        except Exception as exc:
            logger.debug(f"NewsFeed._fetch_newsapi error: {exc}")
            return []

    # ------------------------------------------------------------------
    # CryptoPanic
    # ------------------------------------------------------------------

    async def _fetch_cryptopanic(self, currency: str, auth_token: str) -> list[NewsItem]:
        try:
            import aiohttp
            params = {
                "auth_token": auth_token,
                "currencies": currency,
                "filter":     "hot",
                "public":     "true",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    _CRYPTOPANIC_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()

            items = []
            for post in data.get("results", []):
                title = post.get("title") or ""
                if not title:
                    continue
                items.append(NewsItem(
                    title=title,
                    source="cryptopanic",
                    published_at=datetime.now(timezone.utc),
                    url=post.get("url", ""),
                ))
            return items[: self.max_headlines]

        except Exception as exc:
            logger.debug(f"NewsFeed._fetch_cryptopanic error: {exc}")
            return []

    # ------------------------------------------------------------------
    # RSS fallback (parsing XML minimal, sans feedparser)
    # ------------------------------------------------------------------

    async def _fetch_rss(self, feed_type: str = "default") -> list[NewsItem]:
        urls = _RSS_FEEDS.get(feed_type, _RSS_FEEDS["default"])
        items: list[NewsItem] = []

        try:
            import aiohttp
        except ImportError:
            return []

        async with aiohttp.ClientSession() as session:
            for url in urls:
                try:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=8),
                        headers={"User-Agent": "TradoBot/1.0"},
                    ) as resp:
                        if resp.status != 200:
                            continue
                        text = await resp.text()
                    items.extend(self._parse_rss(text))
                    if len(items) >= self.max_headlines:
                        break
                except Exception as exc:
                    logger.debug(f"NewsFeed._fetch_rss error ({url}): {exc}")

        return items[: self.max_headlines]

    def _parse_rss(self, xml_text: str) -> list[NewsItem]:
        """Parse minimal d'un flux RSS pour extraire les titres."""
        items = []
        try:
            root = ET.fromstring(xml_text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            # RSS 2.0
            for item in root.iter("item"):
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    items.append(NewsItem(
                        title=title_el.text.strip(),
                        source="rss",
                        published_at=datetime.now(timezone.utc),
                    ))

            # Atom
            if not items:
                for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
                    title_el = entry.find("{http://www.w3.org/2005/Atom}title")
                    if title_el is not None and title_el.text:
                        items.append(NewsItem(
                            title=title_el.text.strip(),
                            source="rss_atom",
                            published_at=datetime.now(timezone.utc),
                        ))
        except ET.ParseError:
            pass
        return items
