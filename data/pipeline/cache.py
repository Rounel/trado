"""
data/pipeline/cache.py — Persistance des données via Redis (hot cache) et aiosqlite (cold store).

Redis → prix/features des dernières N bougies (accès O(1) pour les stratégies)
aiosqlite → historique complet pour backtests et analyse offline
"""
from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

import aiosqlite
import redis.asyncio as aioredis
from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings

DB_PATH = "data/cache/trado.db"
OHLCV_KEY = "ohlcv:{symbol}:{timeframe}"
CACHE_TTL = 86400  # 24h


class DataCache:
    """Cache deux niveaux : Redis (hot) + SQLite (cold)."""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self._redis: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Redis (hot cache)
    # ------------------------------------------------------------------

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(self.settings.redis.url, decode_responses=True)
        return self._redis

    async def push_bar(self, symbol: str, timeframe: str, bar: dict, max_len: int = 500) -> None:
        """Pousse une bougie dans le cache Redis (liste circulaire)."""
        try:
            r = await self._get_redis()
            key = OHLCV_KEY.format(symbol=symbol, timeframe=timeframe)
            await r.rpush(key, json.dumps(bar, default=str))
            await r.ltrim(key, -max_len, -1)  # garde les max_len dernières
            await r.expire(key, CACHE_TTL)
        except Exception as exc:
            logger.warning(f"DataCache.push_bar redis error: {exc}")

    async def get_bars(self, symbol: str, timeframe: str, n: int = 200) -> list[dict]:
        """Récupère les n dernières bougies depuis Redis."""
        try:
            r = await self._get_redis()
            key = OHLCV_KEY.format(symbol=symbol, timeframe=timeframe)
            raw = await r.lrange(key, -n, -1)
            return [json.loads(x) for x in raw]
        except Exception as exc:
            logger.warning(f"DataCache.get_bars redis error: {exc}")
            return []

    async def set_feature(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Stocke une feature calculée en cache Redis."""
        try:
            r = await self._get_redis()
            await r.set(f"feature:{key}", json.dumps(value, default=str), ex=ttl)
        except Exception as exc:
            logger.warning(f"DataCache.set_feature error: {exc}")

    async def get_feature(self, key: str) -> Any | None:
        """Récupère une feature depuis le cache Redis."""
        try:
            r = await self._get_redis()
            raw = await r.get(f"feature:{key}")
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning(f"DataCache.get_feature error: {exc}")
            return None

    # ------------------------------------------------------------------
    # SQLite (cold store)
    # ------------------------------------------------------------------

    async def init_db(self) -> None:
        """Crée les tables SQLite si elles n'existent pas."""
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol    TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open      REAL,
                    high      REAL,
                    low       REAL,
                    close     REAL,
                    volume    REAL,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol     TEXT,
                    side       TEXT,
                    price      REAL,
                    quantity   REAL,
                    timestamp  TEXT,
                    order_id   TEXT,
                    strategy   TEXT
                )
            """)
            await db.commit()

    async def insert_bar(self, symbol: str, timeframe: str, bar: dict) -> None:
        """Insère une bougie dans SQLite (ignore les doublons)."""
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    """INSERT OR IGNORE INTO ohlcv
                       (symbol, timeframe, timestamp, open, high, low, close, volume)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        symbol, timeframe,
                        str(bar.get("timestamp")),
                        bar.get("open"), bar.get("high"),
                        bar.get("low"), bar.get("close"), bar.get("volume"),
                    ),
                )
                await db.commit()
        except Exception as exc:
            logger.warning(f"DataCache.insert_bar sqlite error: {exc}")

    async def close(self) -> None:
        """Ferme la connexion Redis proprement."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
