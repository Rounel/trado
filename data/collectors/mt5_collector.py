"""
data/collectors/mt5_collector.py — Collecteur OHLCV via MetaTrader 5.

Récupère les bougies directement depuis le terminal MT5 local.
L'API MetaTrader5 est synchrone ; asyncio.to_thread() est utilisé
pour ne pas bloquer la boucle événementielle.

Prérequis : terminal MT5 démarré + connexion broker active.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator, TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings

# Mapping timeframe Trado → constante MT5
_TF_MAP = {
    "1m":  1,    # mt5.TIMEFRAME_M1
    "5m":  5,    # mt5.TIMEFRAME_M5
    "15m": 15,   # mt5.TIMEFRAME_M15
    "30m": 30,   # mt5.TIMEFRAME_M30
    "1h":  16385,  # mt5.TIMEFRAME_H1
    "4h":  16388,  # mt5.TIMEFRAME_H4
    "1d":  16408,  # mt5.TIMEFRAME_D1
}

_SLEEP_MAP = {
    "1m": 60, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}


def _normalize_symbol(symbol: str) -> str:
    """'EUR/USD' → 'EURUSD', 'BTC/USD' → 'BTCUSD'."""
    return symbol.replace("/", "")


class MT5Collector:
    """Collecteur de bougies OHLCV depuis le terminal MetaTrader 5."""

    DEFAULT_SYMBOL = "EURUSD"
    DEFAULT_TIMEFRAME = "1h"
    DEFAULT_LIMIT = 500

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialisation / fermeture
    # ------------------------------------------------------------------

    def _ensure_init(self) -> None:
        """Initialise MT5 si ce n'est pas encore fait (synchrone)."""
        if self._initialized:
            return
        import MetaTrader5 as mt5  # type: ignore[import]

        cfg = self.settings.mt5
        kwargs: dict = {
            "login": cfg.login,
            "password": cfg.password,
            "server": cfg.server,
            "timeout": cfg.timeout,
        }
        if cfg.path:
            kwargs["path"] = cfg.path

        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5Collector: initialisation échouée — {mt5.last_error()}")
        self._initialized = True
        logger.debug("MT5Collector: terminal initialisé")

    def close(self) -> None:
        import MetaTrader5 as mt5  # type: ignore[import]
        mt5.shutdown()
        self._initialized = False

    # ------------------------------------------------------------------
    # Streaming (interface identique à OHLCVCollector)
    # ------------------------------------------------------------------

    async def stream(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> AsyncIterator[dict]:
        """Génère en continu les dernières bougies fermées depuis MT5."""
        logger.info(f"MT5Collector: streaming {symbol} @ {timeframe}")
        while True:
            try:
                bar = await self.fetch_latest(symbol, timeframe)
                if bar is not None:
                    yield bar
                await asyncio.sleep(_SLEEP_MAP.get(timeframe, 60))
            except Exception as exc:
                logger.warning(f"MT5Collector stream error: {exc}")
                await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Fetch synchrone / asynchrone
    # ------------------------------------------------------------------

    async def fetch(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
        limit: int = DEFAULT_LIMIT,
    ) -> pd.DataFrame:
        """Récupère `limit` bougies OHLCV depuis MT5."""
        return await asyncio.to_thread(self._fetch_sync, symbol, timeframe, limit)

    def _fetch_sync(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        import MetaTrader5 as mt5  # type: ignore[import]

        self._ensure_init()
        mt5_symbol = _normalize_symbol(symbol)
        tf = self._resolve_timeframe(timeframe)

        mt5.symbol_select(mt5_symbol, True)
        rates = mt5.copy_rates_from_pos(mt5_symbol, tf, 0, limit)

        if rates is None or len(rates) == 0:
            logger.warning(f"MT5Collector: aucune donnée pour {mt5_symbol} {timeframe}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={
            "time": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "volume",
        })
        return df[["timestamp", "open", "high", "low", "close", "volume"]].set_index("timestamp")

    async def fetch_latest(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> dict | None:
        """Retourne la dernière bougie fermée sous forme de dict."""
        try:
            df = await self.fetch(symbol, timeframe, limit=2)
            if df.empty:
                return None
            row = df.iloc[-2]  # avant-dernière = dernière bougie fermée
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": row.name,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        except Exception as exc:
            logger.error(f"MT5Collector.fetch_latest error: {exc}")
            return None

    async def fetch_range(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
    ) -> pd.DataFrame:
        """Récupère des bougies entre deux dates."""
        return await asyncio.to_thread(self._fetch_range_sync, symbol, timeframe, date_from, date_to)

    def _fetch_range_sync(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
    ) -> pd.DataFrame:
        import MetaTrader5 as mt5  # type: ignore[import]

        self._ensure_init()
        mt5_symbol = _normalize_symbol(symbol)
        tf = self._resolve_timeframe(timeframe)
        mt5.symbol_select(mt5_symbol, True)

        rates = mt5.copy_rates_range(mt5_symbol, tf, date_from, date_to)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"time": "timestamp", "tick_volume": "volume"})
        return df[["timestamp", "open", "high", "low", "close", "volume"]].set_index("timestamp")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_timeframe(timeframe: str) -> int:
        """Retourne la constante MT5 correspondant au timeframe Trado."""
        tf = _TF_MAP.get(timeframe)
        if tf is None:
            raise ValueError(
                f"Timeframe '{timeframe}' non supporté par MT5. "
                f"Valeurs valides : {list(_TF_MAP.keys())}"
            )
        return tf
