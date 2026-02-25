"""
data/collectors/ohlcv.py — Collecteur OHLCV async pour Binance (ccxt.pro) et Alpaca.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncIterator, TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings


class OHLCVCollector:
    """Collecteur de bougies OHLCV pour Binance et Alpaca."""

    DEFAULT_SYMBOL = "BTC/USDT"
    DEFAULT_TIMEFRAME = "1h"
    DEFAULT_LIMIT = 500

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self._exchange = None  # ccxt.pro exchange (lazy init)

    # ------------------------------------------------------------------
    # Streaming (boucle principale)
    # ------------------------------------------------------------------

    async def stream(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> AsyncIterator[dict]:
        """Génère en continu les dernières bougies fermées."""
        logger.info(f"OHLCVCollector: streaming {symbol} @ {timeframe}")
        while True:
            try:
                bar = await self.fetch_binance_latest(symbol, timeframe)
                if bar is not None:
                    yield bar
                await asyncio.sleep(self._sleep_seconds(timeframe))
            except Exception as exc:
                logger.warning(f"OHLCVCollector stream error: {exc}")
                await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Binance (ccxt.pro)
    # ------------------------------------------------------------------

    async def fetch_binance(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
        limit: int = DEFAULT_LIMIT,
    ) -> pd.DataFrame:
        """Récupère `limit` bougies OHLCV depuis Binance via ccxt.pro."""
        exchange = await self._get_exchange()
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        # Ne pas appeler exchange.close() ici : la connexion est réutilisée
        # par la boucle stream() et la fermer à chaque itération coûte cher.
        return self._to_dataframe(ohlcv)

    async def fetch_binance_latest(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> dict | None:
        """Retourne la dernière bougie fermée sous forme de dict."""
        try:
            df = await self.fetch_binance(symbol, timeframe, limit=2)
            if df.empty:
                return None
            row = df.iloc[-2]  # avant-dernière = dernière bougie fermée
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": row["timestamp"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
        except Exception as exc:
            logger.error(f"fetch_binance_latest error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Alpaca (actions US)
    # ------------------------------------------------------------------

    async def fetch_alpaca(
        self,
        symbol: str,
        timeframe: str = "1Hour",
        start: str = "2024-01-01",
        end: str = "2024-12-31",
    ) -> pd.DataFrame:
        """Récupère des bougies OHLCV depuis Alpaca (actions US)."""
        import alpaca_trade_api as tradeapi

        api = tradeapi.REST(
            key_id=self.settings.alpaca.api_key,
            secret_key=self.settings.alpaca.secret,
            base_url=self.settings.alpaca.base_url,
        )
        # Run dans un thread car alpaca-trade-api est synchrone
        bars = await asyncio.to_thread(
            api.get_bars,
            symbol,
            timeframe,
            start=start,
            end=end,
            adjustment="raw",
        )
        records = [
            {
                "timestamp": bar.t,
                "open": bar.o,
                "high": bar.h,
                "low": bar.l,
                "close": bar.c,
                "volume": bar.v,
            }
            for bar in bars
        ]
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_exchange(self):
        if self._exchange is None:
            import ccxt.pro as ccxtpro  # type: ignore[import]

            # Les données OHLCV sont publiques : on utilise toujours l'API
            # principale de Binance, jamais le testnet.
            # Le testnet (testnet.binance.vision) a des données limitées et
            # est réservé à l'exécution d'ordres (BinanceExecutor).
            self._exchange = ccxtpro.binance(
                {
                    "apiKey": self.settings.binance.api_key,
                    "secret": self.settings.binance.secret,
                    "options": {"defaultType": "spot"},
                }
            )
        return self._exchange

    @staticmethod
    def _to_dataframe(ohlcv: list) -> pd.DataFrame:
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp")

    @staticmethod
    def _sleep_seconds(timeframe: str) -> int:
        mapping = {
            "1m": 60, "5m": 300, "15m": 900,
            "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
        }
        return mapping.get(timeframe, 60)
