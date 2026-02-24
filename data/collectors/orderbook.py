"""
data/collectors/orderbook.py — Order book imbalance collector (Binance via ccxt.pro).

L'imbalance mesure la pression acheteuse vs vendeuse :
  imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
  > 0 → pression acheteuse, < 0 → pression vendeuse
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings


class OrderBookCollector:
    """Collecte et calcule l'imbalance du carnet d'ordres."""

    def __init__(self, settings: "Settings", depth: int = 20) -> None:
        self.settings = settings
        self.depth = depth

    async def fetch_imbalance(self, symbol: str = "BTC/USDT") -> float:
        """
        Retourne l'imbalance du carnet d'ordres pour le symbole donné.
        Valeur dans [-1, +1].
        """
        try:
            import ccxt.pro as ccxtpro  # type: ignore[import]

            exchange = ccxtpro.binance(
                {
                    "apiKey": self.settings.binance.api_key,
                    "secret": self.settings.binance.secret,
                }
            )
            ob = await exchange.fetch_order_book(symbol, limit=self.depth)
            await exchange.close()

            bid_vol = sum(qty for _, qty in ob["bids"])
            ask_vol = sum(qty for _, qty in ob["asks"])
            total = bid_vol + ask_vol
            return (bid_vol - ask_vol) / total if total > 0 else 0.0

        except Exception as exc:
            logger.warning(f"OrderBookCollector.fetch_imbalance error: {exc}")
            return 0.0

    async def fetch_snapshot(self, symbol: str = "BTC/USDT") -> dict:
        """Retourne un snapshot complet du carnet d'ordres."""
        try:
            import ccxt.pro as ccxtpro  # type: ignore[import]

            exchange = ccxtpro.binance(
                {"apiKey": self.settings.binance.api_key, "secret": self.settings.binance.secret}
            )
            ob = await exchange.fetch_order_book(symbol, limit=self.depth)
            await exchange.close()

            bid_vol = sum(qty for _, qty in ob["bids"])
            ask_vol = sum(qty for _, qty in ob["asks"])
            total = bid_vol + ask_vol

            return {
                "symbol": symbol,
                "bids": ob["bids"][: self.depth],
                "asks": ob["asks"][: self.depth],
                "bid_volume": bid_vol,
                "ask_volume": ask_vol,
                "imbalance": (bid_vol - ask_vol) / total if total > 0 else 0.0,
                "spread": ob["asks"][0][0] - ob["bids"][0][0] if ob["bids"] and ob["asks"] else 0.0,
            }
        except Exception as exc:
            logger.warning(f"OrderBookCollector.fetch_snapshot error: {exc}")
            return {}
