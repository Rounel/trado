"""
trading/executor/binance.py — Exécution d'ordres sur Binance via ccxt async.

Supporte : market, limit, stop-loss orders.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from trading.executor.base import BrokerExecutor
from trading.strategies.base import Signal

if TYPE_CHECKING:
    from config.settings import Settings


class BinanceExecutor(BrokerExecutor):
    """Exécute les ordres sur Binance via ccxt.pro."""

    def __init__(self, settings: "Settings") -> None:
        super().__init__(settings)
        self._exchange = None

    async def _get_exchange(self):
        if self._exchange is None:
            import ccxt.pro as ccxtpro  # type: ignore[import]

            self._exchange = ccxtpro.binance(
                {
                    "apiKey":  self.settings.binance.api_key,
                    "secret":  self.settings.binance.secret,
                    "options": {"defaultType": "spot"},
                }
            )
            if self.settings.binance.testnet:
                self._exchange.set_sandbox_mode(True)
        return self._exchange

    async def submit(self, signal: Signal) -> dict:
        """Soumet un ordre market ou limit selon le signal."""
        if not signal.is_actionable():
            return {"status": "skipped", "reason": "HOLD signal"}

        exchange = await self._get_exchange()
        side = "buy" if signal.action == "BUY" else "sell"
        qty  = float(signal.size or 0)

        if qty <= 0:
            logger.warning("BinanceExecutor: quantité nulle — ordre ignoré")
            return {"status": "skipped", "reason": "zero_quantity"}

        try:
            order = await exchange.create_market_order(
                symbol=signal.symbol,
                side=side,
                amount=qty,
                params={"newOrderRespType": "FULL"},
            )
            logger.info(
                f"BinanceExecutor: ordre {side} {qty} {signal.symbol} "
                f"→ id={order.get('id')} status={order.get('status')}"
            )

            # Pose le Stop Loss si défini
            if signal.stop_loss:
                await self._place_stop_loss(exchange, signal, order)

            return {
                "order_id": order.get("id"),
                "status":   order.get("status"),
                "filled":   order.get("filled", 0),
                "price":    order.get("average", order.get("price")),
                "symbol":   signal.symbol,
                "side":     side,
            }

        except Exception as exc:
            logger.error(f"BinanceExecutor.submit error: {exc}")
            return {"status": "error", "error": str(exc)}

    async def _place_stop_loss(self, exchange, signal: Signal, parent_order: dict) -> None:
        """Place un ordre Stop Loss après l'entrée."""
        try:
            side = "sell" if signal.action == "BUY" else "buy"
            qty = float(parent_order.get("filled", signal.size or 0))
            if qty <= 0:
                return

            await exchange.create_order(
                symbol=signal.symbol,
                type="STOP_LOSS_LIMIT",
                side=side,
                amount=qty,
                price=signal.stop_loss,
                params={
                    "stopPrice": signal.stop_loss,
                    "timeInForce": "GTC",
                },
            )
            logger.debug(f"Stop Loss posé @ {signal.stop_loss:.4f}")
        except Exception as exc:
            logger.warning(f"Impossible de poser le Stop Loss : {exc}")

    async def cancel(self, order_id: str, symbol: str) -> bool:
        try:
            exchange = await self._get_exchange()
            await exchange.cancel_order(order_id, symbol)
            return True
        except Exception as exc:
            logger.error(f"BinanceExecutor.cancel error: {exc}")
            return False

    async def get_positions(self) -> list[dict]:
        try:
            exchange = await self._get_exchange()
            balance = await exchange.fetch_balance()
            return [
                {"asset": k, "free": v["free"], "total": v["total"]}
                for k, v in balance.items()
                if isinstance(v, dict) and float(v.get("total", 0)) > 0
            ]
        except Exception as exc:
            logger.error(f"BinanceExecutor.get_positions error: {exc}")
            return []

    async def get_balance(self) -> dict:
        try:
            exchange = await self._get_exchange()
            balance = await exchange.fetch_balance()
            return {
                "USDT": balance.get("USDT", {}).get("free", 0),
                "total": balance.get("total", {}),
            }
        except Exception as exc:
            logger.error(f"BinanceExecutor.get_balance error: {exc}")
            return {}
