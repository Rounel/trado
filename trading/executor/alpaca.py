"""
trading/executor/alpaca.py — Exécution d'ordres sur Alpaca (actions US).
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from trading.executor.base import BrokerExecutor
from trading.strategies.base import Signal

if TYPE_CHECKING:
    from config.settings import Settings


class AlpacaExecutor(BrokerExecutor):
    """Exécute les ordres sur Alpaca (paper ou live) via alpaca-trade-api."""

    def __init__(self, settings: "Settings") -> None:
        super().__init__(settings)
        self._api = None

    def _get_api(self):
        if self._api is None:
            import alpaca_trade_api as tradeapi  # type: ignore[import]

            self._api = tradeapi.REST(
                key_id=self.settings.alpaca.api_key,
                secret_key=self.settings.alpaca.secret,
                base_url=self.settings.alpaca.base_url,
            )
        return self._api

    async def submit(self, signal: Signal) -> dict:
        """Soumet un ordre market sur Alpaca."""
        if not signal.is_actionable():
            return {"status": "skipped", "reason": "HOLD"}

        api = self._get_api()
        side = "buy" if signal.action == "BUY" else "sell"
        qty  = float(signal.size or 0)

        if qty <= 0:
            return {"status": "skipped", "reason": "zero_quantity"}

        try:
            order = await asyncio.to_thread(
                api.submit_order,
                symbol=signal.symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",
            )
            logger.info(
                f"AlpacaExecutor: {side} {qty} {signal.symbol} → id={order.id} status={order.status}"
            )

            return {
                "order_id": order.id,
                "status":   order.status,
                "symbol":   signal.symbol,
                "side":     side,
                "qty":      qty,
            }

        except Exception as exc:
            logger.error(f"AlpacaExecutor.submit error: {exc}")
            return {"status": "error", "error": str(exc)}

    async def cancel(self, order_id: str, symbol: str = "") -> bool:
        try:
            api = self._get_api()
            await asyncio.to_thread(api.cancel_order, order_id)
            return True
        except Exception as exc:
            logger.error(f"AlpacaExecutor.cancel error: {exc}")
            return False

    async def get_positions(self) -> list[dict]:
        try:
            api = self._get_api()
            positions = await asyncio.to_thread(api.list_positions)
            return [
                {
                    "symbol":   p.symbol,
                    "qty":      float(p.qty),
                    "avg_cost": float(p.avg_entry_price),
                    "market_value": float(p.market_value),
                    "unrealized_pnl": float(p.unrealized_pl),
                }
                for p in positions
            ]
        except Exception as exc:
            logger.error(f"AlpacaExecutor.get_positions error: {exc}")
            return []

    async def get_balance(self) -> dict:
        try:
            api = self._get_api()
            account = await asyncio.to_thread(api.get_account)
            return {
                "cash":   float(account.cash),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
            }
        except Exception as exc:
            logger.error(f"AlpacaExecutor.get_balance error: {exc}")
            return {}
