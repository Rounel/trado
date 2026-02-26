"""
trading/executor/binance.py — Exécution d'ordres sur Binance via ccxt async.

Supporte : market, limit, stop-loss orders.

Fragmentation VWAP (v2) :
  Les ordres dont la valeur USD dépasse `settings.binance.vwap_threshold_usd`
  sont automatiquement découpés en `vwap_slices` tranches égales soumises
  avec un délai `vwap_slice_delay_s` entre chaque, afin de minimiser l'impact
  de marché sur le carnet d'ordres.

  Le résultat retourné agrège toutes les tranches :
    - price  = prix moyen pondéré par les quantités remplies
    - filled = quantité totale remplie
    - vwap   = True (marqueur)
"""
from __future__ import annotations

import asyncio
from copy import copy
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
        self._lot_filter_cache: dict[str, dict] = {}

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

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    async def submit(self, signal: Signal) -> dict:
        """
        Soumet un ordre market.
        Si la valeur de l'ordre dépasse le seuil VWAP, le fragmente
        automatiquement en N tranches.
        """
        if not signal.is_actionable():
            return {"status": "skipped", "reason": "HOLD signal"}

        exchange = await self._get_exchange()
        side = "buy" if signal.action == "BUY" else "sell"
        qty  = float(signal.size or 0)

        if qty <= 0:
            logger.warning("BinanceExecutor: quantite nulle — ordre ignore")
            return {"status": "skipped", "reason": "zero_quantity"}

        # Validation et ajustement selon les filtres LOT_SIZE Binance
        qty = await self._apply_lot_filter(signal.symbol, qty)
        if qty is None:
            return {"status": "skipped", "reason": "qty_below_min"}

        # Décision VWAP : compare la valeur estimée de l'ordre au seuil
        entry_price   = float(signal.entry_price or 0)
        order_usd     = qty * entry_price
        vwap_threshold = self.settings.binance.vwap_threshold_usd

        if vwap_threshold > 0 and order_usd >= vwap_threshold:
            return await self._submit_vwap(signal, exchange, side, qty)

        return await self._submit_single(signal, exchange, side, qty)

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

    # ------------------------------------------------------------------
    # Ordre unique (chemin normal)
    # ------------------------------------------------------------------

    async def _submit_single(self, signal: Signal, exchange, side: str, qty: float) -> dict:
        """Soumet un seul ordre market et pose le Stop Loss associé."""
        try:
            order = await exchange.create_market_order(
                symbol=signal.symbol,
                side=side,
                amount=qty,
                params={"newOrderRespType": "FULL"},
            )
            logger.info(
                f"BinanceExecutor: ordre {side} {qty} {signal.symbol} "
                f"id={order.get('id')} status={order.get('status')}"
            )

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

    # ------------------------------------------------------------------
    # Fragmentation VWAP
    # ------------------------------------------------------------------

    async def _submit_vwap(
        self,
        signal:   Signal,
        exchange,
        side:     str,
        qty:      float,
    ) -> dict:
        """
        Fragmente l'ordre en `vwap_slices` tranches égales soumises avec
        un délai `vwap_slice_delay_s` entre chaque.

        Retourne un résultat agrégé avec le prix moyen pondéré et la
        quantité totale remplie.
        """
        cfg           = self.settings.binance
        n_slices      = max(1, cfg.vwap_slices)
        delay_s       = max(0.0, cfg.vwap_slice_delay_s)
        slice_qty     = round(qty / n_slices, 8)

        logger.info(
            f"VWAP [{signal.symbol}] {side}: {n_slices} tranches x "
            f"{slice_qty:.6f} (delai={delay_s}s, total={qty:.6f})"
        )

        total_filled   = 0.0
        weighted_price = 0.0
        fills: list[dict] = []

        for i in range(n_slices):
            # Dernière tranche : prend le reliquat pour éviter les arrondis
            remaining  = round(qty - total_filled, 8)
            if remaining <= 0:
                break
            child_qty = slice_qty if i < n_slices - 1 else remaining

            try:
                order = await exchange.create_market_order(
                    symbol=signal.symbol,
                    side=side,
                    amount=child_qty,
                    params={"newOrderRespType": "FULL"},
                )
                filled_qty  = float(order.get("filled",  child_qty) or child_qty)
                fill_price  = float(order.get("average", order.get("price", 0)) or 0)
                total_filled   += filled_qty
                weighted_price += fill_price * filled_qty
                fills.append(order)
                logger.debug(
                    f"VWAP [{signal.symbol}] tranche {i+1}/{n_slices}: "
                    f"{filled_qty:.6f} @ {fill_price:.4f}"
                )
            except Exception as exc:
                logger.error(
                    f"VWAP [{signal.symbol}] tranche {i+1}/{n_slices} erreur: {exc}"
                )

            # Pause avant la prochaine tranche (sauf après la dernière)
            if i < n_slices - 1:
                await asyncio.sleep(delay_s)

        avg_price = weighted_price / total_filled if total_filled > 0 else 0.0

        # Stop Loss unique sur la quantité totale remplie
        if signal.stop_loss and fills and total_filled > 0:
            sl_signal      = copy(signal)
            sl_signal.size = total_filled
            await self._place_stop_loss(exchange, sl_signal, {"filled": total_filled})

        logger.info(
            f"VWAP [{signal.symbol}] termine: {len(fills)}/{n_slices} tranches, "
            f"rempli={total_filled:.6f} @ prix moyen={avg_price:.4f}"
        )

        return {
            "order_id": f"vwap_{fills[0].get('id') if fills else 'none'}",
            "status":   "closed" if total_filled > 0 else "error",
            "filled":   total_filled,
            "price":    avg_price,
            "symbol":   signal.symbol,
            "side":     side,
            "vwap":     True,
            "slices":   len(fills),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_lot_filter(self, symbol: str) -> dict:
        """Retourne les filtres LOT_SIZE Binance pour un symbole (mis en cache)."""
        if symbol not in self._lot_filter_cache:
            try:
                exchange = await self._get_exchange()
                markets  = await exchange.load_markets()
                mkt      = markets.get(symbol, {})
                limits   = mkt.get("limits", {})
                precision = mkt.get("precision", {})
                self._lot_filter_cache[symbol] = {
                    "min_qty":   float((limits.get("amount") or {}).get("min") or 0),
                    "step_size": float(precision.get("amount") or 0),
                }
                logger.debug(
                    f"LOT_SIZE [{symbol}]: min_qty={self._lot_filter_cache[symbol]['min_qty']}"
                    f"  step_size={self._lot_filter_cache[symbol]['step_size']}"
                )
            except Exception as exc:
                logger.warning(f"Impossible de charger les filtres LOT_SIZE pour {symbol}: {exc}")
                self._lot_filter_cache[symbol] = {"min_qty": 0.0, "step_size": 0.0}
        return self._lot_filter_cache[symbol]

    async def _apply_lot_filter(self, symbol: str, qty: float) -> float | None:
        """
        Ajuste `qty` selon stepSize et vérifie le minQty Binance.
        Retourne la quantité ajustée, ou None si elle est trop petite.
        """
        import math
        lot = await self._get_lot_filter(symbol)
        step = lot["step_size"]
        min_q = lot["min_qty"]

        if step > 0:
            qty = math.floor(qty / step) * step
            qty = round(qty, 8)

        if min_q > 0 and qty < min_q:
            logger.warning(
                f"BinanceExecutor: quantité {qty:.8f} < minQty {min_q:.8f} "
                f"pour {symbol} — ordre ignoré"
            )
            return None
        return qty

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
                    "stopPrice":   signal.stop_loss,
                    "timeInForce": "GTC",
                },
            )
            logger.debug(f"Stop Loss pose @ {signal.stop_loss:.4f}")
        except Exception as exc:
            logger.warning(f"Impossible de poser le Stop Loss : {exc}")
