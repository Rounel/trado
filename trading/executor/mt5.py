"""
trading/executor/mt5.py — Exécution d'ordres via MetaTrader 5.

L'API MetaTrader5 est synchrone ; tous les appels sont enveloppés
dans asyncio.to_thread() pour ne pas bloquer la boucle événementielle.

Prérequis :
  - Terminal MT5 installé et connecté au compte broker
  - `uv add MetaTrader5`  (Windows uniquement)
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from trading.executor.base import BrokerExecutor
from trading.strategies.base import Signal

if TYPE_CHECKING:
    from config.settings import Settings


def _normalize_symbol(symbol: str) -> str:
    """Convertit 'EUR/USD' → 'EURUSD', 'BTC/USD' → 'BTCUSD', etc."""
    return symbol.replace("/", "")


class MT5Executor(BrokerExecutor):
    """Exécute les ordres sur MetaTrader 5 (forex, CFD, indices, crypto MT5)."""

    def __init__(self, settings: "Settings") -> None:
        super().__init__(settings)
        self._connected = False

    # ------------------------------------------------------------------
    # Connexion
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Initialise la connexion au terminal MT5."""
        return await asyncio.to_thread(self._connect_sync)

    def _connect_sync(self) -> bool:
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
            error = mt5.last_error()
            logger.error(f"MT5Executor: connexion échouée — {error}")
            return False

        info = mt5.account_info()
        logger.info(
            f"MT5Executor: connecté — login={info.login} "
            f"server={info.server} balance={info.balance:.2f} {info.currency}"
        )
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Ferme la connexion MT5."""
        await asyncio.to_thread(self._disconnect_sync)

    def _disconnect_sync(self) -> None:
        import MetaTrader5 as mt5  # type: ignore[import]
        mt5.shutdown()
        self._connected = False
        logger.info("MT5Executor: déconnecté")

    # ------------------------------------------------------------------
    # BrokerExecutor interface
    # ------------------------------------------------------------------

    async def submit(self, signal: Signal) -> dict:
        """Envoie un ordre market au terminal MT5."""
        if not signal.is_actionable():
            return {"status": "skipped", "reason": "HOLD signal"}

        if not self._connected:
            ok = await self.connect()
            if not ok:
                return {"status": "error", "reason": "MT5 non connecté"}

        return await asyncio.to_thread(self._submit_sync, signal)

    def _submit_sync(self, signal: Signal) -> dict:
        import MetaTrader5 as mt5  # type: ignore[import]

        symbol = _normalize_symbol(signal.symbol)
        lots = float(signal.size or 0.01)

        # Vérifie que le symbole est dispo sur le serveur
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"MT5Executor: symbole {symbol} introuvable sur ce serveur")
            return {"status": "error", "reason": f"symbole {symbol} inconnu"}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"status": "error", "reason": "tick introuvable"}

        if signal.action == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": order_type,
            "price": price,
            "deviation": 20,          # slippage max en points
            "magic": 20260222,        # identifiant TRADO
            "comment": f"trado:{signal.strategy}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if signal.stop_loss:
            request["sl"] = float(signal.stop_loss)
        if signal.take_profit:
            request["tp"] = float(signal.take_profit)

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = mt5.last_error()
            retcode = result.retcode if result else "N/A"
            logger.error(f"MT5Executor.submit échec — retcode={retcode} error={error}")
            return {"status": "error", "retcode": retcode, "error": str(error)}

        logger.info(
            f"MT5Executor: ordre {signal.action} {lots} {symbol} "
            f"@ {price:.5f} → ticket={result.order}"
        )
        return {
            "order_id": str(result.order),
            "status": "filled",
            "filled": lots,
            "price": price,
            "symbol": symbol,
            "side": signal.action.lower(),
        }

    async def cancel(self, order_id: str, symbol: str) -> bool:
        """Annule un ordre pending MT5 par son ticket."""
        return await asyncio.to_thread(self._cancel_sync, int(order_id), symbol)

    def _cancel_sync(self, ticket: int, symbol: str) -> bool:
        import MetaTrader5 as mt5  # type: ignore[import]

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"MT5Executor: ordre {ticket} annulé")
            return True
        logger.error(f"MT5Executor.cancel échec — ticket={ticket}")
        return False

    async def get_positions(self) -> list[dict]:
        """Retourne les positions ouvertes sur MT5."""
        return await asyncio.to_thread(self._get_positions_sync)

    def _get_positions_sync(self) -> list[dict]:
        import MetaTrader5 as mt5  # type: ignore[import]

        positions = mt5.positions_get()
        if positions is None:
            return []
        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": p.volume,
                "open_price": p.price_open,
                "current_price": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "comment": p.comment,
            }
            for p in positions
        ]

    async def get_balance(self) -> dict:
        """Retourne le solde du compte MT5."""
        return await asyncio.to_thread(self._get_balance_sync)

    def _get_balance_sync(self) -> dict:
        import MetaTrader5 as mt5  # type: ignore[import]

        info = mt5.account_info()
        if info is None:
            return {}
        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "currency": info.currency,
            "leverage": info.leverage,
        }
