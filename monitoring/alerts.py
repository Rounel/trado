"""
monitoring/alerts.py — Alertes Telegram via python-telegram-bot.

Envoie des notifications pour :
  - Signaux de trading exécutés
  - Circuit breaker déclenché
  - Erreurs critiques
  - Rapport journalier de PnL
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings


class TelegramAlerter:
    """Envoi d'alertes Telegram asynchrones."""

    def __init__(self, settings: "Settings") -> None:
        self.token   = settings.telegram.token
        self.chat_id = settings.telegram.chat_id
        self._enabled = bool(self.token and self.chat_id)

    async def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """Envoie un message Telegram."""
        if not self._enabled:
            logger.debug(f"Telegram non configuré — message ignoré: {message[:80]}")
            return False

        try:
            from telegram import Bot  # type: ignore[import]

            bot = Bot(token=self.token)
            async with bot:
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=parse_mode,
                )
            return True
        except Exception as exc:
            logger.warning(f"TelegramAlerter.send error: {exc}")
            return False

    async def alert_trade(self, action: str, symbol: str, price: float, size: float, pnl: float | None = None) -> None:
        """Alerte pour un trade exécuté."""
        emoji = "🟢" if action == "BUY" else "🔴"
        msg = (
            f"{emoji} <b>TRADE — {action} {symbol}</b>\n"
            f"Prix : <code>{price:.4f}</code>\n"
            f"Taille : <code>{size:.4f}</code>"
        )
        if pnl is not None:
            sign = "+" if pnl >= 0 else ""
            msg += f"\nPnL : <code>{sign}{pnl:.2f}$</code>"
        await self.send(msg)

    async def alert_circuit_breaker(self, reason: str) -> None:
        """Alerte critique pour déclenchement du circuit breaker."""
        msg = (
            f"⛔ <b>CIRCUIT BREAKER DÉCLENCHÉ</b>\n"
            f"Raison : <code>{reason}</code>\n"
            f"Trading suspendu automatiquement."
        )
        await self.send(msg)

    async def alert_error(self, error: str, context: str = "") -> None:
        """Alerte pour erreur critique."""
        msg = (
            f"🚨 <b>ERREUR CRITIQUE</b>\n"
            f"<code>{error[:500]}</code>"
        )
        if context:
            msg += f"\nContexte : {context[:200]}"
        await self.send(msg)

    async def daily_report(self, portfolio_value: float, daily_pnl: float, trades_today: int) -> None:
        """Rapport journalier de performance."""
        sign = "+" if daily_pnl >= 0 else ""
        emoji = "📈" if daily_pnl >= 0 else "📉"
        msg = (
            f"{emoji} <b>Rapport Journalier TRADO</b>\n"
            f"Portfolio : <code>{portfolio_value:.2f}$</code>\n"
            f"PnL du jour : <code>{sign}{daily_pnl:.2f}$</code>\n"
            f"Trades : <code>{trades_today}</code>"
        )
        await self.send(msg)

    def send_sync(self, message: str) -> bool:
        """Version synchrone pour usage hors contexte async."""
        try:
            return asyncio.run(self.send(message))
        except RuntimeError:
            # Boucle déjà active
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.send(message))
