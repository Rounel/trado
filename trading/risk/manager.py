"""
trading/risk/manager.py — Gestionnaire de risque central.

Valide chaque signal avant exécution :
  1. Circuit breaker (drawdown, perte journalière, pertes consécutives)
  2. Nombre max de positions ouvertes
  3. Calcule SL/TP et taille de position si non définis dans le signal
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from trading.risk.circuit_breaker import CircuitBreaker
from trading.risk.sizing import PositionSizer
from trading.strategies.base import Signal

if TYPE_CHECKING:
    from config.settings import Settings


class RiskManager:
    """Valide et enrichit les signaux de trading."""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self._breaker = CircuitBreaker(settings)
        self._sizer   = PositionSizer(settings)

        # État des positions ouvertes
        self._open_positions: dict[str, dict] = {}  # symbol → {side, size, entry}
        self._portfolio_value: float = settings.risk.capital_usd

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    def validate_signal(self, signal: Signal) -> bool:
        """
        Valide un signal.
        Retourne True si le signal peut être exécuté, False sinon.
        """
        # 1. Circuit breaker
        if self._breaker.is_paused:
            logger.warning(f"RiskManager: signal refusé — circuit breaker actif ({self._breaker._pause_reason})")
            return False

        if not self._breaker.update(self._portfolio_value):
            return False

        # 2. Nombre max de positions
        if signal.action == "BUY":
            open_count = len(self._open_positions)
            if open_count >= self.settings.risk.max_positions:
                logger.warning(f"RiskManager: max positions atteint ({open_count})")
                return False

        # 3. Confiance minimale
        if signal.confidence < 0.3:
            logger.debug(f"RiskManager: confiance trop faible ({signal.confidence:.2f})")
            return False

        # 4. SL/TP doivent être définis
        if signal.is_actionable() and (signal.stop_loss is None or signal.take_profit is None):
            logger.warning("RiskManager: SL/TP non définis — signal refusé")
            return False

        return True

    def size_signal(self, signal: Signal, capital: float, price: float, atr: float = 0.0) -> Signal:
        """
        Calcule et assigne la taille de position si non définie.
        Retourne le signal enrichi.
        """
        if signal.size is not None:
            return signal

        if atr > 0:
            size = self._sizer.atr_size(capital, price, atr, risk_pct=0.01)
        else:
            size = self._sizer.fixed_pct_size(capital, price, pct=0.02)

        signal.size = round(size, 6)
        return signal

    # ------------------------------------------------------------------
    # Gestion des positions
    # ------------------------------------------------------------------

    def record_open(self, symbol: str, side: str, size: float, price: float) -> None:
        self._open_positions[symbol] = {"side": side, "size": size, "entry": price}

    def record_close(self, symbol: str, exit_price: float) -> float:
        """
        Ferme une position et calcule le PnL.
        Retourne le PnL en USD.
        """
        pos = self._open_positions.pop(symbol, None)
        if pos is None:
            return 0.0

        pnl = (exit_price - pos["entry"]) * pos["size"]
        if pos["side"] == "SELL":
            pnl = -pnl

        if pnl < 0:
            self._breaker.record_loss()
        else:
            self._breaker.record_win()

        self._portfolio_value += pnl
        logger.info(f"Position fermée — {symbol}  PnL={pnl:+.2f}$  Portfolio={self._portfolio_value:.2f}$")
        return pnl

    def update_portfolio(self, value: float) -> None:
        """Met à jour la valeur du portefeuille (ex: depuis un flux de prix)."""
        self._portfolio_value = value
        self._breaker.update(value)

    def status(self) -> dict:
        return {
            "portfolio_value":   self._portfolio_value,
            "open_positions":    len(self._open_positions),
            "circuit_breaker":   self._breaker.status(),
        }
