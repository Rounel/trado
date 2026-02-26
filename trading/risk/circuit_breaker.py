"""
trading/risk/circuit_breaker.py — Garde-fous contre les drawdowns et pertes consécutives.

Déclencheurs :
  - Drawdown > max_drawdown_pct                              → pause complète
  - Perte journalière > max_daily_loss_pct                   → pause journée
  - N pertes consécutives > seuil                            → pause temporaire
  - Trailing stop portefeuille (depuis pic intraday)         → pause journée
"""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings


class CircuitBreaker:
    """Coupe-circuit automatique pour protéger le capital."""

    def __init__(self, settings: "Settings") -> None:
        self.max_drawdown_pct   = settings.risk.max_drawdown_pct
        self.max_daily_loss_pct = settings.risk.max_daily_loss_pct
        self.initial_capital    = settings.risk.capital_usd
        self.max_consecutive_losses = 5

        # Trailing stop portefeuille
        self._ptsl_stop_pct       = settings.risk.portfolio_trailing_stop_pct
        self._ptsl_activation_pct = settings.risk.portfolio_trailing_activation_pct
        self._ptsl_active         = False   # True une fois le gain d'activation atteint
        self._intraday_peak       = self.initial_capital

        self._peak_value          = self.initial_capital
        self._current_value       = self.initial_capital
        self._daily_start_value   = self.initial_capital
        self._daily_date          = date.today()
        self._consecutive_losses  = 0
        self._paused              = False
        self._pause_reason        = ""

    @property
    def is_paused(self) -> bool:
        return self._paused

    def update(self, portfolio_value: float) -> bool:
        """
        Met à jour la valeur du portefeuille et vérifie les seuils.
        Retourne True si le trading peut continuer, False si le circuit est ouvert.
        """
        today = date.today()
        if today != self._daily_date:
            # Nouveau jour → reset tous les compteurs journaliers
            self._daily_date        = today
            self._daily_start_value = portfolio_value
            self._intraday_peak     = portfolio_value
            self._ptsl_active       = False
            self._consecutive_losses = 0
            self._paused            = False

        self._current_value  = portfolio_value
        self._peak_value     = max(self._peak_value, portfolio_value)
        self._intraday_peak  = max(self._intraday_peak, portfolio_value)

        # 1. Drawdown global (depuis le pic absolu)
        drawdown_pct = (self._peak_value - portfolio_value) / self._peak_value * 100
        if drawdown_pct >= self.max_drawdown_pct:
            self._trip(f"Drawdown {drawdown_pct:.1f}% ≥ {self.max_drawdown_pct}%")
            return False

        # 2. Perte journalière
        daily_loss_pct = (self._daily_start_value - portfolio_value) / self._daily_start_value * 100
        if daily_loss_pct >= self.max_daily_loss_pct:
            self._trip(f"Perte journalière {daily_loss_pct:.1f}% ≥ {self.max_daily_loss_pct}%")
            return False

        # 3. Pertes consécutives
        if self._consecutive_losses >= self.max_consecutive_losses:
            self._trip(f"{self._consecutive_losses} pertes consécutives")
            return False

        # 4. Trailing stop portefeuille (verrouille les gains intraday)
        intraday_gain_pct = (self._intraday_peak - self._daily_start_value) / self._daily_start_value * 100
        if intraday_gain_pct >= self._ptsl_activation_pct:
            self._ptsl_active = True

        if self._ptsl_active:
            stop_level = self._intraday_peak * (1 - self._ptsl_stop_pct / 100)
            if portfolio_value <= stop_level:
                self._trip(
                    f"Trailing stop portefeuille: {portfolio_value:.2f}$ <= {stop_level:.2f}$ "
                    f"(pic={self._intraday_peak:.2f}$, gain_max={intraday_gain_pct:.1f}%)"
                )
                return False

        return True

    def record_loss(self) -> None:
        """Enregistre une perte (trade fermé en négatif)."""
        self._consecutive_losses += 1

    def record_win(self) -> None:
        """Réinitialise le compteur de pertes consécutives."""
        self._consecutive_losses = 0

    def sync_capital(self, real_value: float) -> None:
        """Synchronise les valeurs de référence avec le solde réel du compte.

        Appelé au démarrage et périodiquement pour corriger la dérive
        entre le capital interne et le solde Binance réel.
        """
        if real_value <= 0:
            return
        self._current_value = real_value
        self._peak_value    = max(self._peak_value, real_value)
        # Si la journée vient de commencer ou que le daily_start n'a jamais été
        # initialisé depuis un vrai solde, on le recale aussi.
        if self._daily_start_value == self.initial_capital:
            self._daily_start_value = real_value
            self._intraday_peak     = real_value
        logger.info(f"CircuitBreaker: capital synchronisé — {real_value:.2f} USDT")

    def reset(self) -> None:
        """Réinitialisation manuelle (ex: décision opérateur)."""
        self._paused              = False
        self._pause_reason        = ""
        self._consecutive_losses  = 0
        self._ptsl_active         = False
        logger.warning("CircuitBreaker: réinitialisé manuellement")

    def _trip(self, reason: str) -> None:
        if not self._paused:
            self._paused       = True
            self._pause_reason = reason
            logger.critical(f"⛔ CIRCUIT BREAKER DÉCLENCHÉ : {reason}")

    def status(self) -> dict:
        return {
            "paused":            self._paused,
            "reason":            self._pause_reason,
            "drawdown_pct":      (self._peak_value - self._current_value) / self._peak_value * 100,
            "consecutive_losses": self._consecutive_losses,
            "ptsl_active":       self._ptsl_active,
            "intraday_peak":     self._intraday_peak,
        }
