"""
trading/risk/manager.py — Gestionnaire de risque central.

Valide chaque signal avant exécution :
  1. Circuit breaker (drawdown, perte journalière, pertes consécutives, trailing stop portefeuille)
  2. Nombre max de positions ouvertes
  3. Confiance minimale
  4. SL/TP doivent être définis
  5. Ratio Reward/Risk ≥ min_rr_ratio (défaut 3:1)

Gère en continu :
  - Trailing stops par position (ratchet automatique depuis le pic)
  - Système de Livermore (entrées partielles pyramidales)
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
    """Valide, enrichit et surveille les signaux et positions de trading."""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self._breaker = CircuitBreaker(settings)
        self._sizer   = PositionSizer(settings)

        # Positions ouvertes : symbol → {side, size, entry}
        self._open_positions: dict[str, dict] = {}

        # Trailing stops par position : symbol → stop_price courant
        self._trailing_stops: dict[str, float] = {}
        # Prix à atteindre pour activer le trailing (gain d'activation)
        self._trailing_activation: dict[str, float] = {}

        # Suivi Livermore : symbol → {total_size, remaining_fraction, side,
        #                              next_add_price, stop_loss, take_profit, strategy}
        self._livermore_targets: dict[str, dict] = {}

        self._portfolio_value: float = settings.risk.capital_usd

    # ------------------------------------------------------------------
    # Validation de signal
    # ------------------------------------------------------------------

    def validate_signal(self, signal: Signal, price: float | None = None) -> bool:
        """
        Valide un signal avant exécution.
        `price` = cours actuel de l'actif (utilisé pour le calcul R/R).
        Retourne True si le signal peut être exécuté.
        """
        # 1. Circuit breaker
        if self._breaker.is_paused:
            logger.warning(
                f"RiskManager: signal refusé — circuit breaker actif "
                f"({self._breaker._pause_reason})"
            )
            return False

        if not self._breaker.update(self._portfolio_value):
            return False

        # 2. Nombre max de positions ouvertes
        if signal.action == "BUY":
            open_count = len(self._open_positions)
            if open_count >= self.settings.risk.max_positions:
                logger.warning(f"RiskManager: max positions atteint ({open_count})")
                return False

        # 3. Confiance minimale
        if signal.confidence < 0.3:
            logger.debug(f"RiskManager: confiance trop faible ({signal.confidence:.2f})")
            return False

        # 4. SL/TP doivent être définis pour les signaux actionnables
        if signal.is_actionable() and (signal.stop_loss is None or signal.take_profit is None):
            logger.warning("RiskManager: SL/TP non définis — signal refusé")
            return False

        # 5. Filtre Reward/Risk ≥ min_rr_ratio
        entry = price or signal.entry_price
        if signal.is_actionable() and entry and signal.stop_loss and signal.take_profit:
            rr = self._compute_rr(signal.action, entry, signal.stop_loss, signal.take_profit)
            min_rr = self.settings.risk.min_rr_ratio
            if rr < min_rr:
                logger.debug(
                    f"RiskManager: R/R trop faible ({rr:.2f} < {min_rr:.1f}) "
                    f"[{signal.symbol}] — signal refusé"
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def size_signal(self, signal: Signal, capital: float, price: float, atr: float = 0.0) -> Signal:
        """Calcule et assigne la taille de position si non définie. Retourne le signal enrichi."""
        if signal.size is not None:
            return signal

        if atr > 0:
            size = self._sizer.atr_size(capital, price, atr, risk_pct=0.01)
        else:
            size = self._sizer.fixed_pct_size(capital, price, pct=0.02)

        signal.size = round(size, 6)
        return signal

    # ------------------------------------------------------------------
    # Trailing stop par position
    # ------------------------------------------------------------------

    def update_trailing_stops(self, current_prices: dict[str, float]) -> list[str]:
        """
        Ajuste les trailing stops selon les prix courants.
        Retourne la liste des symboles dont le stop a été touché (→ fermer la position).
        """
        triggered: list[str] = []
        trailing_pct = self.settings.risk.trailing_stop_pct / 100

        for symbol, pos in list(self._open_positions.items()):
            price = current_prices.get(symbol)
            if price is None:
                continue

            side            = pos["side"]
            current_stop    = self._trailing_stops.get(symbol)
            activation      = self._trailing_activation.get(symbol)

            if current_stop is None:
                continue

            if side == "BUY":
                # Active le trailing dès que le prix dépasse le niveau d'activation
                if activation is not None and price >= activation:
                    new_stop = price * (1 - trailing_pct)
                    if new_stop > current_stop:
                        self._trailing_stops[symbol] = new_stop
                        logger.debug(
                            f"Trailing stop BUY [{symbol}]: {current_stop:.4f} → {new_stop:.4f} "
                            f"(prix={price:.4f})"
                        )
                # Stop déclenché si le prix descend sous le niveau courant
                if price <= self._trailing_stops[symbol]:
                    logger.info(
                        f"⚡ Trailing stop déclenché [{symbol}] BUY: "
                        f"prix={price:.4f} ≤ stop={self._trailing_stops[symbol]:.4f}"
                    )
                    triggered.append(symbol)

            else:  # SELL / short
                if activation is not None and price <= activation:
                    new_stop = price * (1 + trailing_pct)
                    if new_stop < current_stop:
                        self._trailing_stops[symbol] = new_stop
                        logger.debug(
                            f"Trailing stop SELL [{symbol}]: {current_stop:.4f} → {new_stop:.4f} "
                            f"(prix={price:.4f})"
                        )
                if price >= self._trailing_stops[symbol]:
                    logger.info(
                        f"⚡ Trailing stop déclenché [{symbol}] SELL: "
                        f"prix={price:.4f} ≥ stop={self._trailing_stops[symbol]:.4f}"
                    )
                    triggered.append(symbol)

        return triggered

    def get_trailing_stop(self, symbol: str) -> float | None:
        """Retourne le niveau de trailing stop courant pour un symbole."""
        return self._trailing_stops.get(symbol)

    # ------------------------------------------------------------------
    # Système de Livermore (entrées partielles pyramidales)
    # ------------------------------------------------------------------

    def open_livermore_signal(self, signal: Signal) -> Signal:
        """
        Si Livermore est activé, transforme le signal en entrée partielle (1/3 de la taille).
        Enregistre les paramètres pour le renforcement futur.
        Retourne le signal modifié (ou inchangé si Livermore désactivé / size=None).
        """
        if not self.settings.risk.livermore_enabled:
            return signal

        if not signal.is_actionable() or signal.size is None:
            return signal

        full_size        = signal.size
        init_fraction    = self.settings.risk.livermore_initial_fraction
        initial_size     = full_size * init_fraction
        signal.size      = round(initial_size, 6)

        entry = signal.entry_price
        if entry:
            trigger_pct = self.settings.risk.livermore_add_trigger_pct / 100
            next_add = (
                entry * (1 + trigger_pct) if signal.action == "BUY"
                else entry * (1 - trigger_pct)
            )
            self._livermore_targets[signal.symbol] = {
                "total_size":         full_size,
                "remaining_fraction": 1.0 - init_fraction,
                "filled_fraction":    init_fraction,
                "side":               signal.action,
                "next_add_price":     next_add,
                "stop_loss":          signal.stop_loss,
                "take_profit":        signal.take_profit,
                "strategy":           signal.strategy,
            }
            logger.info(
                f"Livermore [{signal.symbol}]: entrée initiale {init_fraction:.0%} "
                f"({initial_size:.6f} / {full_size:.6f}), "
                f"prochain renforcement si prix {'≥' if signal.action == 'BUY' else '≤'} {next_add:.4f}"
            )

        signal.metadata["livermore"]         = True
        signal.metadata["initial_fraction"]  = init_fraction
        return signal

    def check_livermore_add(self, symbol: str, current_price: float) -> Signal | None:
        """
        Vérifie si le prix courant déclenche un renforcement Livermore.
        Retourne un Signal de renforcement ou None.
        """
        target = self._livermore_targets.get(symbol)
        if target is None or target["remaining_fraction"] <= 1e-6:
            return None

        side      = target["side"]
        next_add  = target["next_add_price"]
        triggered = (
            (side == "BUY"  and current_price >= next_add) or
            (side == "SELL" and current_price <= next_add)
        )
        if not triggered:
            return None

        # Taille du renforcement = même fraction que l'entrée initiale
        add_fraction = min(target["remaining_fraction"], self.settings.risk.livermore_initial_fraction)
        add_size     = target["total_size"] * add_fraction

        # Mise à jour de la cible Livermore
        target["filled_fraction"]    += add_fraction
        target["remaining_fraction"] -= add_fraction
        trigger_pct = self.settings.risk.livermore_add_trigger_pct / 100
        target["next_add_price"] = (
            current_price * (1 + trigger_pct) if side == "BUY"
            else current_price * (1 - trigger_pct)
        )

        logger.info(
            f"Livermore renforcement [{symbol}]: {add_fraction:.0%} supplémentaire "
            f"à {current_price:.4f} (rempli: {target['filled_fraction']:.0%})"
        )

        return Signal(
            action=side,
            symbol=symbol,
            confidence=0.75,  # confiance réduite : le marché confirme, pas une nouvelle analyse
            stop_loss=target["stop_loss"],
            take_profit=target["take_profit"],
            size=round(add_size, 6),
            strategy=target["strategy"],
            entry_price=current_price,
            metadata={
                "livermore":        True,
                "reinforcement":    True,
                "filled_fraction":  target["filled_fraction"],
            },
        )

    # ------------------------------------------------------------------
    # Gestion des positions
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> dict | None:
        """Retourne une copie des détails de la position ouverte, ou None."""
        pos = self._open_positions.get(symbol)
        return dict(pos) if pos else None

    def record_open(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        stop_loss: float | None = None,
    ) -> None:
        """Enregistre l'ouverture d'une position et initialise son trailing stop."""
        self._open_positions[symbol] = {"side": side, "size": size, "entry": price}

        trailing_pct     = self.settings.risk.trailing_stop_pct / 100
        activation_pct   = self.settings.risk.trailing_stop_activation_pct / 100

        if side == "BUY":
            # Stop initial = SL du signal si fourni, sinon prix - trailing_pct
            self._trailing_stops[symbol]      = stop_loss if stop_loss else price * (1 - trailing_pct)
            self._trailing_activation[symbol] = price * (1 + activation_pct)
        else:  # SELL / short
            self._trailing_stops[symbol]      = stop_loss if stop_loss else price * (1 + trailing_pct)
            self._trailing_activation[symbol] = price * (1 - activation_pct)

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

        # Nettoyage trailing stop et Livermore
        self._trailing_stops.pop(symbol, None)
        self._trailing_activation.pop(symbol, None)
        self._livermore_targets.pop(symbol, None)

        logger.info(
            f"Position fermée — {symbol}  PnL={pnl:+.2f}$  "
            f"Portfolio={self._portfolio_value:.2f}$"
        )
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
            "trailing_stops":    dict(self._trailing_stops),
            "livermore_targets": {s: t["filled_fraction"] for s, t in self._livermore_targets.items()},
        }

    # ------------------------------------------------------------------
    # Méthodes privées
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rr(action: str, entry: float, sl: float, tp: float) -> float:
        """Calcule le ratio Reward/Risk. Retourne 0.0 si le signal est mal formé."""
        if action == "BUY":
            risk   = entry - sl
            reward = tp - entry
        else:  # SELL / short
            risk   = sl - entry
            reward = entry - tp

        if risk <= 0 or reward <= 0:
            return 0.0
        return reward / risk
