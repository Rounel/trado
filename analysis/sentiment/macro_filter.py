"""
analysis/sentiment/macro_filter.py — Filtre macro-économique pour le trading.

Rôle :
  - Bloquer le trading pendant les fenêtres de blackout (X min avant/après un événement High).
  - Retourner un multiplicateur de volatilité à appliquer au SL/ATR.
  - Fournir un résumé textuel des événements proches (pour les logs / alertes).

Utilisé par core/engine.py dans _on_bar() avant l'exécution du signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from data.collectors.economic_calendar import CalendarEvent


@dataclass
class MacroContext:
    """État macro actuel pour un symbole donné."""
    is_blackout: bool
    vol_multiplier: float           # à appliquer sur atr_sl_mult
    reason: str                     # explication lisible
    upcoming_events: list           # list[CalendarEvent] dans les 4h
    imminent_events: list           # list[CalendarEvent] dans la fenêtre blackout


class MacroFilter:
    """
    Filtre macro-économique basé sur le calendrier économique.

    Args:
        blackout_before_min : minutes AVANT l'événement pour déclencher le blackout.
        blackout_after_min  : minutes APRÈS l'événement pour maintenir le blackout.
        min_impact          : impact minimum pour déclencher un blackout ("High" ou "Medium").
        vol_mult_medium     : multiplicateur ATR pour événements Medium.
        vol_mult_high       : multiplicateur ATR pour événements High.
    """

    def __init__(
        self,
        blackout_before_min: int   = 30,
        blackout_after_min: int    = 15,
        min_impact: str            = "High",
        vol_mult_medium: float     = 1.3,
        vol_mult_high: float       = 2.0,
    ) -> None:
        self.blackout_before_min = blackout_before_min
        self.blackout_after_min  = blackout_after_min
        self.min_impact          = min_impact
        self.vol_mult_medium     = vol_mult_medium
        self.vol_mult_high       = vol_mult_high

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        events: list["CalendarEvent"],
        now: datetime | None = None,
    ) -> MacroContext:
        """
        Évalue le contexte macro pour un symbole à l'instant `now`.

        Returns:
            MacroContext avec is_blackout, vol_multiplier, raison et listes d'événements.
        """
        now = now or datetime.now(timezone.utc)

        from data.collectors.economic_calendar import SYMBOL_CURRENCIES, IMPACT_RANK
        currencies   = SYMBOL_CURRENCIES.get(symbol.upper(), ["USD"])
        min_rank     = IMPACT_RANK.get(self.min_impact, 3)

        relevant = [
            e for e in events
            if e.country in currencies and e.impact_rank >= min_rank
        ]

        imminent   = []
        upcoming   = []
        max_mult   = 1.0
        blackout   = False
        reason_parts: list[str] = []

        for ev in relevant:
            mins = ev.minutes_until(now)

            # Fenêtre blackout : [−after, +before]
            if -self.blackout_after_min <= mins <= self.blackout_before_min:
                imminent.append(ev)
                blackout = True
                if mins >= 0:
                    reason_parts.append(
                        f"{ev.title} ({ev.country}) dans {mins:.0f}min"
                    )
                else:
                    reason_parts.append(
                        f"{ev.title} ({ev.country}) il y a {abs(mins):.0f}min"
                    )

            # Fenêtre 4h : calcul du multiplicateur de volatilité
            elif 0 <= mins <= 240:
                upcoming.append(ev)

            # Multiplicateur proportionnel à l'impact et à la proximité
            if -self.blackout_after_min <= mins <= 240:
                mult = self._vol_multiplier(ev, mins)
                max_mult = max(max_mult, mult)

        reason = (
            "Blackout : " + " | ".join(reason_parts)
            if blackout
            else (f"{len(upcoming)} événement(s) dans les 4h" if upcoming else "Aucun événement proche")
        )

        if blackout:
            logger.warning(f"[MacroFilter] BLACKOUT {symbol} — {reason}")
        elif upcoming:
            logger.debug(f"[MacroFilter] {symbol} — {reason} (mult={max_mult:.2f})")

        return MacroContext(
            is_blackout=blackout,
            vol_multiplier=max_mult,
            reason=reason,
            upcoming_events=upcoming,
            imminent_events=imminent,
        )

    def is_blackout(
        self,
        symbol: str,
        events: list["CalendarEvent"],
        now: datetime | None = None,
    ) -> bool:
        """Raccourci booléen."""
        return self.evaluate(symbol, events, now).is_blackout

    def get_vol_multiplier(
        self,
        symbol: str,
        events: list["CalendarEvent"],
        now: datetime | None = None,
    ) -> float:
        """Retourne uniquement le multiplicateur de volatilité."""
        return self.evaluate(symbol, events, now).vol_multiplier

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _vol_multiplier(self, ev: "CalendarEvent", minutes_until: float) -> float:
        """
        Calcule le multiplicateur en fonction de l'impact et de la proximité.

        Plus l'événement est proche et impactant, plus le multiplicateur est élevé.
        Décroît linéairement de 0 à 4h.
        """
        if ev.impact == "High":
            base = self.vol_mult_high
        elif ev.impact == "Medium":
            base = self.vol_mult_medium
        else:
            return 1.0

        # Dans la fenêtre blackout : multiplicateur maximal
        if minutes_until <= self.blackout_before_min:
            return base

        # Entre blackout et 4h : décroît vers 1.0
        decay = max(0.0, 1.0 - (minutes_until - self.blackout_before_min) / 210)
        return 1.0 + (base - 1.0) * decay
