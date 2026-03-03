"""
analysis/sentiment/context_builder.py — Construction du contexte enrichi pour Grok.

Agrège les données hétérogènes (news, calendrier, structure de marché)
en un texte compact injecté dans le prompt de GrokCollector.

L'objectif est que Grok dispose de faits récents concrets pour calibrer
son score de sentiment, plutôt que de répondre depuis sa connaissance statique.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.collectors.economic_calendar import CalendarEvent


class ContextBuilder:
    """
    Construit le contexte textuel enrichi à injecter dans le prompt Grok.

    Args:
        max_headlines    : nombre max de titres de news inclus.
        max_events       : nombre max d'événements macro inclus.
        include_market   : inclure le résumé de marché (price, trend).
    """

    def __init__(
        self,
        max_headlines: int = 5,
        max_events: int = 4,
        include_market: bool = True,
    ) -> None:
        self.max_headlines   = max_headlines
        self.max_events      = max_events
        self.include_market  = include_market

    def build(
        self,
        symbol: str,
        headlines: list[str] | None = None,
        events: list["CalendarEvent"] | None = None,
        current_price: float | None = None,
        trend: str | None = None,
        now: datetime | None = None,
    ) -> str:
        """
        Construit le contexte complet.

        Args:
            symbol        : ex. "BTC/USDT"
            headlines     : titres de news récents (liste de str).
            events        : événements macro pertinents (CalendarEvent).
            current_price : prix actuel de l'actif.
            trend         : label de tendance ("bullish" | "bearish" | "ranging").
            now           : datetime de référence (défaut : UTC maintenant).

        Returns:
            Chaîne de caractères prête à être insérée dans le prompt Grok.
        """
        now  = now or datetime.now(timezone.utc)
        parts: list[str] = []

        # ── Contexte de marché ─────────────────────────────────────────
        if self.include_market and (current_price or trend):
            mkt_parts = [f"Asset: {symbol}"]
            if current_price:
                mkt_parts.append(f"Current price: {current_price:.6g}")
            if trend:
                mkt_parts.append(f"Technical trend: {trend}")
            mkt_parts.append(f"Analysis time: {now.strftime('%Y-%m-%d %H:%M UTC')}")
            parts.append(" | ".join(mkt_parts))

        # ── Événements macro à venir ───────────────────────────────────
        if events:
            relevant = sorted(events, key=lambda e: e.event_dt)[: self.max_events]
            ev_lines = ["Upcoming economic events:"]
            for ev in relevant:
                mins = ev.minutes_until(now)
                if mins >= 0:
                    timing = f"in {mins:.0f}min"
                else:
                    timing = f"{abs(mins):.0f}min ago"
                forecast = f" (forecast: {ev.forecast})" if ev.forecast else ""
                previous = f" (prev: {ev.previous})" if ev.previous else ""
                ev_lines.append(
                    f"  [{ev.impact}] {ev.title} ({ev.country}) {timing}{forecast}{previous}"
                )
            parts.append("\n".join(ev_lines))

        # ── Headlines de news ──────────────────────────────────────────
        if headlines:
            top = headlines[: self.max_headlines]
            news_lines = [f"Recent news headlines for {symbol}:"]
            for h in top:
                news_lines.append(f"  - {h}")
            parts.append("\n".join(news_lines))

        if not parts:
            return f"No specific context available for {symbol}."

        return "\n\n".join(parts)

    def build_blackout_context(
        self,
        symbol: str,
        reason: str,
        events: list["CalendarEvent"] | None = None,
    ) -> str:
        """
        Contexte spécifique pour informer Grok qu'une annonce majeure est imminente.
        Utilisé même si le trading est bloqué (pour mettre à jour le score de fond).
        """
        lines = [
            f"IMPORTANT: A major economic event is imminent for {symbol}.",
            f"Reason: {reason}",
        ]
        if events:
            for ev in events[:3]:
                lines.append(f"  [{ev.impact}] {ev.title} — forecast: {ev.forecast or 'N/A'}")
        lines.append(
            "High volatility and unpredictable price action expected. "
            "Assess sentiment with extreme caution."
        )
        return "\n".join(lines)
