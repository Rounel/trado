"""
data/collectors/economic_calendar.py — Calendrier économique temps réel.

Source principale : ForexFactory unofficial JSON API (gratuit, sans clé).
  - Cette semaine : https://nfs.faireconomy.media/ff_calendar_thisweek.json
  - Semaine suivante : https://nfs.faireconomy.media/ff_calendar_nextweek.json

Fallback : événements récurrents calculés algorithmiquement (NFP, FOMC, …).

Mapping symbole → devises concernées :
  BTC/USDT → USD | EURUSD → EUR + USD | AAPL → USD | etc.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings


# ---------------------------------------------------------------------------
# Mapping symbole → pays / devises concernés
# ---------------------------------------------------------------------------

SYMBOL_CURRENCIES: dict[str, list[str]] = {
    # Crypto (toutes exposées au USD)
    "BTC/USDT": ["USD"],
    "ETH/USDT": ["USD"],
    "BNB/USDT": ["USD"],
    "SOL/USDT": ["USD"],
    "XRP/USDT": ["USD"],
    # Actions US
    "AAPL":  ["USD"],
    "TSLA":  ["USD"],
    "NVDA":  ["USD"],
    "AMZN":  ["USD"],
    "MSFT":  ["USD"],
    # Forex
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "AUDUSD": ["AUD", "USD"],
    "USDCHF": ["USD", "CHF"],
    "USDCAD": ["USD", "CAD"],
    "NZDUSD": ["NZD", "USD"],
}

# Impact hierarchy
IMPACT_RANK = {"Holiday": 0, "Low": 1, "Medium": 2, "High": 3}

# URLs du calendrier ForexFactory
_FF_THIS_WEEK = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_FF_NEXT_WEEK = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CalendarEvent:
    title: str
    country: str            # "USD", "EUR", "GBP", …
    event_dt: datetime      # datetime conscient du fuseau (UTC)
    impact: str             # "High" | "Medium" | "Low" | "Holiday"
    forecast: str = ""
    previous: str = ""
    actual: str = ""
    source: str = "forexfactory"

    @property
    def impact_rank(self) -> int:
        return IMPACT_RANK.get(self.impact, 0)

    def minutes_until(self, now: datetime | None = None) -> float:
        """Minutes jusqu'à l'événement (négatif = déjà passé)."""
        ref = now or datetime.now(timezone.utc)
        return (self.event_dt - ref).total_seconds() / 60

    def is_high_impact(self) -> bool:
        return self.impact == "High"

    def __str__(self) -> str:
        return f"[{self.impact}] {self.title} ({self.country}) @ {self.event_dt.strftime('%H:%M UTC')}"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_ff_time(date_str: str, time_str: str) -> datetime | None:
    """
    Combine la date ISO ForexFactory et l'heure textuelle en un datetime UTC.

    date_str : "2025-01-10T00:00:00-05:00"
    time_str : "8:30am" | "Tentative" | "All Day" | ""
    """
    try:
        base = datetime.fromisoformat(date_str)
    except Exception:
        return None

    # Heure non précisée → garder la date à minuit
    if not time_str or time_str.lower() in ("tentative", "all day", ""):
        return base.astimezone(timezone.utc)

    match = re.match(r"(\d{1,2}):(\d{2})(am|pm)", time_str.lower().strip())
    if not match:
        return base.astimezone(timezone.utc)

    hour   = int(match.group(1))
    minute = int(match.group(2))
    period = match.group(3)

    if period == "pm" and hour != 12:
        hour += 12
    elif period == "am" and hour == 12:
        hour = 0

    try:
        local_dt = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return local_dt.astimezone(timezone.utc)
    except Exception:
        return base.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Fallback : événements récurrents
# ---------------------------------------------------------------------------

def _nfp_date(year: int, month: int) -> datetime:
    """
    Retourne le premier vendredi du mois donné (date du NFP américain).
    """
    d = datetime(year, month, 1, 13, 30, tzinfo=timezone.utc)  # 13h30 UTC = 8h30 ET
    # weekday() : 0=lundi … 4=vendredi
    days_to_friday = (4 - d.weekday()) % 7
    return d + timedelta(days=days_to_friday)


def _recurring_events(horizon_days: int = 14) -> list[CalendarEvent]:
    """
    Génère les événements récurrents calculables sur les `horizon_days` prochains jours.

    Couvre :  NFP (1er vendredi du mois), Fed Meeting (~8 fois/an approximé).
    """
    now    = datetime.now(timezone.utc)
    events = []

    for delta_month in range(2):
        year  = (now.replace(day=1) + timedelta(days=32 * delta_month)).year
        month = (now.replace(day=1) + timedelta(days=32 * delta_month)).month
        nfp   = _nfp_date(year, month)
        if now <= nfp <= now + timedelta(days=horizon_days):
            events.append(CalendarEvent(
                title="Non-Farm Payrolls (NFP)",
                country="USD",
                event_dt=nfp,
                impact="High",
                source="recurring",
            ))

    return events


# ---------------------------------------------------------------------------
# EconomicCalendarCollector
# ---------------------------------------------------------------------------

class EconomicCalendarCollector:
    """
    Collecte les événements du calendrier économique et les filtre par symbole.

    Args:
        horizon_hours : fenêtre temporelle de recherche (défaut 48 h).
        min_impact    : impact minimum à retenir ("Low" | "Medium" | "High").
    """

    def __init__(
        self,
        settings: "Settings | None" = None,
        horizon_hours: int = 48,
        min_impact: str = "Medium",
    ) -> None:
        self.settings      = settings
        self.horizon_hours = horizon_hours
        self.min_impact    = min_impact
        self._cache: list[CalendarEvent] = []
        self._last_fetch: datetime | None = None
        self._cache_ttl_s: int = 4 * 3600   # rafraîchi toutes les 4 heures

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(self) -> list[CalendarEvent]:
        """
        Récupère le calendrier complet (semaine courante + suivante).
        Utilise le cache si < 4 heures.
        """
        now = datetime.now(timezone.utc)
        if (
            self._last_fetch is not None
            and (now - self._last_fetch).total_seconds() < self._cache_ttl_s
        ):
            return self._cache

        events = await self._fetch_from_forexfactory()

        # Fallback si l'API est indisponible
        if not events:
            logger.warning("EconomicCalendar: FF API indisponible, fallback événements récurrents.")
            events = _recurring_events()

        self._cache      = events
        self._last_fetch = now
        logger.debug(f"EconomicCalendar: {len(events)} événements chargés.")
        return events

    def filter_for_symbol(
        self,
        events: list[CalendarEvent],
        symbol: str,
        hours_ahead: float | None = None,
    ) -> list[CalendarEvent]:
        """
        Filtre les événements pertinents pour un symbole donné.

        Args:
            symbol      : ex. "BTC/USDT", "EURUSD"
            hours_ahead : fenêtre temporelle (défaut self.horizon_hours)
        """
        horizon = (hours_ahead or self.horizon_hours) * 60  # en minutes
        now = datetime.now(timezone.utc)
        min_rank = IMPACT_RANK.get(self.min_impact, 1)

        currencies = SYMBOL_CURRENCIES.get(symbol.upper(), ["USD"])

        result = []
        for ev in events:
            if ev.country not in currencies:
                continue
            if ev.impact_rank < min_rank:
                continue
            mins = ev.minutes_until(now)
            if -60 <= mins <= horizon:   # 1h de fenêtre passée incluse
                result.append(ev)

        return sorted(result, key=lambda e: e.event_dt)

    def upcoming_high_impact(
        self,
        events: list[CalendarEvent],
        symbol: str,
        hours_ahead: float = 24.0,
    ) -> list[CalendarEvent]:
        """Raccourci : événements High uniquement."""
        return [
            e for e in self.filter_for_symbol(events, symbol, hours_ahead)
            if e.is_high_impact()
        ]

    # ------------------------------------------------------------------
    # Internal HTTP fetch
    # ------------------------------------------------------------------

    async def _fetch_from_forexfactory(self) -> list[CalendarEvent]:
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp non installé — impossible de fetcher le calendrier FF.")
            return []

        events: list[CalendarEvent] = []
        urls = [_FF_THIS_WEEK, _FF_NEXT_WEEK]

        async with aiohttp.ClientSession() as session:
            for url in urls:
                try:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=10),
                        headers={"User-Agent": "TradoBot/1.0"},
                    ) as resp:
                        if resp.status != 200:
                            logger.warning(f"EconomicCalendar: HTTP {resp.status} pour {url}")
                            continue
                        data = await resp.json(content_type=None)
                        events.extend(self._parse_ff_response(data))
                except Exception as exc:
                    logger.warning(f"EconomicCalendar: erreur fetch {url} — {exc}")

        return events

    def _parse_ff_response(self, data: list[dict]) -> list[CalendarEvent]:
        events = []
        for item in data:
            try:
                dt = _parse_ff_time(
                    item.get("date", ""),
                    item.get("time", ""),
                )
                if dt is None:
                    continue

                impact  = item.get("impact", "Low")
                country = item.get("country", "USD").upper()

                events.append(CalendarEvent(
                    title    = item.get("title", ""),
                    country  = country,
                    event_dt = dt,
                    impact   = impact,
                    forecast = str(item.get("forecast") or ""),
                    previous = str(item.get("previous") or ""),
                    actual   = str(item.get("actual") or ""),
                ))
            except Exception as exc:
                logger.debug(f"EconomicCalendar: erreur parsing item — {exc}")
        return events
