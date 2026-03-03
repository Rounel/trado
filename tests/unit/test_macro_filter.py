"""
tests/unit/test_macro_filter.py — Tests unitaires pour le filtre macro et le context builder.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from analysis.sentiment.context_builder import ContextBuilder
from analysis.sentiment.macro_filter import MacroFilter, MacroContext
from data.collectors.economic_calendar import (
    CalendarEvent,
    EconomicCalendarCollector,
    _parse_ff_time,
    _nfp_date,
    _recurring_events,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_event(
    title: str = "Non-Farm Payrolls",
    country: str = "USD",
    minutes_from_now: float = 20.0,
    impact: str = "High",
) -> CalendarEvent:
    now = datetime.now(timezone.utc)
    return CalendarEvent(
        title=title,
        country=country,
        event_dt=now + timedelta(minutes=minutes_from_now),
        impact=impact,
        forecast="200K",
        previous="150K",
    )


# ---------------------------------------------------------------------------
# EconomicCalendarCollector — parsing
# ---------------------------------------------------------------------------

class TestCalendarParsing:
    def test_parse_ff_time_morning(self):
        dt = _parse_ff_time("2025-01-10T00:00:00-05:00", "8:30am")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.hour == 13  # 8:30 ET = 13:30 UTC

    def test_parse_ff_time_afternoon(self):
        dt = _parse_ff_time("2025-01-10T00:00:00-05:00", "2:00pm")
        assert dt is not None
        assert dt.hour == 19  # 14:00 ET = 19:00 UTC

    def test_parse_ff_time_tentative(self):
        dt = _parse_ff_time("2025-01-10T00:00:00-05:00", "Tentative")
        assert dt is not None  # retourne la date sans heure précise

    def test_parse_ff_time_invalid(self):
        dt = _parse_ff_time("not-a-date", "8:00am")
        assert dt is None

    def test_nfp_date_is_friday(self):
        nfp = _nfp_date(2025, 1)
        assert nfp.weekday() == 4  # vendredi

    def test_nfp_date_is_first_friday(self):
        nfp = _nfp_date(2025, 3)
        assert nfp.day <= 7  # premier vendredi → jour ≤ 7

    def test_recurring_events_returns_list(self):
        events = _recurring_events(horizon_days=35)
        assert isinstance(events, list)
        # Au moins un NFP dans les 35 prochains jours
        nfp_events = [e for e in events if "Non-Farm" in e.title]
        assert len(nfp_events) >= 1

    def test_calendar_event_minutes_until(self):
        ev = _make_event(minutes_from_now=30.0)
        mins = ev.minutes_until()
        assert 28 <= mins <= 32

    def test_calendar_event_str(self):
        ev = _make_event()
        s = str(ev)
        assert "High" in s
        assert "Non-Farm" in s


class TestCalendarCollectorFilter:
    collector = EconomicCalendarCollector(min_impact="Low")

    def test_filter_for_btc_usdt(self):
        events = [_make_event("CPI", "USD", 60), _make_event("ECB Rate", "EUR", 60)]
        result = self.collector.filter_for_symbol(events, "BTC/USDT")
        assert all(e.country == "USD" for e in result)

    def test_filter_for_eurusd(self):
        events = [
            _make_event("CPI", "USD", 60),
            _make_event("ECB Decision", "EUR", 60),
            _make_event("BoE Rate", "GBP", 60),
        ]
        result = self.collector.filter_for_symbol(events, "EURUSD")
        countries = {e.country for e in result}
        assert "USD" in countries
        assert "EUR" in countries
        assert "GBP" not in countries

    def test_filter_excludes_far_future(self):
        # Événement dans 100h → en dehors de horizon_hours=48
        ev = _make_event(minutes_from_now=100 * 60)
        result = self.collector.filter_for_symbol([ev], "BTC/USDT")
        assert len(result) == 0

    def test_filter_includes_recent_past(self):
        # Événement il y a 30 min → inclus dans la fenêtre passée
        ev = _make_event(minutes_from_now=-30.0)
        result = self.collector.filter_for_symbol([ev], "BTC/USDT")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# MacroFilter
# ---------------------------------------------------------------------------

class TestMacroFilter:
    mf = MacroFilter(blackout_before_min=30, blackout_after_min=15)

    def test_blackout_imminent_event(self):
        ev = _make_event(minutes_from_now=20, impact="High")  # dans 20 min < 30 min
        ctx = self.mf.evaluate("BTC/USDT", [ev])
        assert ctx.is_blackout is True
        assert len(ctx.imminent_events) == 1

    def test_no_blackout_far_event(self):
        ev = _make_event(minutes_from_now=120, impact="High")  # dans 2h → pas de blackout
        ctx = self.mf.evaluate("BTC/USDT", [ev])
        assert ctx.is_blackout is False
        assert len(ctx.upcoming_events) >= 1

    def test_no_blackout_low_impact(self):
        ev = _make_event(minutes_from_now=10, impact="Low")
        # MacroFilter avec min_impact="High" → Low ignoré
        mf = MacroFilter(blackout_before_min=30, min_impact="High")
        ctx = mf.evaluate("BTC/USDT", [ev])
        assert ctx.is_blackout is False

    def test_vol_multiplier_increases_near_event(self):
        ev_far  = _make_event(minutes_from_now=200, impact="High")
        ev_near = _make_event(minutes_from_now=10,  impact="High")
        mult_far  = self.mf.evaluate("BTC/USDT", [ev_far]).vol_multiplier
        mult_near = self.mf.evaluate("BTC/USDT", [ev_near]).vol_multiplier
        assert mult_near > mult_far

    def test_vol_multiplier_high_gt_medium(self):
        ev_high   = _make_event(minutes_from_now=60, impact="High")
        ev_medium = _make_event(minutes_from_now=60, impact="Medium")
        mult_high   = self.mf.evaluate("BTC/USDT", [ev_high]).vol_multiplier
        mult_medium = self.mf.evaluate("BTC/USDT", [ev_medium]).vol_multiplier
        assert mult_high >= mult_medium

    def test_wrong_currency_ignored(self):
        # Événement EUR → ne doit pas bloquer BTC/USDT (USD)
        ev = _make_event(minutes_from_now=10, impact="High", country="EUR")
        ctx = self.mf.evaluate("BTC/USDT", [ev])
        assert ctx.is_blackout is False

    def test_blackout_just_after_event(self):
        # Il y a 10 min (< 15 min après) → encore en blackout
        ev = _make_event(minutes_from_now=-10, impact="High")
        ctx = self.mf.evaluate("BTC/USDT", [ev])
        assert ctx.is_blackout is True

    def test_context_has_reason(self):
        ev = _make_event(minutes_from_now=5)
        ctx = self.mf.evaluate("BTC/USDT", [ev])
        assert isinstance(ctx.reason, str)
        assert len(ctx.reason) > 0

    def test_empty_events(self):
        ctx = self.mf.evaluate("BTC/USDT", [])
        assert ctx.is_blackout is False
        assert ctx.vol_multiplier == pytest.approx(1.0)

    def test_shortcut_is_blackout(self):
        ev = _make_event(minutes_from_now=5)
        assert self.mf.is_blackout("BTC/USDT", [ev]) is True

    def test_shortcut_get_vol_multiplier(self):
        mult = self.mf.get_vol_multiplier("BTC/USDT", [])
        assert mult == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

class TestContextBuilder:
    cb = ContextBuilder(max_headlines=3, max_events=2)

    def test_build_with_headlines(self):
        ctx = self.cb.build(
            symbol="BTC/USDT",
            headlines=["Bitcoin surges to new ATH", "Fed holds rates steady"],
        )
        assert "Bitcoin surges" in ctx
        assert "BTC/USDT" in ctx

    def test_build_with_events(self):
        ev = _make_event(minutes_from_now=45)
        ctx = self.cb.build(symbol="BTC/USDT", events=[ev])
        assert "Non-Farm" in ctx
        assert "High" in ctx

    def test_build_with_price_and_trend(self):
        ctx = self.cb.build(
            symbol="ETH/USDT",
            current_price=3500.0,
            trend="bullish",
        )
        assert "3500" in ctx
        assert "bullish" in ctx

    def test_build_empty_returns_fallback(self):
        ctx = self.cb.build(symbol="BTC/USDT")
        assert "No specific context" in ctx

    def test_build_truncates_headlines(self):
        headlines = [f"News {i}" for i in range(10)]
        ctx = self.cb.build(symbol="BTC/USDT", headlines=headlines)
        # max_headlines=3 → seulement 3 titres inclus
        count = sum(1 for h in headlines[:3] if h in ctx)
        assert count == 3

    def test_build_blackout_context(self):
        ev = _make_event(minutes_from_now=5)
        ctx = self.cb.build_blackout_context("BTC/USDT", "NFP imminent", [ev])
        assert "IMPORTANT" in ctx
        assert "High volatility" in ctx
        assert "BTC/USDT" in ctx
