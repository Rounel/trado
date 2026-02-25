"""
tests/unit/test_risk.py — Tests unitaires pour le gestionnaire de risque.
"""
import pytest

from config.settings import Settings, RiskConfig
from trading.risk.circuit_breaker import CircuitBreaker
from trading.risk.sizing import PositionSizer
from trading.risk.manager import RiskManager
from trading.strategies.base import Signal


@pytest.fixture
def settings() -> Settings:
    """Settings de test avec des seuils explicites pour faciliter les tests."""
    s = Settings()
    s.risk = RiskConfig(
        max_drawdown_pct=10.0,
        max_daily_loss_pct=3.0,
        capital_usd=10_000.0,
        max_positions=3,
        kelly_fraction=0.25,
        atr_sl_multiplier=2.0,
        atr_tp_multiplier=6.0,
        min_rr_ratio=3.0,
        trailing_stop_pct=1.5,
        trailing_stop_activation_pct=0.5,
        portfolio_trailing_stop_pct=2.0,
        portfolio_trailing_activation_pct=2.0,
        livermore_enabled=False,   # désactivé dans les tests unitaires de base
        livermore_initial_fraction=0.33,
        livermore_add_trigger_pct=0.5,
    )
    return s


@pytest.fixture
def risk_manager(settings) -> RiskManager:
    return RiskManager(settings)


class TestCircuitBreaker:
    def test_no_trip_normal_conditions(self, settings):
        cb = CircuitBreaker(settings)
        assert cb.update(10_000.0) is True
        # 2% drawdown : sous les deux seuils (max_drawdown=10%, max_daily_loss=3%)
        assert cb.update(9_800.0) is True

    def test_trip_on_max_drawdown(self, settings):
        cb = CircuitBreaker(settings)
        cb.update(10_000.0)
        result = cb.update(8_500.0)  # 15% drawdown → déclenché
        assert result is False
        assert cb.is_paused is True

    def test_trip_on_consecutive_losses(self, settings):
        cb = CircuitBreaker(settings)
        for _ in range(5):
            cb.record_loss()
        result = cb.update(9_800.0)
        assert result is False

    def test_reset_clears_pause(self, settings):
        cb = CircuitBreaker(settings)
        cb.update(8_000.0)  # déclenche
        assert cb.is_paused is True
        cb.reset()
        assert cb.is_paused is False

    def test_win_resets_consecutive_losses(self, settings):
        cb = CircuitBreaker(settings)
        for _ in range(3):
            cb.record_loss()
        cb.record_win()
        assert cb._consecutive_losses == 0

    def test_status_returns_dict(self, settings):
        cb = CircuitBreaker(settings)
        status = cb.status()
        assert "paused" in status
        assert "drawdown_pct" in status


class TestPositionSizer:
    def test_fixed_pct_size(self, settings):
        sizer = PositionSizer(settings)
        qty = sizer.fixed_pct_size(capital=10_000, price=50_000, pct=0.02)
        assert abs(qty - 0.004) < 1e-6

    def test_atr_size_positive(self, settings):
        sizer = PositionSizer(settings)
        qty = sizer.atr_size(capital=10_000, price=50_000, atr=500, risk_pct=0.01)
        assert qty > 0

    def test_zero_price_returns_zero(self, settings):
        sizer = PositionSizer(settings)
        assert sizer.fixed_pct_size(10_000, 0) == 0.0
        assert sizer.atr_size(10_000, 0, 500) == 0.0

    def test_kelly_positive_edge(self, settings):
        sizer = PositionSizer(settings)
        qty = sizer.kelly_size(
            win_rate=0.6, avg_win=200, avg_loss=100,
            capital=10_000, price=50_000
        )
        assert qty > 0

    def test_max_quantity(self, settings):
        sizer = PositionSizer(settings)
        qty = sizer.max_quantity(capital=10_000, price=50_000)
        assert abs(qty - 0.2) < 1e-6


class TestRiskManager:
    def _make_signal(self, action="BUY", confidence=0.8, sl=40_000.0, tp=45_000.0):
        return Signal(
            action=action,
            symbol="BTC/USDT",
            confidence=confidence,
            stop_loss=sl,
            take_profit=tp,
            strategy="test",
        )

    def test_valid_buy_signal_accepted(self, risk_manager):
        signal = self._make_signal("BUY")
        assert risk_manager.validate_signal(signal) is True

    def test_low_confidence_rejected(self, risk_manager):
        signal = self._make_signal(confidence=0.1)
        assert risk_manager.validate_signal(signal) is False

    def test_missing_sl_rejected(self, risk_manager):
        signal = self._make_signal(sl=None)
        assert risk_manager.validate_signal(signal) is False

    def test_missing_tp_rejected(self, risk_manager):
        signal = self._make_signal(tp=None)
        assert risk_manager.validate_signal(signal) is False

    def test_max_positions_blocks_buy(self, settings):
        rm = RiskManager(settings)
        # Remplit les 3 positions max
        for i in range(3):
            rm.record_open(f"SYM{i}", "BUY", 0.1, 50_000)
        signal = self._make_signal("BUY")
        assert rm.validate_signal(signal) is False

    def test_close_position_updates_portfolio(self, risk_manager):
        risk_manager.record_open("BTC/USDT", "BUY", 0.1, 50_000)
        pnl = risk_manager.record_close("BTC/USDT", exit_price=52_000)
        assert pnl == pytest.approx(200.0)

    def test_status_returns_dict(self, risk_manager):
        status = risk_manager.status()
        assert "portfolio_value" in status
        assert "open_positions" in status
        assert "circuit_breaker" in status
