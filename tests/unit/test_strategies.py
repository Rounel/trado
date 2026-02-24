"""
tests/unit/test_strategies.py — Tests unitaires pour les stratégies de trading.
"""
import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from trading.strategies.base import Signal
from trading.strategies.ema_rsi import EmaRsiStrategy


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def strategy(settings) -> EmaRsiStrategy:
    return EmaRsiStrategy(settings=settings, symbol="BTC/USDT")


def _make_bar(close: float = 50_000.0, **kwargs) -> dict:
    """Crée un bar OHLCV factice."""
    return {
        "open":   kwargs.get("open", close * 0.999),
        "high":   kwargs.get("high", close * 1.002),
        "low":    kwargs.get("low",  close * 0.998),
        "close":  close,
        "volume": kwargs.get("volume", 100.0),
    }


def _warmup_strategy(strategy: EmaRsiStrategy, n: int = 215) -> None:
    """Envoie n barres pour dépasser la période de warmup."""
    np.random.seed(0)
    prices = 50_000 + np.cumsum(np.random.randn(n) * 100)
    for p in prices:
        strategy.on_bar(_make_bar(p))


class TestEmaRsiStrategy:
    def test_returns_none_during_warmup(self, strategy):
        """Pas de signal avant les 210 barres de warmup."""
        for _ in range(100):
            result = strategy.on_bar(_make_bar())
        assert result is None

    def test_returns_none_or_signal_after_warmup(self, strategy):
        """Après warmup, retourne None ou un Signal valide."""
        _warmup_strategy(strategy)
        result = strategy.on_bar(_make_bar(50_000))
        assert result is None or isinstance(result, Signal)

    def test_signal_has_required_fields(self, strategy):
        """Un Signal non-None contient toujours action, symbol, confidence."""
        _warmup_strategy(strategy)
        # Force un signal BUY en simulant un contexte haussier
        # (Le signal peut être None ou BUY/SELL selon les données aléatoires)
        for _ in range(5):
            result = strategy.on_bar(_make_bar(51_000))
            if result is not None:
                assert result.action in ("BUY", "SELL", "HOLD")
                assert result.symbol == "BTC/USDT"
                assert 0.0 <= result.confidence <= 1.0
                break

    def test_strategy_name(self, strategy):
        assert strategy.name == "EmaRsiStrategy"

    def test_on_tick_returns_none(self, strategy):
        """on_tick n'est pas implémenté → None."""
        result = strategy.on_tick({"price": 50_000})
        assert result is None

    def test_signal_dataclass(self):
        """Signal est correctement instancié."""
        sig = Signal(
            action="BUY",
            symbol="BTC/USDT",
            confidence=0.75,
            stop_loss=48_000.0,
            take_profit=53_000.0,
            strategy="EmaRsiStrategy",
        )
        assert sig.is_actionable() is True
        assert sig.action == "BUY"

    def test_hold_signal_not_actionable(self):
        sig = Signal(action="HOLD", symbol="BTC/USDT", confidence=0.0)
        assert sig.is_actionable() is False


class TestSignalFusion:
    def test_fusion_buy_threshold(self):
        from analysis.aggregator.signal_fusion import SignalFusion, Decision
        fusion = SignalFusion()
        result = fusion.fuse(
            tech_score=0.8, ai_score=0.7, sentiment_score=0.6, ob_imbalance=0.5
        )
        assert result.decision == Decision.BUY

    def test_fusion_sell_threshold(self):
        from analysis.aggregator.signal_fusion import SignalFusion, Decision
        fusion = SignalFusion()
        result = fusion.fuse(
            tech_score=-0.8, ai_score=-0.7, sentiment_score=-0.6, ob_imbalance=-0.5
        )
        assert result.decision == Decision.SELL

    def test_fusion_hold_neutral(self):
        from analysis.aggregator.signal_fusion import SignalFusion, Decision
        fusion = SignalFusion()
        result = fusion.fuse(
            tech_score=0.05, ai_score=0.05, sentiment_score=0.0, ob_imbalance=0.0
        )
        assert result.decision == Decision.HOLD

    def test_score_bounded(self):
        from analysis.aggregator.signal_fusion import SignalFusion
        fusion = SignalFusion()
        result = fusion.fuse(10.0, 10.0, 10.0, 10.0)
        assert -1.0 <= result.score <= 1.0
