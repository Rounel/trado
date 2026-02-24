"""
tests/integration/test_pipeline.py — Tests d'intégration du pipeline complet.

Ces tests vérifient que les composants s'enchaînent correctement sans appel réseau.
Les appels broker/API sont mockés.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """DataFrame OHLCV de 250 bougies pour les tests."""
    np.random.seed(7)
    n = 250
    close = 40_000 + np.cumsum(np.random.randn(n) * 150)
    return pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.003,
        "low":    close * 0.997,
        "close":  close,
        "volume": np.random.uniform(50, 300, n),
    })


class TestFeaturePipeline:
    def test_feature_builder_adds_columns(self, sample_ohlcv_df):
        from data.pipeline.features import FeatureBuilder
        result = FeatureBuilder.build(sample_ohlcv_df)
        assert "returns" in result.columns
        assert "log_returns" in result.columns
        assert "sma_20" in result.columns

    def test_feature_builder_no_na(self, sample_ohlcv_df):
        from data.pipeline.features import FeatureBuilder
        result = FeatureBuilder.build(sample_ohlcv_df)
        assert not result.isnull().any().any()

    def test_normalizer_transforms(self, sample_ohlcv_df):
        from data.pipeline.features import FeatureBuilder
        from data.pipeline.normalizer import Normalizer

        df = FeatureBuilder.build(sample_ohlcv_df)
        norm = Normalizer(method="zscore")
        feature_cols = FeatureBuilder.get_feature_columns()
        available = [c for c in feature_cols if c in df.columns]
        transformed = norm.fit_transform(df, columns=available)
        assert transformed[available].mean().abs().max() < 1.0


class TestBacktestMetrics:
    def test_compute_metrics_basic(self):
        from backtest.metrics import compute_metrics
        equity = pd.Series([10_000, 10_100, 10_050, 10_200, 10_180, 10_300])
        trades = [100.0, -50.0, 150.0, -20.0]
        metrics = compute_metrics(equity, trades)

        assert metrics.total_trades == 4
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 1
        assert metrics.profit_factor > 1.0
        assert metrics.win_rate == pytest.approx(0.75)

    def test_summary_contains_sharpe(self):
        from backtest.metrics import compute_metrics
        equity = pd.Series([10_000, 10_200, 10_400])
        metrics = compute_metrics(equity, [200.0, 200.0])
        summary = metrics.summary()
        assert "Sharpe" in summary

    def test_metrics_all_losses(self):
        from backtest.metrics import compute_metrics
        equity = pd.Series([10_000, 9_900, 9_800])
        metrics = compute_metrics(equity, [-100.0, -100.0])
        assert metrics.total_return_pct < 0
        assert metrics.win_rate == 0.0


class TestIndicatorsPipeline:
    def test_full_indicator_pipeline(self, sample_ohlcv_df):
        from analysis.technical.indicators import TechnicalIndicators
        result = TechnicalIndicators.add_all(sample_ohlcv_df)
        assert len(result) > 0
        assert result["rsi_14"].between(0, 100).all()

    def test_tech_score_from_indicators(self, sample_ohlcv_df):
        from analysis.technical.indicators import TechnicalIndicators
        from analysis.aggregator.signal_fusion import SignalFusion

        df = TechnicalIndicators.add_all(sample_ohlcv_df).dropna()
        last = df.iloc[-1].to_dict()
        score = SignalFusion.tech_score_from_indicators(last)
        assert -1.0 <= score <= 1.0


class TestRLEnvironment:
    def test_env_reset(self):
        from models.rl_agent.environment import TradingEnv
        np.random.seed(0)
        prices = 40_000 + np.cumsum(np.random.randn(200) * 100)
        features = np.random.randn(200, 10).astype(np.float32)
        env = TradingEnv(prices=prices, features=features, window_size=20)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape

    def test_env_step(self):
        from models.rl_agent.environment import TradingEnv
        np.random.seed(1)
        prices = 40_000 + np.cumsum(np.random.randn(200) * 100)
        features = np.random.randn(200, 5).astype(np.float32)
        env = TradingEnv(prices=prices, features=features, window_size=10)
        env.reset()
        obs, reward, done, truncated, info = env.step(2)  # BUY
        assert isinstance(reward, float)
        assert "portfolio_value" in info

    def test_env_full_episode(self):
        from models.rl_agent.environment import TradingEnv
        np.random.seed(2)
        n = 150
        prices = 40_000 + np.cumsum(np.random.randn(n) * 100)
        features = np.random.randn(n, 5).astype(np.float32)
        env = TradingEnv(prices=prices, features=features, window_size=10)
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
        assert done or steps == 200
