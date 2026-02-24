"""
tests/unit/test_indicators.py — Tests unitaires pour les indicateurs techniques.
"""
import numpy as np
import pandas as pd
import pytest

from analysis.technical.indicators import TechnicalIndicators


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Génère un DataFrame OHLCV synthétique de 250 bougies."""
    np.random.seed(42)
    n = 250
    close = 40_000 + np.cumsum(np.random.randn(n) * 200)
    df = pd.DataFrame({
        "open":   close * (1 + np.random.uniform(-0.002, 0.002, n)),
        "high":   close * (1 + np.random.uniform(0.001, 0.005, n)),
        "low":    close * (1 - np.random.uniform(0.001, 0.005, n)),
        "close":  close,
        "volume": np.random.uniform(100, 500, n),
    })
    return df


class TestTechnicalIndicators:
    def test_add_all_returns_dataframe(self, sample_ohlcv):
        result = TechnicalIndicators.add_all(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_rsi_column_exists(self, sample_ohlcv):
        result = TechnicalIndicators.add_all(sample_ohlcv)
        assert "rsi_14" in result.columns

    def test_rsi_range(self, sample_ohlcv):
        result = TechnicalIndicators.add_all(sample_ohlcv)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_ema_columns_exist(self, sample_ohlcv):
        result = TechnicalIndicators.add_all(sample_ohlcv)
        for col in ("ema_20", "ema_50", "ema_200"):
            assert col in result.columns

    def test_macd_columns_exist(self, sample_ohlcv):
        result = TechnicalIndicators.add_all(sample_ohlcv)
        for col in ("macd", "macd_signal", "macd_hist"):
            assert col in result.columns

    def test_bollinger_bands_exist(self, sample_ohlcv):
        result = TechnicalIndicators.add_all(sample_ohlcv)
        for col in ("bb_upper", "bb_mid", "bb_lower"):
            assert col in result.columns

    def test_atr_positive(self, sample_ohlcv):
        result = TechnicalIndicators.add_all(sample_ohlcv)
        atr = result["atr_14"].dropna()
        assert (atr > 0).all()

    def test_bb_upper_gt_lower(self, sample_ohlcv):
        result = TechnicalIndicators.add_all(sample_ohlcv).dropna()
        assert (result["bb_upper"] >= result["bb_lower"]).all()

    def test_rsi_signal_oversold(self):
        assert TechnicalIndicators.rsi_signal(25.0) == "oversold"

    def test_rsi_signal_overbought(self):
        assert TechnicalIndicators.rsi_signal(75.0) == "overbought"

    def test_rsi_signal_neutral(self):
        assert TechnicalIndicators.rsi_signal(50.0) == "neutral"

    def test_ema_trend_strong_uptrend(self):
        trend = TechnicalIndicators.ema_trend(100, 98, 95, 90)
        assert trend == "strong_uptrend"

    def test_ema_trend_strong_downtrend(self):
        trend = TechnicalIndicators.ema_trend(80, 85, 90, 95)
        assert trend == "strong_downtrend"

    def test_atr_stops_structure(self):
        stops = TechnicalIndicators.atr_stops(close=50_000, atr=1000)
        assert "stop_loss_long" in stops
        assert "take_profit_long" in stops
        assert stops["stop_loss_long"] < 50_000
        assert stops["take_profit_long"] > 50_000

    def test_original_df_not_modified(self, sample_ohlcv):
        original_cols = set(sample_ohlcv.columns)
        TechnicalIndicators.add_all(sample_ohlcv)
        assert set(sample_ohlcv.columns) == original_cols
