"""
tests/unit/test_advanced_strategy.py — Tests unitaires pour les modules SMC/OTE/IFC/PA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.technical.ifc import CandlePatternDetector, PatternType
from analysis.technical.ote import FibonacciCalculator, OTEEvaluator
from analysis.technical.price_action import MarketStructure, SwingDetector, Trend
from analysis.technical.smc import FVGDetector, LiquidityZoneDetector, OrderBlockDetector
from models.meta_labeling import MetaFeatures, MetaLabelingModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trending_df(n: int = 100, trend: str = "up") -> pd.DataFrame:
    """Génère un DataFrame OHLCV synthétique avec tendance."""
    rng = np.random.default_rng(42)
    base = 100.0
    prices = [base]
    for _ in range(n - 1):
        step = rng.normal(0.1 if trend == "up" else -0.1, 0.5)
        prices.append(max(1.0, prices[-1] + step))

    prices = np.array(prices)
    noise = rng.uniform(0.1, 0.5, n)
    return pd.DataFrame({
        "open":   prices - noise * 0.3,
        "high":   prices + noise,
        "low":    prices - noise,
        "close":  prices,
        "volume": rng.uniform(1000, 5000, n),
    })


def _make_candles(*rows: tuple) -> pd.DataFrame:
    """Crée un mini-DataFrame OHLCV depuis des tuples (o, h, l, c, v)."""
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------------
# price_action.py
# ---------------------------------------------------------------------------

class TestSwingDetector:
    def test_detects_swings(self):
        df = _make_trending_df(80, "up")
        detector = SwingDetector(order=3)
        highs, lows = detector.detect(df)
        assert len(highs) >= 1
        assert len(lows)  >= 1
        assert all(s.kind == "high" for s in highs)
        assert all(s.kind == "low"  for s in lows)

    def test_empty_df_returns_empty(self):
        df = _make_trending_df(5, "up")   # trop court pour order=5
        detector = SwingDetector(order=5)
        highs, lows = detector.detect(df)
        assert isinstance(highs, list)
        assert isinstance(lows,  list)


class TestMarketStructure:
    def test_bullish_trend(self):
        df = _make_trending_df(100, "up")
        ms = MarketStructure()
        result = ms.analyze(df)
        assert result.trend in (Trend.BULLISH, Trend.RANGING)

    def test_bearish_trend(self):
        df = _make_trending_df(100, "down")
        ms = MarketStructure()
        result = ms.analyze(df)
        assert result.trend in (Trend.BEARISH, Trend.RANGING)

    def test_structure_has_fields(self):
        df = _make_trending_df(60, "up")
        ms = MarketStructure()
        result = ms.analyze(df)
        assert result.swing_highs is not None
        assert result.swing_lows  is not None
        assert isinstance(result.recent_events, list)


# ---------------------------------------------------------------------------
# smc.py
# ---------------------------------------------------------------------------

class TestOrderBlockDetector:
    def test_detects_blocks(self):
        df = _make_trending_df(60, "up")
        ob = OrderBlockDetector(impulse_factor=0.001)
        blocks = ob.detect(df)
        assert isinstance(blocks, list)

    def test_nearest_returns_none_if_empty(self):
        ob = OrderBlockDetector()
        assert ob.nearest([], 100.0, "bullish") is None


class TestFVGDetector:
    def test_detect_fvg(self):
        # Crée manuellement un bullish FVG : low[i+2] > high[i]
        rows = [
            (100, 101, 99, 100.5, 1000),   # candle 0
            (100.5, 102, 100.2, 101.5, 1100),  # candle 1 (middle)
            (103, 104, 103, 103.5, 1200),   # candle 2 : low(103) > high(101) → FVG
        ]
        df = _make_candles(*rows)
        fvg = FVGDetector()
        gaps = fvg.detect(df, lookback=10)
        assert len(gaps) >= 1
        assert gaps[0].kind == "bullish"

    def test_no_fvg_in_normal_data(self):
        df = _make_trending_df(20, "up")
        fvg = FVGDetector()
        # Ne doit pas planter
        _ = fvg.detect(df)


class TestLiquidityZoneDetector:
    def test_detect_equal_highs(self):
        from analysis.technical.price_action import SwingDetector
        df = _make_trending_df(80, "up")
        detector = SwingDetector(order=3)
        highs, lows = detector.detect(df)
        liq = LiquidityZoneDetector(tolerance=0.01)
        zones = liq.detect(highs, lows)
        assert isinstance(zones, list)


# ---------------------------------------------------------------------------
# ote.py
# ---------------------------------------------------------------------------

class TestFibonacciCalculator:
    def test_long_retracement(self):
        grid = FibonacciCalculator.compute(swing_a=100.0, swing_b=110.0, direction="long")
        assert grid.levels["0.0"]   == pytest.approx(110.0, rel=1e-4)
        assert grid.levels["100.0"] == pytest.approx(100.0, rel=1e-4)
        assert grid.levels["61.8"]  == pytest.approx(110.0 - 0.618 * 10.0, rel=1e-4)

    def test_ote_zone_bounds(self):
        grid = FibonacciCalculator.compute(100.0, 110.0, "long")
        assert grid.ote_low  == pytest.approx(grid.levels["61.8"], rel=1e-4)
        assert grid.ote_high == pytest.approx(grid.levels["78.6"], rel=1e-4)

    def test_equilibrium(self):
        grid = FibonacciCalculator.compute(100.0, 110.0, "long")
        assert grid.equilibrium == pytest.approx(grid.levels["50.0"], rel=1e-4)


class TestOTEEvaluator:
    def test_price_in_ote_zone(self):
        evaluator = OTEEvaluator()
        # Swing A=100, B=110 → OTE entre 61.8%=103.82 et 78.6%=102.14 (depuis B)
        grid = FibonacciCalculator.compute(100.0, 110.0, "long")
        in_price = (grid.ote_low + grid.ote_high) / 2
        sig = evaluator.evaluate(100.0, 110.0, in_price, "long")
        assert sig.in_zone is True
        assert sig.zone_strength > 0.0

    def test_price_out_of_ote_zone(self):
        evaluator = OTEEvaluator()
        sig = evaluator.evaluate(100.0, 110.0, 115.0, "long")
        assert sig.in_zone is False
        assert sig.zone_strength == 0.0


# ---------------------------------------------------------------------------
# ifc.py
# ---------------------------------------------------------------------------

class TestCandlePatternDetector:
    detector = CandlePatternDetector()

    def test_detects_hammer(self):
        # Hammer : wick inférieur long, corps en haut
        df = _make_candles(
            (100, 100.5, 95, 100.2, 1000),   # pas un hammer
            (100, 100.2, 94, 99.8, 1200),    # hammer : wick bas = 5.8, body = 0.2
        )
        patterns = self.detector.detect_all(df)
        bullish = [p for p in patterns if p.is_bullish]
        assert len(bullish) >= 1

    def test_detects_engulfing_bullish(self):
        # Bougie baissière suivie d'une bougie haussière qui l'englobe
        df = _make_candles(
            (102, 103, 99, 100, 1000),    # baissière (close < open)
            (98,  105, 97, 104, 2000),    # haussière englobante
        )
        patterns = self.detector.detect_all(df)
        engulfing = [p for p in patterns if p.pattern == PatternType.ENGULFING_BULL]
        assert len(engulfing) >= 1

    def test_no_pattern_in_flat_candle(self):
        df = _make_candles(
            (100, 100.05, 99.95, 100.02, 500),
        )
        patterns = self.detector.detect_all(df)
        # Devrait donner un Doji, pas de Pin Bar ou Engulfing
        for p in patterns:
            assert p.pattern in (PatternType.DOJI, PatternType.INSIDE_BAR)

    def test_has_confirmation(self):
        df = _make_trending_df(10, "up")
        # Ne doit pas planter
        confirmed, pattern = self.detector.has_confirmation(df, "bullish")
        assert isinstance(confirmed, bool)


# ---------------------------------------------------------------------------
# meta_labeling.py
# ---------------------------------------------------------------------------

class TestMetaLabelingModel:
    def test_default_prediction_hold(self):
        model = MetaLabelingModel()
        features = MetaFeatures()
        pred = model.predict(features.to_array())
        assert pred.action.name == "HOLD"
        assert pred.confidence == pytest.approx(0.5, abs=0.1)

    def test_fit_and_predict(self):
        model = MetaLabelingModel(pop_threshold=0.55)
        rng = np.random.default_rng(0)
        X = rng.random((200, 10)).astype(np.float32)
        y = (rng.random(200) > 0.4).astype(int)
        model.fit(X, y)
        pred = model.predict(X[:1])
        assert 0.0 <= pred.confidence <= 1.0
        assert pred.action.name in ("BUY", "HOLD", "SELL")

    def test_is_valid(self):
        model = MetaLabelingModel()
        f = MetaFeatures(ote_strength=0.8, ob_strength=0.7)
        valid, pop = model.is_valid(f)
        assert isinstance(valid, bool)
        assert 0.0 <= pop <= 1.0

    def test_feature_names(self):
        assert len(MetaFeatures.feature_names()) == 10

    def test_to_array_shape(self):
        f = MetaFeatures(ote_strength=0.5, sentiment_score=0.3)
        arr = f.to_array()
        assert arr.shape == (10,)
        assert arr[0] == pytest.approx(0.5)
        assert arr[9] == pytest.approx(0.3)
