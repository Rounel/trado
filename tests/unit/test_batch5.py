"""
tests/unit/test_batch5.py — Tests unitaires du Batch 5.

Couvre :
  5a. FinBERT INT8 quantization (sans télécharger le modèle)
  5b. VWAP order fragmentation (BinanceExecutor mock)
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ──────────────────────────────────────────────
# 5a — FinBERT quantization
# ──────────────────────────────────────────────

class TestFinBERTQuantization:
    """Teste la logique de l'analyseur sans charger le vrai modèle."""

    def test_score_returns_zero_when_unavailable(self):
        """Sans transformers (ou simulé indispo), score doit retourner 0.0."""
        from analysis.sentiment.finbert_analyzer import FinBERTAnalyzer
        analyzer = FinBERTAnalyzer()
        # Forcer is_available = False
        analyzer._available = False
        assert analyzer.score("Apple beats earnings estimates") == 0.0

    def test_score_batch_returns_zeros_when_unavailable(self):
        from analysis.sentiment.finbert_analyzer import FinBERTAnalyzer
        analyzer = FinBERTAnalyzer()
        analyzer._available = False
        results = analyzer.score_batch(["text1", "text2", "text3"])
        assert results == [0.0, 0.0, 0.0]

    def test_score_empty_text_returns_zero(self):
        from analysis.sentiment.finbert_analyzer import FinBERTAnalyzer
        analyzer = FinBERTAnalyzer()
        analyzer._available = True  # même si dispo, texte vide → 0.0
        assert analyzer.score("") == 0.0

    def test_cache_hit_skips_pipeline(self):
        """Un texte déjà évalué doit être servi depuis le cache sans rappeler le pipeline."""
        from analysis.sentiment.finbert_analyzer import FinBERTAnalyzer
        analyzer = FinBERTAnalyzer(use_cache=True)
        analyzer._available = True
        analyzer._cache["fed hikes rates"] = 0.42

        with patch("analysis.sentiment.finbert_analyzer._load_pipeline") as mock_pipe:
            result = analyzer.score("fed hikes rates")
            mock_pipe.assert_not_called()
        assert result == pytest.approx(0.42)

    def test_no_cache_mode_always_calls_pipeline(self):
        """use_cache=False doit toujours appeler le pipeline même si texte déjà vu."""
        from analysis.sentiment.finbert_analyzer import FinBERTAnalyzer
        analyzer = FinBERTAnalyzer(use_cache=False)
        analyzer._available = True

        fake_results = [[{"label": "positive", "score": 0.8},
                         {"label": "negative", "score": 0.1},
                         {"label": "neutral",  "score": 0.1}]]
        mock_pipeline = MagicMock(return_value=fake_results)

        with patch("analysis.sentiment.finbert_analyzer._load_pipeline", return_value=mock_pipeline):
            s1 = analyzer.score("bull run")
            s2 = analyzer.score("bull run")  # même texte
        assert mock_pipeline.call_count == 2

    def test_score_computation_positive(self):
        """Vérifie P(positive) - P(negative) pour un texte bullish."""
        from analysis.sentiment.finbert_analyzer import FinBERTAnalyzer
        analyzer = FinBERTAnalyzer(use_cache=False)
        analyzer._available = True

        fake_results = [[{"label": "positive", "score": 0.90},
                         {"label": "negative", "score": 0.05},
                         {"label": "neutral",  "score": 0.05}]]
        mock_pipeline = MagicMock(return_value=fake_results)

        with patch("analysis.sentiment.finbert_analyzer._load_pipeline", return_value=mock_pipeline):
            score = analyzer.score("record profits and strong guidance")

        assert score == pytest.approx(0.85, abs=1e-6)
        assert 0.0 < score <= 1.0

    def test_score_computation_negative(self):
        """Vérifie le score pour un texte bearish."""
        from analysis.sentiment.finbert_analyzer import FinBERTAnalyzer
        analyzer = FinBERTAnalyzer(use_cache=False)
        analyzer._available = True

        fake_results = [[{"label": "positive", "score": 0.05},
                         {"label": "negative", "score": 0.88},
                         {"label": "neutral",  "score": 0.07}]]
        mock_pipeline = MagicMock(return_value=fake_results)

        with patch("analysis.sentiment.finbert_analyzer._load_pipeline", return_value=mock_pipeline):
            score = analyzer.score("massive losses and bankruptcy risk")

        assert score == pytest.approx(-0.83, abs=1e-6)
        assert -1.0 <= score < 0.0

    def test_score_clamped_to_minus_one_plus_one(self):
        """Le score doit toujours rester dans [-1, +1]."""
        from analysis.sentiment.finbert_analyzer import FinBERTAnalyzer
        analyzer = FinBERTAnalyzer(use_cache=False)
        analyzer._available = True

        # Probabilités anormalement élevées (somme > 1 simulée)
        fake_results = [[{"label": "positive", "score": 0.99},
                         {"label": "negative", "score": 0.0},
                         {"label": "neutral",  "score": 0.01}]]
        mock_pipeline = MagicMock(return_value=fake_results)

        with patch("analysis.sentiment.finbert_analyzer._load_pipeline", return_value=mock_pipeline):
            score = analyzer.score("extremely bullish")

        assert -1.0 <= score <= 1.0

    def test_label_mapping(self):
        from analysis.sentiment.finbert_analyzer import FinBERTAnalyzer
        a = FinBERTAnalyzer()
        assert a.label(0.7)   == "very_bullish"
        assert a.label(0.3)   == "bullish"
        assert a.label(0.0)   == "neutral"
        assert a.label(-0.3)  == "bearish"
        assert a.label(-0.7)  == "very_bearish"

    def test_quantize_flag_passed_to_load_pipeline(self):
        """Le flag quantize doit être transmis à _load_pipeline."""
        from analysis.sentiment.finbert_analyzer import FinBERTAnalyzer
        analyzer = FinBERTAnalyzer(use_cache=False, quantize=False)
        analyzer._available = True

        fake_results = [[{"label": "neutral", "score": 0.9},
                         {"label": "positive", "score": 0.05},
                         {"label": "negative", "score": 0.05}]]
        mock_pipeline = MagicMock(return_value=fake_results)

        with patch("analysis.sentiment.finbert_analyzer._load_pipeline", return_value=mock_pipeline) as mock_load:
            analyzer.score("markets are calm")
            mock_load.assert_called_once_with(False)

    def test_quantize_true_flag_passed(self):
        from analysis.sentiment.finbert_analyzer import FinBERTAnalyzer
        analyzer = FinBERTAnalyzer(use_cache=False, quantize=True)
        analyzer._available = True

        fake_results = [[{"label": "neutral", "score": 0.9},
                         {"label": "positive", "score": 0.05},
                         {"label": "negative", "score": 0.05}]]
        mock_pipeline = MagicMock(return_value=fake_results)

        with patch("analysis.sentiment.finbert_analyzer._load_pipeline", return_value=mock_pipeline) as mock_load:
            analyzer.score("steady markets")
            mock_load.assert_called_once_with(True)


# ──────────────────────────────────────────────
# 5b — VWAP fragmentation
# ──────────────────────────────────────────────

def _make_settings(vwap_threshold=10_000.0, vwap_slices=3, vwap_slice_delay_s=0.0, testnet=True):
    """Construit un Settings minimal avec une config Binance paramétrable."""
    settings = MagicMock()
    settings.binance.api_key  = "test"
    settings.binance.secret   = "test"
    settings.binance.testnet  = testnet
    settings.binance.vwap_threshold_usd  = vwap_threshold
    settings.binance.vwap_slices         = vwap_slices
    settings.binance.vwap_slice_delay_s  = vwap_slice_delay_s
    return settings


def _make_signal(symbol="BTC/USDT", action="BUY", size=0.3, entry_price=50_000.0,
                 stop_loss=49_000.0, take_profit=56_000.0):
    from trading.strategies.base import Signal
    return Signal(
        action=action,
        symbol=symbol,
        confidence=0.8,
        stop_loss=stop_loss,
        take_profit=take_profit,
        size=size,
        strategy="test",
        entry_price=entry_price,
    )


def _make_order_response(order_id: str, filled: float, price: float):
    return {"id": order_id, "status": "closed", "filled": filled, "average": price, "price": price}


class TestVWAPFragmentation:

    def test_single_order_below_threshold(self):
        """Ordre < seuil VWAP → chemin _submit_single (un seul appel exchange)."""
        settings = _make_settings(vwap_threshold=100_000.0)  # seuil très haut
        signal   = _make_signal(size=0.1, entry_price=50_000.0)  # 5 000 $ < 100 000$

        from trading.executor.binance import BinanceExecutor
        executor = BinanceExecutor(settings)

        mock_exchange = AsyncMock()
        mock_exchange.create_market_order = AsyncMock(
            return_value=_make_order_response("o1", 0.1, 50_000.0)
        )
        mock_exchange.create_order = AsyncMock()  # SL

        async def run():
            executor._exchange = mock_exchange
            return await executor.submit(signal)

        result = asyncio.get_event_loop().run_until_complete(run())
        assert mock_exchange.create_market_order.call_count == 1
        assert result["price"] == 50_000.0
        assert result.get("vwap") is None  # pas de VWAP

    def test_vwap_triggered_above_threshold(self):
        """Ordre >= seuil VWAP → 3 appels exchange (vwap_slices=3)."""
        settings = _make_settings(vwap_threshold=10_000.0, vwap_slices=3, vwap_slice_delay_s=0.0)
        signal   = _make_signal(size=0.3, entry_price=50_000.0)  # 15 000 $ >= 10 000 $

        from trading.executor.binance import BinanceExecutor
        executor = BinanceExecutor(settings)

        call_count = 0
        async def fake_market_order(symbol, side, amount, params=None):
            nonlocal call_count
            call_count += 1
            return _make_order_response(f"o{call_count}", amount, 50_000.0)

        mock_exchange = AsyncMock()
        mock_exchange.create_market_order = fake_market_order
        mock_exchange.create_order        = AsyncMock()  # SL

        async def run():
            executor._exchange = mock_exchange
            return await executor.submit(signal)

        result = asyncio.get_event_loop().run_until_complete(run())

        assert call_count == 3, f"Attendu 3 tranches, eu {call_count}"
        assert result["vwap"] is True
        assert result["slices"] == 3
        assert result["filled"] == pytest.approx(0.3, abs=1e-6)
        assert result["price"]  == pytest.approx(50_000.0)

    def test_vwap_average_price_weighted(self):
        """Prix moyen VWAP doit être la moyenne pondérée des prix de chaque tranche."""
        settings = _make_settings(vwap_threshold=0.01, vwap_slices=3, vwap_slice_delay_s=0.0)
        signal   = _make_signal(size=0.3, entry_price=50_000.0)

        from trading.executor.binance import BinanceExecutor
        executor = BinanceExecutor(settings)

        prices = [49_900.0, 50_000.0, 50_100.0]
        call_idx = [0]

        async def fake_market_order(symbol, side, amount, params=None):
            p = prices[call_idx[0] % 3]
            call_idx[0] += 1
            return _make_order_response(f"o{call_idx[0]}", amount, p)

        mock_exchange = AsyncMock()
        mock_exchange.create_market_order = fake_market_order
        mock_exchange.create_order        = AsyncMock()

        async def run():
            executor._exchange = mock_exchange
            return await executor.submit(signal)

        result = asyncio.get_event_loop().run_until_complete(run())

        # Les 3 tranches = 0.1 chacune → moyenne pondérée = (49900+50000+50100)/3 = 50000
        assert result["price"] == pytest.approx(50_000.0, abs=1.0)

    def test_vwap_disabled_when_threshold_zero(self):
        """vwap_threshold_usd=0 désactive VWAP → chemin single même pour gros ordres."""
        settings = _make_settings(vwap_threshold=0.0)
        signal   = _make_signal(size=10.0, entry_price=50_000.0)  # 500 000 $

        from trading.executor.binance import BinanceExecutor
        executor = BinanceExecutor(settings)

        mock_exchange = AsyncMock()
        mock_exchange.create_market_order = AsyncMock(
            return_value=_make_order_response("o1", 10.0, 50_000.0)
        )
        mock_exchange.create_order = AsyncMock()

        async def run():
            executor._exchange = mock_exchange
            return await executor.submit(signal)

        result = asyncio.get_event_loop().run_until_complete(run())
        assert mock_exchange.create_market_order.call_count == 1
        assert result.get("vwap") is None

    def test_vwap_last_slice_absorbs_rounding(self):
        """La dernière tranche doit prendre le reliquat exact pour que filled == size."""
        settings = _make_settings(vwap_threshold=0.01, vwap_slices=3, vwap_slice_delay_s=0.0)
        # 0.1 / 3 = 0.033333... → arrondi peut créer un écart
        signal   = _make_signal(size=0.1, entry_price=1.0)

        from trading.executor.binance import BinanceExecutor
        executor = BinanceExecutor(settings)

        async def fake_market_order(symbol, side, amount, params=None):
            return _make_order_response("ox", amount, 1.0)

        mock_exchange = AsyncMock()
        mock_exchange.create_market_order = fake_market_order
        mock_exchange.create_order        = AsyncMock()

        async def run():
            executor._exchange = mock_exchange
            return await executor.submit(signal)

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result["filled"] == pytest.approx(0.1, abs=1e-7)

    def test_vwap_partial_fill_on_error(self):
        """Si une tranche échoue, le résultat agrège les tranches réussies."""
        settings = _make_settings(vwap_threshold=0.01, vwap_slices=3, vwap_slice_delay_s=0.0)
        signal   = _make_signal(size=0.3, entry_price=1.0)

        from trading.executor.binance import BinanceExecutor
        executor = BinanceExecutor(settings)

        call_idx = [0]

        async def fake_market_order(symbol, side, amount, params=None):
            call_idx[0] += 1
            if call_idx[0] == 2:  # La 2e tranche échoue
                raise RuntimeError("exchange error")
            return _make_order_response(f"o{call_idx[0]}", amount, 1.0)

        mock_exchange = AsyncMock()
        mock_exchange.create_market_order = fake_market_order
        mock_exchange.create_order        = AsyncMock()

        async def run():
            executor._exchange = mock_exchange
            return await executor.submit(signal)

        result = asyncio.get_event_loop().run_until_complete(run())
        # La tranche 2 échoue, mais la tranche 3 (dernière) absorbe le reliquat (0.2)
        # → 2 fills réussis, total_filled = 0.3 (tranche 1: 0.1 + tranche 3: 0.2)
        assert result["slices"] == 2
        assert result["filled"] == pytest.approx(0.3, abs=1e-6)

    def test_hold_signal_skipped(self):
        """Un signal HOLD ne doit pas déclencher d'ordre."""
        settings = _make_settings()
        from trading.strategies.base import Signal
        from trading.executor.binance import BinanceExecutor

        signal   = Signal(action="HOLD", symbol="BTC/USDT", confidence=0.5)
        executor = BinanceExecutor(settings)

        mock_exchange = AsyncMock()
        executor._exchange = mock_exchange

        async def run():
            return await executor.submit(signal)

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result["status"] == "skipped"
        mock_exchange.create_market_order.assert_not_called()

    def test_zero_qty_skipped(self):
        """Un signal avec size=0 doit être ignoré."""
        settings = _make_settings()
        from trading.executor.binance import BinanceExecutor
        signal   = _make_signal(size=0.0)
        executor = BinanceExecutor(settings)

        mock_exchange = AsyncMock()
        executor._exchange = mock_exchange

        async def run():
            return await executor.submit(signal)

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result["status"] == "skipped"
        mock_exchange.create_market_order.assert_not_called()
