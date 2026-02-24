"""
core/engine.py — Orchestrateur async principal de TRADO.

Boucle de vie :
  start() → une tâche asyncio PAR symbole, en parallèle
           → collecte → features [+ sentiment + OB si ai] → signal → risk → executor
  stop()  → annulation propre de toutes les tâches
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings

# Fréquence de rafraîchissement du sentiment Grok (secondes)
# Évite d'appeler l'API Grok à chaque bougie (coûteux + lent)
_SENTIMENT_REFRESH_S = 900   # 15 minutes

# Symboles par défaut selon le broker
_DEFAULT_SYMBOLS: dict[str, list[str]] = {
    "binance": ["BTC/USDT"],
    "alpaca":  ["AAPL"],
    "mt5":     ["EURUSD"],
}


class TradoEngine:
    """Orchestrateur central — supporte plusieurs symboles en parallèle."""

    def __init__(
        self,
        settings: "Settings",
        strategy_name: str = "ema_rsi",
        broker: str = "binance",
        symbols: list[str] | None = None,
        timeframe: str = "1h",
    ) -> None:
        self.settings = settings
        self.strategy_name = strategy_name
        self.broker = broker
        self.symbols = symbols or _DEFAULT_SYMBOLS.get(broker, ["BTC/USDT"])
        self.timeframe = timeframe
        self._running = False
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Démarre une boucle de trading par symbole, toutes en parallèle."""
        from monitoring.logger import setup_logger
        setup_logger(self.settings)

        logger.info(
            f"TradoEngine starting — strategy={self.strategy_name} broker={self.broker} "
            f"env={self.settings.env} symbols={self.symbols} timeframe={self.timeframe}"
        )

        self._running = True

        # L'exécuteur est partagé entre tous les symboles (une seule connexion broker)
        executor = self._build_executor()

        # Une tâche asyncio par symbole
        self._tasks = [
            asyncio.create_task(
                self._run_symbol(symbol, executor),
                name=f"trado:{symbol}",
            )
            for symbol in self.symbols
        ]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("TradoEngine: tâches annulées proprement")
        except Exception as exc:
            logger.exception(f"TradoEngine: erreur inattendue — {exc}")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Arrête proprement tous les symboles."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("TradoEngine stopped.")

    # ------------------------------------------------------------------
    # Boucle par symbole
    # ------------------------------------------------------------------

    async def _run_symbol(self, symbol: str, executor) -> None:
        """Boucle indépendante pour un symbole donné."""
        collector = self._build_collector()
        strategy  = self._build_strategy(symbol)
        risk_mgr  = self._build_risk_manager()

        # Enrichissement AI : sentiment Grok + order book (uniquement pour AIStrategy)
        grok         = None
        ob_collector = None
        if self.strategy_name == "ai":
            from analysis.sentiment.grok_analyzer import GrokAnalyzer
            from data.collectors.orderbook import OrderBookCollector
            grok         = GrokAnalyzer(self.settings)
            ob_collector = OrderBookCollector(self.settings)

        # Cache sentiment par symbole : (score, timestamp)
        sentiment_cache: dict[str, tuple[float, float]] = {}

        logger.info(f"[{symbol}] boucle démarrée ({self.timeframe})")
        try:
            async for bar in collector.stream(symbol=symbol, timeframe=self.timeframe):
                if not self._running:
                    break
                await self._on_bar(
                    bar, strategy, risk_mgr, executor,
                    grok=grok,
                    ob_collector=ob_collector,
                    sentiment_cache=sentiment_cache,
                )
        except asyncio.CancelledError:
            logger.info(f"[{symbol}] boucle arrêtée")
        except Exception as exc:
            logger.exception(f"[{symbol}] erreur — {exc}")

    # ------------------------------------------------------------------
    # Callbacks internes
    # ------------------------------------------------------------------

    async def _on_bar(
        self,
        bar: dict,
        strategy,
        risk_mgr,
        executor,
        grok=None,
        ob_collector=None,
        sentiment_cache: dict | None = None,
    ) -> None:
        """Traite une nouvelle bougie OHLCV, enrichit avec sentiment + OB si AIStrategy."""
        from data.pipeline.features import FeatureBuilder
        features = FeatureBuilder.from_bar(bar)
        symbol   = bar.get("symbol", "")

        # Enrichissement async (sentiment + order book en parallèle)
        if grok is not None or ob_collector is not None:
            sentiment_score, ob_imbalance = await self._fetch_enrichments(
                symbol, grok, ob_collector, sentiment_cache or {}
            )
            features["sentiment_score"] = sentiment_score
            features["ob_imbalance"]    = ob_imbalance

        signal = strategy.on_bar(features)
        if signal is None:
            return

        if not risk_mgr.validate_signal(signal):
            logger.debug(f"Signal refusé par RiskManager : {signal}")
            return

        logger.info(f"Exécution du signal : {signal}")
        await executor.submit(signal)

    async def _fetch_enrichments(
        self,
        symbol: str,
        grok,
        ob_collector,
        sentiment_cache: dict,
    ) -> tuple[float, float]:
        """Récupère sentiment et imbalance OB en parallèle. Cache le sentiment 15 min."""
        now = time.monotonic()

        # Sentiment : utilise le cache si récent
        cached = sentiment_cache.get(symbol)
        if cached and (now - cached[1]) < _SENTIMENT_REFRESH_S:
            sentiment_score = cached[0]
            sentiment_task  = None
        else:
            sentiment_task = grok.score(symbol) if grok else None

        ob_task = ob_collector.fetch_imbalance(symbol) if ob_collector else None

        # Lance les deux en parallèle
        tasks = [t for t in (sentiment_task, ob_task) if t is not None]
        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []

        idx = 0
        if sentiment_task is not None:
            val = results[idx]
            sentiment_score = float(val) if not isinstance(val, Exception) else 0.0
            sentiment_cache[symbol] = (sentiment_score, now)
            idx += 1

        ob_imbalance = 0.0
        if ob_task is not None:
            val = results[idx]
            ob_imbalance = float(val) if not isinstance(val, Exception) else 0.0

        return sentiment_score, ob_imbalance

    async def _on_tick(self, tick: dict, strategy, risk_mgr, executor) -> None:
        """Traite un tick en temps réel (hook pour stratégies tick-level)."""
        signal = strategy.on_tick(tick)
        if signal and risk_mgr.validate_signal(signal):
            await executor.submit(signal)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_collector(self):
        if self.broker == "mt5":
            from data.collectors.mt5_collector import MT5Collector
            return MT5Collector(settings=self.settings)
        from data.collectors.ohlcv import OHLCVCollector
        return OHLCVCollector(settings=self.settings)

    def _build_strategy(self, symbol: str):
        if self.strategy_name == "ema_rsi":
            from trading.strategies.ema_rsi import EmaRsiStrategy
            return EmaRsiStrategy(settings=self.settings, symbol=symbol)
        if self.strategy_name == "ai":
            from trading.strategies.ai_strategy import AIStrategy
            return AIStrategy(settings=self.settings, symbol=symbol)
        raise ValueError(f"Stratégie inconnue : {self.strategy_name}")

    def _build_risk_manager(self):
        from trading.risk.manager import RiskManager
        return RiskManager(settings=self.settings)

    def _build_executor(self):
        if self.broker == "mt5":
            from trading.executor.mt5 import MT5Executor
            return MT5Executor(settings=self.settings)
        if self.broker == "alpaca":
            from trading.executor.alpaca import AlpacaExecutor
            return AlpacaExecutor(settings=self.settings)
        from trading.executor.binance import BinanceExecutor
        return BinanceExecutor(settings=self.settings)
