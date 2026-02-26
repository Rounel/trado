"""
core/engine.py — Orchestrateur async principal de TRADO.

Boucle de vie :
  start() → une tâche asyncio PAR symbole, en parallèle
           → collecte → features [+ sentiment + OB si ai] → signal → risk → executor
  stop()  → annulation propre de toutes les tâches

Journalisation (v2) :
  Chaque trade fermé et chaque snapshot de portefeuille sont écrits dans
  data/cache/trado_journal.db via data.pipeline.trade_journal.
  Le dashboard Streamlit lit ces données en temps réel.
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings

_SENTIMENT_REFRESH_S  = 900   # 15 minutes entre deux appels Grok par symbole
_BALANCE_SYNC_BARS    = 10    # resynchronisation du solde toutes les N bougies

_DEFAULT_SYMBOLS: dict[str, list[str]] = {
    "binance": ["BTC/USDT"],
    "alpaca":  ["AAPL"],
    "mt5":     ["EURUSD"],
}


class TradoEngine:
    """Orchestrateur central — supporte plusieurs symboles en parallèle."""

    def __init__(
        self,
        settings:      "Settings",
        strategy_name: str = "ema_rsi",
        broker:        str = "binance",
        symbols:       list[str] | None = None,
        timeframe:     str = "1h",
    ) -> None:
        self.settings      = settings
        self.strategy_name = strategy_name
        self.broker        = broker
        self.symbols       = symbols or _DEFAULT_SYMBOLS.get(broker, ["BTC/USDT"])
        self.timeframe     = timeframe
        self._running      = False
        self._tasks:       list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Démarre une boucle de trading par symbole, toutes en parallèle."""
        from monitoring.logger import setup_logger
        from data.pipeline.trade_journal import init_async as journal_init

        setup_logger(self.settings)

        # Initialise le journal SQLite (crée les tables si absentes)
        await journal_init()

        logger.info(
            f"TradoEngine starting — strategy={self.strategy_name} broker={self.broker} "
            f"env={self.settings.env} symbols={self.symbols} timeframe={self.timeframe}"
        )

        self._running = True
        executor      = self._build_executor()

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
        from data.pipeline.trade_journal import load_open_positions

        collector = self._build_collector()
        strategy  = self._build_strategy(symbol)
        risk_mgr  = self._build_risk_manager()

        # ── Restauration des positions persistées ─────────────────────
        saved_positions = await load_open_positions(symbol=symbol)
        if saved_positions:
            risk_mgr.restore_positions(saved_positions)

        # ── Synchronisation initiale du solde réel ────────────────────
        await self._sync_balance(executor, [risk_mgr])

        grok         = None
        ob_collector = None
        if self.strategy_name == "ai":
            from analysis.sentiment.grok_analyzer import GrokAnalyzer
            from data.collectors.orderbook import OrderBookCollector
            grok         = GrokAnalyzer(self.settings)
            ob_collector = OrderBookCollector(self.settings)

        sentiment_cache: dict[str, tuple[float, float]] = {}
        bar_count = 0

        logger.info(f"[{symbol}] boucle démarrée ({self.timeframe})")
        try:
            async for bar in collector.stream(
                symbol=symbol,
                timeframe=self.timeframe,
                warmup_bars=strategy.warmup_bars,
            ):
                bar_count += 1
                if bar_count % _BALANCE_SYNC_BARS == 0:
                    await self._sync_balance(executor, [risk_mgr])
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
    # Traitement barre par barre
    # ------------------------------------------------------------------

    async def _on_bar(
        self,
        bar:            dict,
        strategy,
        risk_mgr,
        executor,
        grok=None,
        ob_collector=None,
        sentiment_cache: dict | None = None,
    ) -> None:
        """Traite une nouvelle bougie OHLCV."""
        from data.pipeline.features import FeatureBuilder
        from data.pipeline.trade_journal import (
            record_closed_trade, record_snapshot,
            save_open_position, delete_open_position,
        )
        from trading.strategies.base import Signal as TradoSignal

        features    = FeatureBuilder.from_bar(bar)
        symbol      = bar.get("symbol", "")
        close_price = float(bar.get("close", 0) or 0)

        # Enrichissement async (sentiment + order book en parallèle)
        if grok is not None or ob_collector is not None:
            sentiment_score, ob_imbalance = await self._fetch_enrichments(
                symbol, grok, ob_collector, sentiment_cache or {}
            )
            features["sentiment_score"] = sentiment_score
            features["ob_imbalance"]    = ob_imbalance

        # ── 1. Trailing stops ──────────────────────────────────────────
        triggered = risk_mgr.update_trailing_stops({symbol: close_price})
        for sym in triggered:
            logger.info(f"[{sym}] Trailing stop -> fermeture position au marche")
            pos = risk_mgr.get_position(sym)   # récupère avant la fermeture
            close_sig = TradoSignal(action="SELL", symbol=sym, confidence=1.0, strategy="trailing_stop")
            try:
                order      = await executor.submit(close_sig)
                exit_price = float((order or {}).get("price", close_price) or close_price)
            except Exception as exc:
                logger.error(f"[{sym}] Erreur fermeture trailing stop : {exc}")
                exit_price = close_price
            pnl = risk_mgr.record_close(sym, exit_price)
            await delete_open_position(sym)
            if pos:
                await record_closed_trade(
                    symbol=sym, side=pos["side"],
                    entry=pos["entry"], exit_price=exit_price,
                    size=pos["size"], pnl=pnl, strategy="trailing_stop",
                )

        # ── 2. Renforcement Livermore ──────────────────────────────────
        reinforce = risk_mgr.check_livermore_add(symbol, close_price)
        if reinforce is not None:
            try:
                order       = await executor.submit(reinforce)
                entry_price = float((order or {}).get("price", close_price) or close_price)
            except Exception as exc:
                logger.error(f"[{symbol}] Erreur renforcement Livermore : {exc}")
                entry_price = close_price
            if reinforce.size:
                risk_mgr.record_open(
                    symbol, reinforce.action, reinforce.size,
                    entry_price, reinforce.stop_loss,
                )
                await save_open_position(
                    symbol=symbol, side=reinforce.action,
                    size=reinforce.size, entry=entry_price,
                    stop_loss=reinforce.stop_loss,
                    trailing_stop=risk_mgr.get_trailing_stop(symbol),
                    trailing_activation=risk_mgr.get_trailing_activation(symbol),
                )

        # ── 3. Signal stratégie normal ─────────────────────────────────
        signal = strategy.on_bar(features)
        if signal is None:
            # Snapshot portfolio même sans trade
            await record_snapshot(
                risk_mgr._portfolio_value,
                len(risk_mgr._open_positions),
            )
            return

        signal.entry_price = close_price

        if not risk_mgr.validate_signal(signal, price=close_price):
            logger.debug(f"Signal refuse par RiskManager : {signal}")
            await record_snapshot(risk_mgr._portfolio_value, len(risk_mgr._open_positions))
            return

        # ── 4. Sizing ─────────────────────────────────────────────────
        atr = float(signal.metadata.get("atr", 0) or 0)
        risk_mgr.size_signal(signal, risk_mgr._portfolio_value, close_price, atr)

        # ── 5. Entrée Livermore (partielle) ───────────────────────────
        risk_mgr.open_livermore_signal(signal)

        logger.info(f"Execution du signal : {signal}")
        try:
            order       = await executor.submit(signal)
            entry_price = float((order or {}).get("price", close_price) or close_price)
        except Exception as exc:
            logger.error(f"[{symbol}] Erreur execution ordre : {exc}")
            await record_snapshot(risk_mgr._portfolio_value, len(risk_mgr._open_positions))
            return

        # ── 6. Enregistrement position + snapshot ─────────────────────
        if signal.size:
            risk_mgr.record_open(
                signal.symbol, signal.action, signal.size,
                entry_price, signal.stop_loss,
            )
            await save_open_position(
                symbol=signal.symbol, side=signal.action,
                size=signal.size, entry=entry_price,
                stop_loss=signal.stop_loss,
                trailing_stop=risk_mgr.get_trailing_stop(signal.symbol),
                trailing_activation=risk_mgr.get_trailing_activation(signal.symbol),
            )

        await record_snapshot(
            risk_mgr._portfolio_value,
            len(risk_mgr._open_positions),
        )

    # ------------------------------------------------------------------
    # Enrichissement async
    # ------------------------------------------------------------------

    async def _fetch_enrichments(
        self,
        symbol:          str,
        grok,
        ob_collector,
        sentiment_cache: dict,
    ) -> tuple[float, float]:
        """Récupère sentiment et imbalance OB en parallèle. Cache le sentiment 15 min."""
        now = time.monotonic()

        cached = sentiment_cache.get(symbol)
        if cached and (now - cached[1]) < _SENTIMENT_REFRESH_S:
            sentiment_score = cached[0]
            sentiment_task  = None
        else:
            sentiment_task = grok.score(symbol) if grok else None

        ob_task = ob_collector.fetch_imbalance(symbol) if ob_collector else None

        tasks   = [t for t in (sentiment_task, ob_task) if t is not None]
        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []

        idx = 0
        if sentiment_task is not None:
            val             = results[idx]
            sentiment_score = float(val) if not isinstance(val, Exception) else 0.0
            sentiment_cache[symbol] = (sentiment_score, now)
            idx += 1

        ob_imbalance = 0.0
        if ob_task is not None:
            val          = results[idx]
            ob_imbalance = float(val) if not isinstance(val, Exception) else 0.0

        return sentiment_score, ob_imbalance

    async def _sync_balance(self, executor, risk_managers: list) -> None:
        """Récupère le solde USDT réel depuis le broker et le propage aux RiskManagers."""
        try:
            balance = await executor.get_balance()
            usdt = float(balance.get("USDT", 0) or 0)
            if usdt > 0:
                for rm in risk_managers:
                    rm.sync_balance(usdt)
        except Exception as exc:
            logger.warning(f"_sync_balance: impossible de récupérer le solde — {exc}")

    async def _on_tick(self, tick: dict, strategy, risk_mgr, executor) -> None:
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
        raise ValueError(f"Strategie inconnue : {self.strategy_name}")

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
