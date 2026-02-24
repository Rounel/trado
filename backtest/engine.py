"""
backtest/engine.py — Moteur de backtest (VectorBT + frais/slippage réalistes).

Workflow :
  1. Télécharge les données historiques (ccxt)
  2. Calcule les indicateurs techniques
  3. Applique la stratégie barre par barre
  4. Calcule les métriques de performance
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from analysis.technical.indicators import TechnicalIndicators
from backtest.metrics import BacktestMetrics, compute_metrics

if TYPE_CHECKING:
    from config.settings import Settings


class BacktestEngine:
    """Moteur de backtest event-driven avec frais et slippage."""

    # Paramètres réalistes pour Binance spot
    COMMISSION_PCT = 0.001   # 0.1% par trade
    SLIPPAGE_PCT   = 0.0005  # 0.05% glissement de prix

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings

    def run(
        self,
        strategy_name: str,
        symbol: str = "BTC/USDT",
        start: str = "2024-01-01",
        end: str = "2024-12-31",
        timeframe: str = "1h",
        initial_capital: float = 10_000.0,
    ) -> BacktestMetrics:
        """Lance le backtest et retourne les métriques."""
        logger.info(f"BacktestEngine: {strategy_name}  {symbol}  {start}→{end}  {timeframe}")

        # 1. Chargement des données
        df = self._load_data(symbol, timeframe, start, end)
        if df.empty:
            logger.error("Aucune donnée récupérée.")
            return self._empty_metrics()

        # 2. Indicateurs
        df = TechnicalIndicators.add_all(df)
        df.dropna(inplace=True)

        # 3. Stratégie
        strategy = self._build_strategy(strategy_name)

        # 4. Simulation barre par barre
        capital = initial_capital
        position = 0.0
        entry_price = 0.0
        equity_values = [capital]
        trades: list[float] = []

        for i, (_, row) in enumerate(df.iterrows()):
            features = row.to_dict()
            features["close"] = row["close"]

            signal = strategy.on_bar(features)

            if signal and signal.action == "BUY" and position == 0:
                # Entrée long
                entry_price = row["close"] * (1 + self.SLIPPAGE_PCT)
                qty = (capital * 0.95) / entry_price
                cost = qty * entry_price * (1 + self.COMMISSION_PCT)
                if cost <= capital:
                    capital -= cost
                    position = qty

            elif signal and signal.action == "SELL" and position > 0:
                # Sortie long
                exit_price = row["close"] * (1 - self.SLIPPAGE_PCT)
                proceeds = position * exit_price * (1 - self.COMMISSION_PCT)
                pnl = proceeds - (position * entry_price)
                capital += proceeds
                trades.append(pnl)
                position = 0.0

            # SL/TP check
            if position > 0 and signal:
                if signal.stop_loss and row["low"] < signal.stop_loss:
                    exit_price = signal.stop_loss * (1 - self.SLIPPAGE_PCT)
                    proceeds = position * exit_price * (1 - self.COMMISSION_PCT)
                    pnl = proceeds - (position * entry_price)
                    capital += proceeds
                    trades.append(pnl)
                    position = 0.0
                elif signal.take_profit and row["high"] > signal.take_profit:
                    exit_price = signal.take_profit * (1 - self.SLIPPAGE_PCT)
                    proceeds = position * exit_price * (1 - self.COMMISSION_PCT)
                    pnl = proceeds - (position * entry_price)
                    capital += proceeds
                    trades.append(pnl)
                    position = 0.0

            # Valeur totale
            total_value = capital + position * row["close"]
            equity_values.append(total_value)

        # Fermeture position finale
        if position > 0:
            last_price = df["close"].iloc[-1]
            proceeds = position * last_price * (1 - self.COMMISSION_PCT)
            pnl = proceeds - (position * entry_price)
            capital += proceeds
            trades.append(pnl)

        equity_curve = pd.Series(equity_values)
        metrics = compute_metrics(equity_curve, trades)
        logger.info(f"\n{metrics.summary()}")
        return metrics

    def _load_data(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        """Charge les données historiques via le collecteur OHLCV."""
        from data.collectors.ohlcv import OHLCVCollector
        collector = OHLCVCollector(settings=self.settings)
        try:
            df = asyncio.run(collector.fetch_binance(symbol, timeframe, limit=5000))
            if not df.empty:
                df = df.loc[start:end]
            return df
        except Exception as exc:
            logger.error(f"Erreur chargement données: {exc}")
            return pd.DataFrame()

    def _build_strategy(self, name: str):
        if name == "ema_rsi":
            from trading.strategies.ema_rsi import EmaRsiStrategy
            return EmaRsiStrategy(settings=self.settings)
        elif name == "ai":
            from trading.strategies.ai_strategy import AIStrategy
            return AIStrategy(settings=self.settings)
        raise ValueError(f"Stratégie inconnue : {name}")

    @staticmethod
    def _empty_metrics() -> BacktestMetrics:
        return BacktestMetrics(
            total_return_pct=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
            calmar_ratio=0.0, max_drawdown_pct=0.0, profit_factor=0.0,
            win_rate=0.0, total_trades=0, winning_trades=0, losing_trades=0,
            avg_win=0.0, avg_loss=0.0, best_trade=0.0, worst_trade=0.0,
        )
