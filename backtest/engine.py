"""
backtest/engine.py — Moteur de backtest event-driven.

Slippage adaptatif (v2) :
  Le slippage fixe 0.05% est remplacé par un modèle en deux composantes :
  1. Volatilité  : ATR élevé → spread plus large (×1 à ×2)
  2. Impact marché : gros ordre par rapport au volume de la barre → déplacement de prix
                     < 0.5% du volume → ×1   |   0.5-2% → ×2   |   > 2% → ×3
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


# Frais de commission fixes (Binance spot maker/taker)
COMMISSION_PCT = 0.001   # 0.1% par trade
BASE_SLIPPAGE  = 0.0005  # 0.05% slippage de base


def adaptive_slippage(
    close:          float,
    atr:            float,
    order_usd:      float,
    bar_volume_usd: float,
    base:           float = BASE_SLIPPAGE,
) -> float:
    """
    Calcule le slippage adaptatif en pourcentage du prix.

    Composante volatilité :
      atr_pct = ATR / close  →  multiplicateur linéaire [1.0 … 2.0] pour ATR ∈ [0 … 5%]

    Composante impact de marché :
      impact = order_usd / bar_volume_usd
        < 0.5% du volume  → ×1.0 (ordre atomique, pas d'impact)
        0.5 – 2% du volume→ ×2.0 (ordre notable)
        > 2% du volume    → ×3.0 (ordre déplace le carnet)
    """
    # Volatilité
    atr_pct   = atr / max(close, 1e-8)
    vol_mult  = 1.0 + min(atr_pct / 0.05, 1.0)  # cap à 2.0× pour ATR ≥ 5%

    # Impact de marché
    impact_r = order_usd / max(bar_volume_usd, 1.0)
    if impact_r > 0.02:
        impact_mult = 3.0
    elif impact_r > 0.005:
        impact_mult = 2.0
    else:
        impact_mult = 1.0

    return base * vol_mult * impact_mult


class BacktestEngine:
    """Moteur de backtest event-driven avec frais et slippage adaptatif."""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings

    def run(
        self,
        strategy_name:   str,
        symbol:          str   = "BTC/USDT",
        start:           str   = "2024-01-01",
        end:             str   = "2024-12-31",
        timeframe:       str   = "1h",
        initial_capital: float = 10_000.0,
    ) -> BacktestMetrics:
        """Lance le backtest et retourne les métriques."""
        logger.info(f"BacktestEngine: {strategy_name}  {symbol}  {start} -> {end}  {timeframe}")

        df = self._load_data(symbol, timeframe, start, end)
        if df.empty:
            logger.error("Aucune donnée récupérée.")
            return self._empty_metrics()

        df = TechnicalIndicators.add_all(df)
        df.dropna(inplace=True)

        strategy        = self._build_strategy(strategy_name)
        capital         = initial_capital
        position        = 0.0
        entry_price     = 0.0
        entry_slip      = 0.0   # slippage mémorisé à l'entrée
        equity_values   = [capital]
        trades: list[float] = []

        for _, row in df.iterrows():
            features        = row.to_dict()
            close           = float(row["close"])
            atr             = float(row.get("atr_14", 0) or 0)
            bar_vol_usd     = float(row.get("volume", 0) or 0) * close

            signal = strategy.on_bar(features)

            if signal and signal.action == "BUY" and position == 0:
                order_usd    = capital * 0.95
                slip         = adaptive_slippage(close, atr, order_usd, bar_vol_usd)
                entry_slip   = slip
                entry_price  = close * (1 + slip)
                qty          = order_usd / entry_price
                cost         = qty * entry_price * (1 + COMMISSION_PCT)
                if cost <= capital:
                    capital  -= cost
                    position  = qty

            elif signal and signal.action == "SELL" and position > 0:
                order_usd   = position * close
                slip        = adaptive_slippage(close, atr, order_usd, bar_vol_usd)
                exit_price  = close * (1 - slip)
                proceeds    = position * exit_price * (1 - COMMISSION_PCT)
                pnl         = proceeds - (position * entry_price)
                capital    += proceeds
                trades.append(pnl)
                position    = 0.0

            # SL/TP check (utilise le slippage d'entrée pour la cohérence)
            if position > 0 and signal:
                if signal.stop_loss and row["low"] < signal.stop_loss:
                    order_usd  = position * signal.stop_loss
                    slip       = adaptive_slippage(signal.stop_loss, atr, order_usd, bar_vol_usd)
                    exit_price = signal.stop_loss * (1 - slip)
                    proceeds   = position * exit_price * (1 - COMMISSION_PCT)
                    pnl        = proceeds - (position * entry_price)
                    capital   += proceeds
                    trades.append(pnl)
                    position   = 0.0
                elif signal.take_profit and row["high"] > signal.take_profit:
                    order_usd  = position * signal.take_profit
                    slip       = adaptive_slippage(signal.take_profit, atr, order_usd, bar_vol_usd)
                    exit_price = signal.take_profit * (1 - slip)
                    proceeds   = position * exit_price * (1 - COMMISSION_PCT)
                    pnl        = proceeds - (position * entry_price)
                    capital   += proceeds
                    trades.append(pnl)
                    position   = 0.0

            equity_values.append(capital + position * close)

        # Fermeture de la position finale
        if position > 0:
            last = df["close"].iloc[-1]
            proceeds = position * last * (1 - COMMISSION_PCT)
            pnl      = proceeds - (position * entry_price)
            capital += proceeds
            trades.append(pnl)

        equity_curve = pd.Series(equity_values)
        metrics = compute_metrics(equity_curve, trades)
        logger.info(f"\n{metrics.summary()}")
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_data(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
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
