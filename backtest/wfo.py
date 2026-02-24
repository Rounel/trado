"""
backtest/wfo.py — Walk-Forward Optimization (WFO).

Divise l'historique en fenêtres glissantes train/test
et agrège les métriques de performance sur tous les plis.

Principe :
  |--train--|--test--|
               |--train--|--test--|
                            |--train--|--test--|
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger

from backtest.metrics import BacktestMetrics, compute_metrics


@dataclass
class WFOResult:
    """Résultat agrégé de l'optimisation Walk-Forward."""
    folds:             int
    avg_sharpe:        float
    avg_sortino:       float
    avg_max_drawdown:  float
    avg_total_return:  float
    avg_win_rate:      float
    robustness_score:  float  # % de plis avec Sharpe > 0
    fold_metrics:      list[BacktestMetrics] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  TRADO — Walk-Forward Optimization Results",
            "=" * 55,
            f"  Folds analysés   : {self.folds}",
            f"  Sharpe moyen     : {self.avg_sharpe:.3f}",
            f"  Sortino moyen    : {self.avg_sortino:.3f}",
            f"  Max DD moyen     : {self.avg_max_drawdown:.2f}%",
            f"  Return moyen     : {self.avg_total_return:+.2f}%",
            f"  Win rate moyen   : {self.avg_win_rate*100:.1f}%",
            f"  Robustesse       : {self.robustness_score*100:.1f}% des plis Sharpe > 0",
            "=" * 55,
        ]
        return "\n".join(lines)


class WalkForwardOptimizer:
    """
    Optimisation Walk-Forward pour valider la robustesse d'une stratégie.

    Args:
        train_size : nombre de barres pour l'entraînement
        test_size  : nombre de barres pour le test
        step_size  : avancement entre chaque pli (défaut = test_size)
    """

    def __init__(
        self,
        train_size: int = 2000,
        test_size:  int = 500,
        step_size:  int | None = None,
    ) -> None:
        self.train_size = train_size
        self.test_size  = test_size
        self.step_size  = step_size or test_size

    def run(
        self,
        df: pd.DataFrame,
        strategy_fn,
        initial_capital: float = 10_000.0,
    ) -> WFOResult:
        """
        Lance l'optimisation sur le DataFrame.

        Args:
            df          : DataFrame OHLCV + indicateurs techniques
            strategy_fn : callable(train_df) → strategy — permet d'entraîner sur chaque pli
            initial_capital : capital de départ

        Returns:
            WFOResult
        """
        n = len(df)
        fold_metrics: list[BacktestMetrics] = []
        fold = 0

        start = 0
        while start + self.train_size + self.test_size <= n:
            train_end = start + self.train_size
            test_end  = train_end + self.test_size

            train_df = df.iloc[start:train_end]
            test_df  = df.iloc[train_end:test_end]

            fold += 1
            logger.info(f"WFO pli {fold}: train={len(train_df)} barres, test={len(test_df)} barres")

            # Entraînement + backtest sur le pli test
            strategy = strategy_fn(train_df)
            metrics  = self._run_fold(test_df, strategy, initial_capital)
            fold_metrics.append(metrics)

            start += self.step_size

        if not fold_metrics:
            logger.warning("WFO: pas assez de données pour au moins un pli")
            return WFOResult(folds=0, avg_sharpe=0, avg_sortino=0,
                             avg_max_drawdown=0, avg_total_return=0,
                             avg_win_rate=0, robustness_score=0)

        return WFOResult(
            folds=len(fold_metrics),
            avg_sharpe=float(np.mean([m.sharpe_ratio for m in fold_metrics])),
            avg_sortino=float(np.mean([m.sortino_ratio for m in fold_metrics])),
            avg_max_drawdown=float(np.mean([m.max_drawdown_pct for m in fold_metrics])),
            avg_total_return=float(np.mean([m.total_return_pct for m in fold_metrics])),
            avg_win_rate=float(np.mean([m.win_rate for m in fold_metrics])),
            robustness_score=float(np.mean([m.sharpe_ratio > 0 for m in fold_metrics])),
            fold_metrics=fold_metrics,
        )

    def _run_fold(
        self,
        test_df: pd.DataFrame,
        strategy,
        initial_capital: float,
    ) -> BacktestMetrics:
        """Simule la stratégie sur le pli de test."""
        capital = initial_capital
        position = 0.0
        entry_price = 0.0
        equity_values = [capital]
        trades: list[float] = []

        for _, row in test_df.iterrows():
            features = row.to_dict()
            signal = strategy.on_bar(features)

            if signal and signal.action == "BUY" and position == 0:
                qty = (capital * 0.95) / row["close"]
                cost = qty * row["close"] * 1.001
                if cost <= capital:
                    capital -= cost
                    position = qty
                    entry_price = row["close"]

            elif signal and signal.action == "SELL" and position > 0:
                proceeds = position * row["close"] * 0.999
                trades.append(proceeds - position * entry_price)
                capital += proceeds
                position = 0.0

            equity_values.append(capital + position * row["close"])

        return compute_metrics(pd.Series(equity_values), trades)
