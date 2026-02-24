"""
backtest/metrics.py — Métriques de performance : Sharpe, Sortino, Calmar, Profit Factor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestMetrics:
    """Ensemble des métriques d'un backtest."""
    total_return_pct:  float
    sharpe_ratio:      float
    sortino_ratio:     float
    calmar_ratio:      float
    max_drawdown_pct:  float
    profit_factor:     float
    win_rate:          float
    total_trades:      int
    winning_trades:    int
    losing_trades:     int
    avg_win:           float
    avg_loss:          float
    best_trade:        float
    worst_trade:       float

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "  TRADO — Backtest Performance Report",
            "=" * 50,
            f"  Total Return    : {self.total_return_pct:+.2f}%",
            f"  Sharpe Ratio    : {self.sharpe_ratio:.3f}",
            f"  Sortino Ratio   : {self.sortino_ratio:.3f}",
            f"  Calmar Ratio    : {self.calmar_ratio:.3f}",
            f"  Max Drawdown    : {self.max_drawdown_pct:.2f}%",
            f"  Profit Factor   : {self.profit_factor:.2f}",
            f"  Win Rate        : {self.win_rate*100:.1f}%",
            f"  Total Trades    : {self.total_trades}",
            f"  Winning / Losing: {self.winning_trades} / {self.losing_trades}",
            f"  Avg Win / Loss  : {self.avg_win:.2f}$ / {self.avg_loss:.2f}$",
            f"  Best  Trade     : {self.best_trade:+.2f}$",
            f"  Worst Trade     : {self.worst_trade:+.2f}$",
            "=" * 50,
        ]
        return "\n".join(lines)


def compute_metrics(
    equity_curve: pd.Series,
    trades: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760,  # heures dans une année
) -> BacktestMetrics:
    """
    Calcule toutes les métriques depuis la courbe d'équité et la liste des PnL de trades.

    Args:
        equity_curve     : Serie des valeurs de portefeuille
        trades           : Liste des PnL par trade (en USD)
        risk_free_rate   : Taux sans risque annuel
        periods_per_year : Nombre de périodes par an (8760 = hourly, 252 = daily)
    """
    returns = equity_curve.pct_change().dropna()
    initial = equity_curve.iloc[0]
    final   = equity_curve.iloc[-1]

    total_return_pct = (final - initial) / initial * 100

    # Sharpe
    excess = returns - risk_free_rate / periods_per_year
    sharpe = float(excess.mean() / (excess.std() + 1e-8) * np.sqrt(periods_per_year))

    # Sortino
    downside = returns[returns < 0]
    sortino = float(
        excess.mean() / (downside.std() + 1e-8) * np.sqrt(periods_per_year)
    )

    # Max Drawdown
    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max * 100
    max_dd = float(drawdowns.min())

    # Calmar
    calmar = float(total_return_pct / (abs(max_dd) + 1e-8))

    # Trade stats
    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    profit_factor = (sum(wins) / (abs(sum(losses)) + 1e-8)) if losses else float("inf")
    win_rate = len(wins) / (len(trades) + 1e-8) if trades else 0.0

    return BacktestMetrics(
        total_return_pct=total_return_pct,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown_pct=max_dd,
        profit_factor=profit_factor,
        win_rate=win_rate,
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        avg_win=float(np.mean(wins)) if wins else 0.0,
        avg_loss=float(np.mean(losses)) if losses else 0.0,
        best_trade=max(trades) if trades else 0.0,
        worst_trade=min(trades) if trades else 0.0,
    )
