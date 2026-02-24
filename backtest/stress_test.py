"""
backtest/stress_test.py — Stress Testing sur crises historiques.

Simule la stratégie sur des périodes de crise réelles pour évaluer
sa résilience face aux chocs de marché extrêmes.

Scénarios inclus :
  - COVID Crash (Fév-Mars 2020)
  - LUNA/UST Collapse (Mai 2022)
  - FTX Collapse (Novembre 2022)
  - Crypto Winter 2022 (Nov 2022 → Jan 2023)
  - Bear Crypto 2018
  - SVB / Récession US (Mars 2023)
  - Krach Flash BTC (Mai 2021)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from analysis.technical.indicators import TechnicalIndicators
from backtest.metrics import BacktestMetrics, compute_metrics

if TYPE_CHECKING:
    from config.settings import Settings


# ──────────────────────────────────────────────
# Scénarios de stress prédéfinis
# ──────────────────────────────────────────────

STRESS_SCENARIOS: dict[str, dict] = {
    "covid_crash": {
        "name":    "COVID-19 Crash",
        "start":   "2020-02-20",
        "end":     "2020-03-23",
        "symbol":  "BTC/USDT",
        "note":    "BTC -60% en 32 jours. Liquidations massives.",
    },
    "luna_collapse": {
        "name":    "LUNA/UST Collapse",
        "start":   "2022-05-05",
        "end":     "2022-05-20",
        "symbol":  "BTC/USDT",
        "note":    "Effondrement de l'écosystème Terra. BTC -40% en 2 semaines.",
    },
    "ftx_collapse": {
        "name":    "FTX Collapse",
        "start":   "2022-11-06",
        "end":     "2022-11-18",
        "symbol":  "BTC/USDT",
        "note":    "Faillite de FTX. BTC -25% en 12 jours.",
    },
    "crypto_winter_2022": {
        "name":    "Crypto Winter 2022",
        "start":   "2022-06-01",
        "end":     "2022-12-31",
        "symbol":  "BTC/USDT",
        "note":    "Marché baissier prolongé. BTC -75% depuis ATH.",
    },
    "crypto_bear_2018": {
        "name":    "Bear Market Crypto 2018",
        "start":   "2018-01-15",
        "end":     "2018-12-15",
        "symbol":  "BTC/USDT",
        "note":    "Marché baissier majeur post-bull 2017. BTC -84%.",
    },
    "svb_crisis": {
        "name":    "Crise SVB / Récession US",
        "start":   "2023-03-08",
        "end":     "2023-03-20",
        "symbol":  "BTC/USDT",
        "note":    "Faillite Silicon Valley Bank. BTC -20% en 12 jours.",
    },
    "btc_flash_crash_2021": {
        "name":    "Flash Crash BTC (Mai 2021)",
        "start":   "2021-05-12",
        "end":     "2021-05-25",
        "symbol":  "BTC/USDT",
        "note":    "Tweet d'Elon Musk + interdiction Chine. BTC -50% en 2 semaines.",
    },
}


# ──────────────────────────────────────────────
# Résultats
# ──────────────────────────────────────────────

@dataclass
class StressResult:
    scenario_id:  str
    scenario_name: str
    symbol:        str
    period:        str
    note:          str
    metrics:       BacktestMetrics
    data_available: bool = True

    def summary_line(self) -> str:
        m = self.metrics
        status = "✅" if m.total_return_pct > -10 else ("⚠️" if m.total_return_pct > -25 else "❌")
        return (
            f"{status} {self.scenario_name:<30} | "
            f"Return: {m.total_return_pct:+6.1f}%  "
            f"DD: {m.max_drawdown_pct:5.1f}%  "
            f"Sharpe: {m.sharpe_ratio:+.2f}  "
            f"Trades: {m.total_trades}"
        )


@dataclass
class StressTestReport:
    results:       list[StressResult] = field(default_factory=list)
    strategy_name: str = ""
    symbol:        str = ""

    def summary(self) -> str:
        lines = [
            "=" * 75,
            f"  TRADO — Stress Test Report  [{self.strategy_name} | {self.symbol}]",
            "=" * 75,
        ]
        for r in self.results:
            if r.data_available:
                lines.append(r.summary_line())
            else:
                lines.append(f"  ⚪ {r.scenario_name:<30} | Données non disponibles")

        surviving = [r for r in self.results if r.data_available and r.metrics.total_return_pct > -20]
        lines += [
            "=" * 75,
            f"  Résilience : {len(surviving)}/{len(self.results)} scénarios avec return > -20%",
            "=" * 75,
        ]
        return "\n".join(lines)

    def resilience_score(self) -> float:
        """Score de résilience [0, 1] = proportion de scénarios survivants."""
        available = [r for r in self.results if r.data_available]
        if not available:
            return 0.0
        surviving = [r for r in available if r.metrics.total_return_pct > -20]
        return len(surviving) / len(available)


# ──────────────────────────────────────────────
# Moteur de stress test
# ──────────────────────────────────────────────

class StressTester:
    """Exécute les stress tests sur des scénarios de crise historiques."""

    COMMISSION = 0.001
    SLIPPAGE   = 0.0005

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings

    def run(
        self,
        strategy_name: str = "ema_rsi",
        symbol: str = "BTC/USDT",
        scenarios: list[str] | None = None,
        initial_capital: float = 10_000.0,
    ) -> StressTestReport:
        """
        Lance les stress tests pour les scénarios demandés.

        Args:
            strategy_name : "ema_rsi" | "ai"
            symbol        : symbole à tester
            scenarios     : liste d'IDs de scénarios (None = tous)
            initial_capital : capital de départ

        Returns:
            StressTestReport
        """
        ids = scenarios or list(STRESS_SCENARIOS.keys())
        report = StressTestReport(strategy_name=strategy_name, symbol=symbol)

        for sid in ids:
            if sid not in STRESS_SCENARIOS:
                logger.warning(f"Scénario inconnu : {sid}")
                continue

            scenario = STRESS_SCENARIOS[sid]
            # Override symbol from scenario if set
            sym = scenario.get("symbol", symbol)
            logger.info(f"Stress test [{scenario['name']}] {sym} {scenario['start']}→{scenario['end']}")

            result = self._run_scenario(
                scenario_id=sid,
                scenario=scenario,
                symbol=sym,
                strategy_name=strategy_name,
                initial_capital=initial_capital,
            )
            report.results.append(result)

        logger.info(f"\n{report.summary()}")
        return report

    def _run_scenario(
        self,
        scenario_id: str,
        scenario: dict,
        symbol: str,
        strategy_name: str,
        initial_capital: float,
    ) -> StressResult:
        """Exécute un scénario unique."""
        df = self._load_data(symbol, scenario["start"], scenario["end"])

        if df.empty:
            return StressResult(
                scenario_id=scenario_id,
                scenario_name=scenario["name"],
                symbol=symbol,
                period=f"{scenario['start']} → {scenario['end']}",
                note=scenario.get("note", ""),
                metrics=self._empty_metrics(),
                data_available=False,
            )

        df = TechnicalIndicators.add_all(df)
        df.dropna(inplace=True)

        strategy = self._build_strategy(strategy_name)
        metrics  = self._simulate(df, strategy, initial_capital)

        return StressResult(
            scenario_id=scenario_id,
            scenario_name=scenario["name"],
            symbol=symbol,
            period=f"{scenario['start']} → {scenario['end']}",
            note=scenario.get("note", ""),
            metrics=metrics,
            data_available=True,
        )

    def _simulate(self, df: pd.DataFrame, strategy, capital: float) -> BacktestMetrics:
        """Simulation barre par barre avec frais et slippage."""
        position    = 0.0
        entry_price = 0.0
        equity      = [capital]
        trades: list[float] = []

        for _, row in df.iterrows():
            features = row.to_dict()
            signal   = strategy.on_bar(features)

            if signal and signal.action == "BUY" and position == 0:
                ep  = row["close"] * (1 + self.SLIPPAGE)
                qty = (capital * 0.95) / ep
                if qty * ep * (1 + self.COMMISSION) <= capital:
                    capital    -= qty * ep * (1 + self.COMMISSION)
                    position    = qty
                    entry_price = ep

            elif signal and signal.action == "SELL" and position > 0:
                xp       = row["close"] * (1 - self.SLIPPAGE)
                proceeds = position * xp * (1 - self.COMMISSION)
                trades.append(proceeds - position * entry_price)
                capital += proceeds
                position = 0.0

            equity.append(capital + position * row["close"])

        if position > 0:
            xp       = df["close"].iloc[-1]
            proceeds = position * xp * (1 - self.COMMISSION)
            trades.append(proceeds - position * entry_price)
            capital += proceeds

        return compute_metrics(pd.Series(equity), trades)

    def _load_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        from data.collectors.ohlcv import OHLCVCollector
        collector = OHLCVCollector(settings=self.settings)
        try:
            df = asyncio.run(collector.fetch_binance(symbol, "1h", limit=10_000))
            if not df.empty:
                df = df.loc[start:end]
            return df
        except Exception as exc:
            logger.error(f"StressTester: données non disponibles — {exc}")
            return pd.DataFrame()

    def _build_strategy(self, name: str):
        from config.settings import Settings
        if name == "ema_rsi":
            from trading.strategies.ema_rsi import EmaRsiStrategy
            return EmaRsiStrategy(settings=self.settings)
        from trading.strategies.ai_strategy import AIStrategy
        return AIStrategy(settings=self.settings)

    @staticmethod
    def _empty_metrics() -> BacktestMetrics:
        return BacktestMetrics(
            total_return_pct=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown_pct=0, profit_factor=0, win_rate=0, total_trades=0,
            winning_trades=0, losing_trades=0, avg_win=0, avg_loss=0,
            best_trade=0, worst_trade=0,
        )
