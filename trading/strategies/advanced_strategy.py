"""
trading/strategies/advanced_strategy.py — AdvancedStrategy : Price Action + SMC + OTE + IFC.
Implémente l'interface BaseStrategy (on_bar / warmup_bars) pour s'intégrer dans TradoEngine.

Pipeline de génération de signal :
  1. MarketStructure  → trend, BOS/CHOCH
  2. OrderBlocks      → zones institutionnelles actives
  3. FVGDetector      → imbalances non comblées
  4. OTEEvaluator     → zone d'entrée optimale (61,8 %–78,6 %)
  5. CandlePatterns   → confirmation IFC (Pin Bar, Engulfing, …)
  6. LiquidityZones   → zones de concentration de stops
  7. StopHuntDetector → manipulation institutionnelle récente
  8. MetaLabelingModel→ filtre PoP ≥ pop_threshold
  → Signal avec SL/TP structurels

Hérite de la structure Signal/BaseStrategy définie dans trading/strategies/base.py.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from loguru import logger

from analysis.technical.ifc import CandlePattern, CandlePatternDetector
from analysis.technical.ote import OTEEvaluator, OTESignal
from analysis.technical.price_action import MarketStructure, StructureAnalysis, Trend
from analysis.technical.smc import (
    FVGDetector,
    LiquidityZoneDetector,
    OrderBlock,
    OrderBlockDetector,
    StopHuntDetector,
)
from models.meta_labeling import MetaFeatures, MetaLabelingModel
from trading.strategies.base import BaseStrategy
from trading.strategies.base import Signal as TradoSignal

if TYPE_CHECKING:
    from config.settings import Settings


# ---------------------------------------------------------------------------
# Signal dataclass (compatible avec trading/strategies/base.py)
# ---------------------------------------------------------------------------

@dataclass
class AdvancedSignal:
    """Signal enrichi avec les données structurelles SMC/OTE."""
    action: Literal["BUY", "SELL", "HOLD"]
    symbol: str
    confidence: float           # [0, 1]
    stop_loss: float
    take_profit: float
    entry_price: float
    size: float = 0.0           # calculé par PositionSizer en aval
    strategy: str = "advanced"

    # Contexte
    trend: Trend = Trend.RANGING
    ote_signal: OTESignal | None = None
    order_block: OrderBlock | None = None
    ifc_pattern: CandlePattern | None = None
    pop_score: float = 0.5
    structure_events: list = field(default_factory=list)

    @property
    def risk_reward(self) -> float:
        """R:R ratio du trade."""
        risk   = abs(self.entry_price - self.stop_loss) + 1e-8
        reward = abs(self.take_profit  - self.entry_price)
        return reward / risk


# ---------------------------------------------------------------------------
# AdvancedStrategy
# ---------------------------------------------------------------------------

class AdvancedStrategy(BaseStrategy):
    """
    Stratégie avancée combinant Price Action, SMC, OTE et IFC.

    Implémente BaseStrategy pour s'intégrer directement dans TradoEngine via on_bar().

    Args:
        settings         : configuration globale TRADO.
        symbol           : symbole traité (ex. "BTC/USDT").
        pop_threshold    : seuil de probabilité de profit (défaut 0.55).
        min_rr           : ratio R:R minimum pour valider un trade (défaut 1.5).
        swing_order      : ordre pour la détection des swing points (défaut 5).
        atr_sl_mult      : multiplicateur ATR pour le SL de repli (défaut 1.5).
        atr_tp_mult      : multiplicateur ATR pour le TP de repli (défaut 3.0).
    """

    WARMUP_BARS  = 60
    MAX_HISTORY  = 300

    def __init__(
        self,
        settings: "Settings",
        symbol: str = "BTC/USDT",
        pop_threshold: float = 0.55,
        min_rr: float = 1.5,
        swing_order: int = 5,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 3.0,
    ) -> None:
        super().__init__(settings)
        self.symbol       = symbol
        self.pop_threshold = pop_threshold
        self.min_rr       = min_rr
        self.atr_sl_mult  = atr_sl_mult
        self.atr_tp_mult  = atr_tp_mult

        # Buffer OHLCV interne (comme EmaRsiStrategy)
        self._buffer: deque[dict] = deque(maxlen=self.MAX_HISTORY)

        # Sous-modules
        self._structure  = MarketStructure()
        self._ob         = OrderBlockDetector()
        self._fvg        = FVGDetector()
        self._ote        = OTEEvaluator()
        self._ifc        = CandlePatternDetector()
        self._liquidity  = LiquidityZoneDetector()
        self._stop_hunt  = StopHuntDetector()
        self._meta       = MetaLabelingModel(pop_threshold=pop_threshold)

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "AdvancedStrategy"

    @property
    def warmup_bars(self) -> int:
        return self.WARMUP_BARS

    def on_bar(self, features: dict) -> TradoSignal | None:
        """
        Adaptateur BaseStrategy : reçoit un bar dict, maintient le buffer OHLCV,
        et appelle generate_signal() quand le buffer est suffisamment rempli.

        Le dict `features` doit contenir open, high, low, close, volume.
        Les clés optionnelles sentiment_score et ob_imbalance sont transmises.
        """
        self._buffer.append({
            "open":   float(features.get("open",   features.get("close", 0))),
            "high":   float(features.get("high",   features.get("close", 0))),
            "low":    float(features.get("low",    features.get("close", 0))),
            "close":  float(features.get("close",  0)),
            "volume": float(features.get("volume", 1)),
        })

        if len(self._buffer) < self.WARMUP_BARS:
            return None

        df = pd.DataFrame(list(self._buffer))

        adv = self.generate_signal(
            df=df,
            symbol=self.symbol,
            sentiment_score=float(features.get("sentiment_score", 0.0)),
            orderbook_imbalance=float(features.get("ob_imbalance", 0.0)),
        )

        if adv.action == "HOLD":
            return None

        return TradoSignal(
            action=adv.action,
            symbol=adv.symbol,
            confidence=adv.confidence,
            stop_loss=adv.stop_loss,
            take_profit=adv.take_profit,
            strategy=self.name,
            metadata={
                "pop":   adv.pop_score,
                "rr":    adv.risk_reward,
                "trend": adv.trend.value,
                "atr":   self._compute_atr(df),
            },
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        sentiment_score: float = 0.0,
        orderbook_imbalance: float = 0.0,
    ) -> AdvancedSignal:
        """
        Génère un signal de trading enrichi.

        Args:
            df                   : DataFrame OHLCV (colonnes : open, high, low, close, volume).
            symbol               : ex. "BTC/USDT".
            sentiment_score      : score Grok [-1, 1].
            orderbook_imbalance  : (bid_vol - ask_vol) / total [-1, 1].

        Returns:
            AdvancedSignal (action=HOLD si aucun setup valide).
        """
        if len(df) < 30:
            logger.warning(f"[AdvancedStrategy] DataFrame trop court ({len(df)} bougies).")
            return self._hold(symbol, float(df["close"].iloc[-1]))

        current_price = float(df["close"].iloc[-1])
        atr = self._compute_atr(df)

        # --- Étape 1 : Structure de marché ---
        structure = self._structure.analyze(df)

        # --- Étape 2 : Order Blocks actifs ---
        blocks = self._ob.detect(df)

        # --- Étape 3 : Fair Value Gaps ---
        fvgs = self._fvg.detect(df)

        # --- Étape 4 : OTE ---
        ote_signals = self._ote.find_ote_setups(
            structure.swing_highs,
            structure.swing_lows,
            current_price,
        )

        # --- Étape 5 : Patterns IFC ---
        confirmed_bull, bull_pattern = self._ifc.has_confirmation(df, "bullish")
        confirmed_bear, bear_pattern = self._ifc.has_confirmation(df, "bearish")

        # --- Étape 6 : Liquidity zones ---
        liq_zones = self._liquidity.detect(structure.swing_highs, structure.swing_lows)

        # --- Étape 7 : Stop hunt ---
        stop_hunts = self._stop_hunt.detect(df)
        recent_hunt = stop_hunts[-1] if stop_hunts else None

        # --- Étape 8 : Composer le meilleur setup ---
        signal = self._compose_signal(
            df=df,
            symbol=symbol,
            current_price=current_price,
            atr=atr,
            structure=structure,
            blocks=blocks,
            fvgs=fvgs,
            ote_signals=ote_signals,
            confirmed_bull=confirmed_bull,
            bull_pattern=bull_pattern,
            confirmed_bear=confirmed_bear,
            bear_pattern=bear_pattern,
            liq_zones=liq_zones,
            sentiment_score=sentiment_score,
            orderbook_imbalance=orderbook_imbalance,
        )

        return signal

    # ------------------------------------------------------------------
    # Signal composition
    # ------------------------------------------------------------------

    def _compose_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
        atr: float,
        structure: StructureAnalysis,
        blocks: list,
        fvgs: list,
        ote_signals: list[OTESignal],
        confirmed_bull: bool,
        bull_pattern: CandlePattern | None,
        confirmed_bear: bool,
        bear_pattern: CandlePattern | None,
        liq_zones: list,
        sentiment_score: float,
        orderbook_imbalance: float,
    ) -> AdvancedSignal:

        volume_ratio = self._volume_ratio(df)
        atr_norm     = atr / (current_price + 1e-8)
        fvg_present  = float(bool(fvgs))
        liq_count    = float(liq_zones[0].count) if liq_zones else 0.0

        # ----- Tentative LONG -----
        long_ote  = next((s for s in ote_signals if s.direction == "long"), None)
        bull_ob   = self._ob.nearest(blocks, current_price, "bullish")
        trend_ok_long = structure.trend in (Trend.BULLISH, Trend.RANGING)

        if long_ote and confirmed_bull and trend_ok_long:
            features = MetaFeatures(
                ote_strength=long_ote.zone_strength,
                ob_strength=bull_ob.strength if bull_ob else 0.0,
                pattern_strength=bull_pattern.strength if bull_pattern else 0.0,
                volume_ratio=volume_ratio,
                atr_normalized=atr_norm,
                trend_aligned=1.0 if structure.trend == Trend.BULLISH else 0.5,
                fvg_present=fvg_present,
                liquidity_count=liq_count,
                orderbook_imbalance=orderbook_imbalance,
                sentiment_score=sentiment_score,
            )
            valid, pop = self._meta.is_valid(features)

            if valid:
                sl, tp = self._structural_sl_tp(
                    current_price, "BUY", atr,
                    structure, long_ote, bull_ob,
                )
                sig = AdvancedSignal(
                    action="BUY",
                    symbol=symbol,
                    confidence=pop,
                    stop_loss=sl,
                    take_profit=tp,
                    entry_price=current_price,
                    trend=structure.trend,
                    ote_signal=long_ote,
                    order_block=bull_ob,
                    ifc_pattern=bull_pattern,
                    pop_score=pop,
                    structure_events=structure.recent_events,
                )
                if sig.risk_reward >= self.min_rr:
                    logger.info(
                        f"[AdvancedStrategy] BUY {symbol} @ {current_price:.4f} "
                        f"| SL={sl:.4f} TP={tp:.4f} R:R={sig.risk_reward:.2f} PoP={pop:.2f}"
                    )
                    return sig

        # ----- Tentative SHORT -----
        short_ote = next((s for s in ote_signals if s.direction == "short"), None)
        bear_ob   = self._ob.nearest(blocks, current_price, "bearish")
        trend_ok_short = structure.trend in (Trend.BEARISH, Trend.RANGING)

        if short_ote and confirmed_bear and trend_ok_short:
            features = MetaFeatures(
                ote_strength=short_ote.zone_strength,
                ob_strength=bear_ob.strength if bear_ob else 0.0,
                pattern_strength=bear_pattern.strength if bear_pattern else 0.0,
                volume_ratio=volume_ratio,
                atr_normalized=atr_norm,
                trend_aligned=1.0 if structure.trend == Trend.BEARISH else 0.5,
                fvg_present=fvg_present,
                liquidity_count=liq_count,
                orderbook_imbalance=-orderbook_imbalance,
                sentiment_score=-sentiment_score,
            )
            valid, pop = self._meta.is_valid(features)

            if valid:
                sl, tp = self._structural_sl_tp(
                    current_price, "SELL", atr,
                    structure, short_ote, bear_ob,
                )
                sig = AdvancedSignal(
                    action="SELL",
                    symbol=symbol,
                    confidence=pop,
                    stop_loss=sl,
                    take_profit=tp,
                    entry_price=current_price,
                    trend=structure.trend,
                    ote_signal=short_ote,
                    order_block=bear_ob,
                    ifc_pattern=bear_pattern,
                    pop_score=pop,
                    structure_events=structure.recent_events,
                )
                if sig.risk_reward >= self.min_rr:
                    logger.info(
                        f"[AdvancedStrategy] SELL {symbol} @ {current_price:.4f} "
                        f"| SL={sl:.4f} TP={tp:.4f} R:R={sig.risk_reward:.2f} PoP={pop:.2f}"
                    )
                    return sig

        return self._hold(symbol, current_price)

    # ------------------------------------------------------------------
    # SL / TP structurels
    # ------------------------------------------------------------------

    def _structural_sl_tp(
        self,
        price: float,
        action: Literal["BUY", "SELL"],
        atr: float,
        structure: StructureAnalysis,
        ote: OTESignal | None,
        ob: OrderBlock | None,
    ) -> tuple[float, float]:
        """
        Calcule SL et TP à partir des niveaux structurels.

        Priorité :
          1. SL sous/au-dessus de l'Order Block (le plus précis).
          2. SL sous/au-dessus du Swing Low/High récent.
          3. Repli ATR.
        """
        if action == "BUY":
            # SL : sous l'OB ou sous le dernier Higher Low
            if ob:
                sl = ob.bottom * (1 - 0.001)
            elif structure.last_higher_low:
                sl = structure.last_higher_low * (1 - 0.001)
            else:
                sl = price - self.atr_sl_mult * atr

            # TP : prochain swing high ou OTE 0% (retour au sommet)
            if ote:
                tp = ote.grid.swing_b * (1 + 0.001)
            elif structure.last_higher_high:
                tp = structure.last_higher_high * (1 + 0.001)
            else:
                tp = price + self.atr_tp_mult * atr

        else:  # SELL
            if ob:
                sl = ob.top * (1 + 0.001)
            elif structure.last_lower_high:
                sl = structure.last_lower_high * (1 + 0.001)
            else:
                sl = price + self.atr_sl_mult * atr

            if ote:
                tp = ote.grid.swing_b * (1 - 0.001)
            elif structure.last_lower_low:
                tp = structure.last_lower_low * (1 - 0.001)
            else:
                tp = price - self.atr_tp_mult * atr

        return round(sl, 8), round(tp, 8)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        high = df["high"]
        low  = df["low"]
        close_prev = df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low  - close_prev).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1] or 0.0)

    def _volume_ratio(self, df: pd.DataFrame, window: int = 20) -> float:
        avg = df["volume"].rolling(window).mean().iloc[-1]
        cur = df["volume"].iloc[-1]
        return float(cur / (avg + 1e-8))

    def _hold(self, symbol: str, price: float) -> AdvancedSignal:
        return AdvancedSignal(
            action="HOLD",
            symbol=symbol,
            confidence=0.0,
            stop_loss=price * 0.99,
            take_profit=price * 1.01,
            entry_price=price,
        )
