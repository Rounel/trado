"""
trading/strategies/ai_strategy.py — Stratégie IA : TFT + RL + Sentiment Grok.

Pipeline :
  1. Buffer OHLCV → TechnicalIndicators → score technique
  2. Features normalisées → TFT + RL → score IA (si modèles chargés)
  3. Sentiment Grok + Order Book (injectés par l'engine) → score externe
  4. SignalFusion + CPO (poids adaptatifs via market_context) → BUY / SELL / HOLD
  5. ProbabilityOfProfitFilter → rejet si PoP < 0.50
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from analysis.aggregator.signal_fusion import SignalFusion
from analysis.corrective.pop_filter import ProbabilityOfProfitFilter
from analysis.technical.indicators import TechnicalIndicators
from trading.strategies.base import BaseStrategy, Signal

if TYPE_CHECKING:
    from config.settings import Settings


class AIStrategy(BaseStrategy):
    """Stratégie 2 — TFT + RL Agent + Grok Sentiment + Order Book."""

    # Nombre minimal de bougies pour calculer les indicateurs (EMA 200 + marge)
    WARMUP_BARS = 210

    def __init__(self, settings: "Settings", symbol: str = "BTC/USDT") -> None:
        super().__init__(settings)
        self.symbol    = symbol
        self._fusion   = SignalFusion()
        self._pop      = ProbabilityOfProfitFilter(min_pop=0.50)
        self._buffer: deque[dict] = deque(maxlen=self.WARMUP_BARS + 10)
        self._tft = None    # chargé via load_models()
        self._rl  = None    # chargé via load_models()

    @property
    def name(self) -> str:
        return "AIStrategy"

    def on_bar(self, features: dict) -> Signal | None:
        """Génère un signal IA en combinant TFT, RL, technique et sentiment."""
        self._buffer.append(features)
        if len(self._buffer) < self.WARMUP_BARS:
            logger.debug(f"AIStrategy [{self.symbol}]: warmup {len(self._buffer)}/{self.WARMUP_BARS}")
            return None

        # --- Calcul des indicateurs sur le buffer complet ---
        df = pd.DataFrame(list(self._buffer))
        df = TechnicalIndicators.add_all(df)
        last = df.iloc[-1]

        if pd.isna(last.get("ema_200") or float("nan")):
            return None

        close = float(last.get("close", 0) or 0)
        if not close:
            return None

        # --- Régime de marché (ajuste les poids de fusion dynamiquement) ---
        from analysis.regime.detector import RegimeDetector
        regime = RegimeDetector().detect(df).value

        # --- Score technique (RSI + EMA + MACD calculés sur buffer) ---
        tech_score = SignalFusion.tech_score_from_indicators(last.to_dict())

        # --- Score IA (TFT + RL si chargés, moyenne pondérée) ---
        ai_score = 0.0
        ai_count = 0
        if self._tft is not None:
            X = self._features_to_array(df)          # (1, seq_len=60, 12)
            pred = self._tft.predict(X)
            ai_score += SignalFusion.ai_score_from_prediction(pred)
            ai_count += 1
        if self._rl is not None:
            X_rl = self._features_to_array_rl(last.to_dict())  # (1, 12) pour PPO
            pred_rl = self._rl.predict(X_rl)
            ai_score += SignalFusion.ai_score_from_prediction(pred_rl)
            ai_count += 1
        if ai_count > 1:
            ai_score /= ai_count

        # --- Sentiment + OB injectés par l'engine (async, via features) ---
        sentiment_score = float(features.get("sentiment_score", 0.0) or 0.0)
        ob_imbalance    = float(features.get("ob_imbalance", 0.0) or 0.0)

        # --- Market context pour CPO et PoP ---
        atr_14       = float(last.get("atr_14", 0) or 0)
        atr_series   = df["atr_14"].dropna()
        atr_roll_50  = float(atr_series.rolling(50).mean().iloc[-1]) if len(atr_series) >= 50 else float(atr_series.mean() or 1e-8)
        atr_ratio    = atr_14 / max(atr_roll_50, 1e-8)
        market_context = {
            "atr_ratio":    atr_ratio,
            "volume_ratio": float(last.get("volume_ratio", 1.0) or 1.0),
            "adx":          float(last.get("adx_14", 25.0) or 25.0),
        }

        # --- Fusion finale avec CPO (poids adaptatifs régime + conditions marché) ---
        fused = self._fusion.fuse(
            tech_score=tech_score,
            ai_score=ai_score,
            sentiment_score=sentiment_score,
            ob_imbalance=ob_imbalance,
            regime=regime,
            market_context=market_context,
        )

        if fused.decision.value == "HOLD":
            return None

        # --- Filtre Corrective AI (PoP < 0.50 → rejet) ---
        allowed, pop = self._pop.should_trade(fused, market_context)
        if not allowed:
            return None

        atr = atr_14
        stops = TechnicalIndicators.atr_stops(
            close, atr,
            sl_mult=self.settings.risk.atr_sl_multiplier,
            tp_mult=self.settings.risk.atr_tp_multiplier,
        )
        action = fused.decision.value
        sl = stops["stop_loss_long"]   if action == "BUY" else stops["stop_loss_short"]
        tp = stops["take_profit_long"] if action == "BUY" else stops["take_profit_short"]

        w = fused.weights_used or {}
        logger.info(
            f"AIStrategy [{self.symbol}]: {action} @ {close:.5f} | "
            f"regime={regime}  score={fused.score:+.3f}  conf={fused.confidence:.2f}  "
            f"PoP={pop:.2f}\n"
            f"  tech={tech_score:+.2f}(x{w.get('tech',0):.2f})  "
            f"ai={ai_score:+.2f}(x{w.get('ai',0):.2f})  "
            f"sent={sentiment_score:+.2f}(x{w.get('sentiment',0):.2f})  "
            f"ob={ob_imbalance:+.2f}(x{w.get('ob',0):.2f})  "
            f"atr_ratio={atr_ratio:.2f}"
        )

        return Signal(
            action=action,
            symbol=self.symbol,
            confidence=fused.confidence,
            stop_loss=sl,
            take_profit=tp,
            strategy=self.name,
            metadata={
                "fused_score":    fused.score,
                "tech_score":     tech_score,
                "ai_score":       ai_score,
                "sentiment":      sentiment_score,
                "ob_imbalance":   ob_imbalance,
                "regime":         regime,
                "weights":        fused.weights_used,
                "cpo":            (fused.metadata or {}).get("cpo"),
                "pop":            pop,
                "rsi":            float(last.get("rsi_14", 0) or 0),
                "macd_hist":      float(last.get("macd_hist", 0) or 0),
                "atr":            atr,
                "atr_ratio":      atr_ratio,
            },
        )

    def load_models(self, tft_path: str | None = None, rl_path: str | None = None) -> None:
        """Charge les modèles pré-entraînés depuis le disque."""
        if tft_path:
            from models.tft.trainer import TFTTrainer
            self._tft = TFTTrainer(input_size=20)
            self._tft.load(tft_path)
            logger.info(f"TFT chargé depuis {tft_path}")
        if rl_path:
            from models.rl_agent.trainer import RLTrainer
            self._rl = RLTrainer()
            self._rl.load(rl_path)
            logger.info(f"RL Agent chargé depuis {rl_path}")

    # Colonnes features (ordre fixe, doit correspondre à training/pipeline.py)
    _FEATURE_COLS = [
        "open", "high", "low", "close", "volume",
        "rsi_14", "ema_20", "ema_50", "macd_hist",
        "bb_pct", "atr_14", "volume_ratio",
    ]
    _SEQ_LEN = 60

    @classmethod
    def _features_to_array(cls, df: pd.DataFrame) -> "np.ndarray":
        """Construit (1, seq_len, n_features) depuis les dernières bougies du buffer."""
        import numpy as np
        arr = df[cls._FEATURE_COLS].fillna(0).values[-cls._SEQ_LEN:].astype(np.float32)
        if len(arr) < cls._SEQ_LEN:
            pad = np.zeros((cls._SEQ_LEN - len(arr), len(cls._FEATURE_COLS)), dtype=np.float32)
            arr = np.concatenate([pad, arr], axis=0)
        return arr.reshape(1, cls._SEQ_LEN, len(cls._FEATURE_COLS))  # (batch=1, seq=60, features=12)

    @classmethod
    def _features_to_array_rl(cls, indicators: dict) -> "np.ndarray":
        """Construit un vecteur plat (1, n_features) pour le PPO."""
        import numpy as np
        arr = np.array(
            [float(indicators.get(k, 0) or 0) for k in cls._FEATURE_COLS],
            dtype=np.float32,
        )
        return arr.reshape(1, -1)
