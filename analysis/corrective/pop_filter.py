"""
analysis/corrective/pop_filter.py — Filtre Corrective AI basé sur la Probabilité de Profit (PoP).

Principe :
  Chaque signal fusionné est soumis à une évaluation de sa qualité intrinsèque.
  Si la Probabilité de Profit estimée < min_pop (défaut 0.50), le signal est rejeté.

La PoP est calculée comme la moyenne géométrique de 4 composantes :
  1. Alignement     : fraction des composantes (tech/AI/sentiment) pointant dans le même sens
  2. Force du signal : |score fusionné| normalisé
  3. Adéquation régime-direction : signal cohérent avec le régime détecté
  4. Pénalité volatilité : ATR élevé = incertitude = PoP réduite

La moyenne géométrique est plus stricte que l'arithmétique : une seule composante
très faible (ex: désalignement total) suffit à bloquer le trade.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from analysis.aggregator.signal_fusion import Decision, FusedSignal


class ProbabilityOfProfitFilter:
    """
    Filtre Corrective AI — rejette les signaux dont la PoP < min_pop.

    Usage dans AIStrategy :
        pop_filter = ProbabilityOfProfitFilter(min_pop=0.50)
        allowed, pop = pop_filter.should_trade(fused_signal, market_context)
        if not allowed:
            return None   # signal refusé
    """

    def __init__(self, min_pop: float = 0.50) -> None:
        self.min_pop = float(max(0.0, min(1.0, min_pop)))

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    def compute(
        self,
        fused: "FusedSignal",
        market_context: dict | None = None,
    ) -> float:
        """
        Calcule la Probabilité de Profit [0.0, 1.0].

        Args:
            fused          : résultat de SignalFusion.fuse()
            market_context : dict optionnel avec clés :
                               atr_ratio    — ATR_courant / ATR_rolling_mean (défaut 1.0)
                               volume_ratio — volume / volume_rolling_mean   (défaut 1.0)

        Returns:
            float en [0.0, 1.0]
        """
        context = market_context or {}
        components: list[float] = []

        # 1. Alignement des composantes
        alignment = self._alignment_score(fused)
        components.append(alignment)

        # 2. Force du signal (|score| normalisé ; 0.5 correspond à un signal "fort")
        strength = min(abs(fused.score) / 0.50, 1.0)
        components.append(max(strength, 1e-6))

        # 3. Adéquation régime × direction
        regime_fit = self._regime_fit(fused.regime, fused.decision)
        components.append(regime_fit)

        # 4. Pénalité volatilité
        atr_ratio  = float(context.get("atr_ratio", 1.0) or 1.0)
        vol_penalty = self._volatility_penalty(atr_ratio)
        components.append(vol_penalty)

        # PoP = moyenne géométrique des composantes
        pop = math.exp(
            sum(math.log(max(c, 1e-6)) for c in components) / len(components)
        )
        return float(min(max(pop, 0.0), 1.0))

    def should_trade(
        self,
        fused: "FusedSignal",
        market_context: dict | None = None,
    ) -> tuple[bool, float]:
        """
        Décision binaire + PoP.

        Returns:
            (trade_autorisé, pop_score)
        """
        pop     = self.compute(fused, market_context)
        allowed = pop >= self.min_pop

        if not allowed:
            logger.debug(
                f"PoP filter: signal {fused.decision.value} refusé "
                f"(PoP={pop:.2f} < seuil={self.min_pop:.2f}  "
                f"score={fused.score:+.3f}  regime={fused.regime})"
            )
        return allowed, pop

    # ------------------------------------------------------------------
    # Composantes privées
    # ------------------------------------------------------------------

    def _alignment_score(self, fused: "FusedSignal") -> float:
        """
        Fraction des composantes non-neutres pointant dans la même direction que la décision.
        [0.0 = tous opposés, 1.0 = tous alignés]
        """
        from analysis.aggregator.signal_fusion import Decision

        if fused.decision == Decision.HOLD:
            return 0.5   # HOLD n'a pas de direction — contribution neutre

        direction = 1.0 if fused.decision == Decision.BUY else -1.0
        components = [fused.tech_score, fused.ai_score, fused.sentiment_score]

        # Filtre les composantes quasi-neutres (bruit < 5%)
        active = [c for c in components if abs(c) > 0.05]
        if not active:
            return 0.5   # aucune composante active → neutre

        aligned = sum(1 for c in active if c * direction > 0)
        return aligned / len(active)

    @staticmethod
    def _regime_fit(regime: str, decision: "Decision") -> float:
        """
        Score [0.5, 1.0] selon la cohérence entre régime et direction du signal.
        Un BUY en trend_up ou un SELL en trend_down est idéal (1.0).
        Aller contre le régime (BUY en trend_down) est pénalisé (0.5).
        """
        from analysis.aggregator.signal_fusion import Decision

        if decision == Decision.HOLD:
            return 0.5

        _TABLE: dict[tuple[str, str], float] = {
            ("trend_up",   "BUY"):  1.00,
            ("trend_up",   "SELL"): 0.50,
            ("trend_down", "SELL"): 1.00,
            ("trend_down", "BUY"):  0.50,
            ("range",      "BUY"):  0.70,
            ("range",      "SELL"): 0.70,
            ("volatile",   "BUY"):  0.55,
            ("volatile",   "SELL"): 0.55,
            ("unknown",    "BUY"):  0.65,
            ("unknown",    "SELL"): 0.65,
        }
        return _TABLE.get((regime, decision.value), 0.65)

    @staticmethod
    def _volatility_penalty(atr_ratio: float) -> float:
        """
        Pénalité liée à la volatilité relative.
        atr_ratio = ATR_courant / ATR_rolling_mean_50
          > 2.5 → marché très volatile → PoP réduite (0.40)
          > 1.8 → volatile → pénalité modérée (0.65)
          [0.7, 1.8] → normal → pas de pénalité (1.00)
          < 0.7 → faible volatilité → légère prime (1.00)
        """
        if atr_ratio > 2.5:
            return 0.40
        elif atr_ratio > 1.8:
            return 0.65
        else:
            return 1.00
