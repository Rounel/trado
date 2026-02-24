"""
models/rl_agent/environment.py — Environnement Gymnasium custom pour le trading.

TradingEnv expose :
  - observation_space : OHLCV + indicateurs techniques (N features)
  - action_space      : Discrete(3) → 0=SELL, 1=HOLD, 2=BUY
  - reward            : Sharpe ratio différentiel ou PnL normalisé
"""
from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Environnement de trading pour l'entraînement RL (PPO via Stable-Baselines3).

    Paramètres:
        prices   : np.ndarray shape (T,) — prix de clôture
        features : np.ndarray shape (T, n_features) — features normalisées
        initial_capital : float — capital de départ
        commission_pct  : float — frais de transaction (%)
        window_size     : int   — nombre de pas observés à chaque step
    """

    metadata = {"render_modes": ["human"]}

    ACTION_SELL = 0
    ACTION_HOLD = 1
    ACTION_BUY  = 2

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        initial_capital: float = 10_000.0,
        commission_pct: float = 0.001,
        window_size: int = 60,
    ) -> None:
        super().__init__()

        assert len(prices) == len(features), "prices et features doivent avoir la même longueur"
        assert len(prices) > window_size, "Pas assez de données"

        self.prices = prices
        self.features = features
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.window_size = window_size

        n_features = features.shape[1]
        # Observation : fenêtre de features + position courante (0=short, 0.5=flat, 1=long)
        obs_shape = (window_size * n_features + 3,)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # SELL / HOLD / BUY

        self._reset_state()

    def _reset_state(self) -> None:
        self._step_idx = self.window_size
        self._capital = self.initial_capital
        self._position = 0.0      # quantité BTC détenue
        self._entry_price = 0.0
        self._portfolio_values: list[float] = [self.initial_capital]
        self._prev_portfolio = self.initial_capital

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        price = self.prices[self._step_idx]
        reward = 0.0
        done = False

        # ---- Exécution de l'action ----
        if action == self.ACTION_BUY and self._capital > 0:
            qty = (self._capital * 0.95) / price  # 95% du capital
            cost = qty * price * (1 + self.commission_pct)
            if cost <= self._capital:
                self._capital -= cost
                self._position += qty
                self._entry_price = price

        elif action == self.ACTION_SELL and self._position > 0:
            proceeds = self._position * price * (1 - self.commission_pct)
            self._capital += proceeds
            self._position = 0.0

        # ---- Valeur du portefeuille ----
        portfolio_value = self._capital + self._position * price
        self._portfolio_values.append(portfolio_value)

        # ---- Reward = rendement log normalisé ----
        if self._prev_portfolio > 0:
            reward = float(np.log(portfolio_value / self._prev_portfolio + 1e-8))
        self._prev_portfolio = portfolio_value

        # ---- Avancement ----
        self._step_idx += 1
        if self._step_idx >= len(self.prices) - 1:
            done = True

        info = {
            "portfolio_value": portfolio_value,
            "capital": self._capital,
            "position": self._position,
            "price": price,
        }
        return self._get_obs(), reward, done, False, info

    def _get_obs(self) -> np.ndarray:
        window = self.features[self._step_idx - self.window_size : self._step_idx]
        flat_features = window.flatten().astype(np.float32)

        price = self.prices[self._step_idx]
        portfolio_value = self._capital + self._position * price
        extra = np.array(
            [
                self._position * price / (portfolio_value + 1e-8),  # pct en position
                self._capital / (portfolio_value + 1e-8),            # pct en cash
                (price - self._entry_price) / (self._entry_price + 1e-8),  # PnL latent
            ],
            dtype=np.float32,
        )
        return np.concatenate([flat_features, extra])

    def sharpe_ratio(self) -> float:
        """Calcule le Sharpe ratio de l'épisode."""
        returns = np.diff(self._portfolio_values) / (np.array(self._portfolio_values[:-1]) + 1e-8)
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(365 * 24))

    def render(self) -> None:
        price = self.prices[self._step_idx - 1]
        pv = self._capital + self._position * price
        print(f"Step {self._step_idx} | Price={price:.2f} | Portfolio={pv:.2f} | Pos={self._position:.4f}")
