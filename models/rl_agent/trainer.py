"""
models/rl_agent/trainer.py — Entraînement PPO via Stable-Baselines3.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from models.base import BaseModel, Action, Prediction
from models.rl_agent.environment import TradingEnv


class RLTrainer(BaseModel):
    """Entraîne un agent PPO sur TradingEnv et l'utilise pour l'inférence."""

    def __init__(
        self,
        total_timesteps: int = 500_000,
        n_steps: int = 2048,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        device: str = "auto",
    ) -> None:
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self._agent = None
        self._env = None

    @property
    def name(self) -> str:
        return "PPO-RL"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray | None = None) -> None:
        """
        y_train ignoré — l'agent apprend par RL, pas en supervisé.
        X_train : features de shape (T, n_features).
        y_train peut être un vecteur de prix (T,) passé par le pipeline,
        sinon on utilise X_train[:,3] (colonne 'close').
        """
        from stable_baselines3 import PPO  # type: ignore[import]

        if y_train is not None and y_train.ndim == 1 and len(y_train) == len(X_train):
            prices = y_train.astype(np.float32)
        else:
            prices = X_train[:, 3].astype(np.float32)  # colonne close
        features = X_train

        self._env = TradingEnv(prices=prices, features=features)
        self._agent = PPO(
            "MlpPolicy",
            self._env,
            verbose=0,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            device=self.device,
        )
        logger.info(f"RL Agent (PPO) — début entraînement {self.total_timesteps} timesteps")
        self._agent.learn(total_timesteps=self.total_timesteps)
        logger.info("RL Agent (PPO) — entraînement terminé")

    def predict(self, X: np.ndarray) -> Prediction:
        """Prédit l'action et les probabilités pour un état courant."""
        if self._agent is None:
            raise RuntimeError("Agent non encore entraîné — appeler fit() d'abord.")

        import torch
        obs = X.flatten().astype(np.float32)
        action, _ = self._agent.predict(obs, deterministic=True)
        action_int = int(action)

        # Récupère les probabilités réelles de la politique PPO
        try:
            obs_tensor, _ = self._agent.policy.obs_to_tensor(obs)
            with torch.no_grad():
                dist = self._agent.policy.get_distribution(obs_tensor)
                probs_arr = dist.distribution.probs.cpu().numpy().flatten()
            probs = {
                "SELL": float(probs_arr[0]),
                "HOLD": float(probs_arr[1]),
                "BUY":  float(probs_arr[2]),
            }
        except Exception:
            # Fallback : confiance totale sur l'action choisie
            probs = {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
            probs[Action(action_int).name] = 1.0

        return Prediction(
            action=Action(action_int),
            confidence=probs[Action(action_int).name],
            probabilities=probs,
            model_name=self.name,
            metadata={"raw_action": action_int},
        )

    def save(self, path: str) -> None:
        if self._agent is None:
            raise RuntimeError("Aucun agent à sauvegarder.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._agent.save(path)
        logger.info(f"RL Agent sauvegardé → {path}")

    def load(self, path: str) -> None:
        from stable_baselines3 import PPO

        self._agent = PPO.load(path, device=self.device)
        logger.info(f"RL Agent chargé ← {path}")
