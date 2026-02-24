"""
models/lstm/trainer.py — Entraînement LSTM avec Walk-Forward Validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from models.base import BaseModel, Action, Prediction
from models.lstm.architecture import LSTMModel


class LSTMTrainer(BaseModel):
    """Entraîne et utilise un LSTM pour prédire BUY/HOLD/SELL."""

    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    @property
    def name(self) -> str:
        return "LSTM"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Entraîne le LSTM sur les données X_train (séquences) et y_train (labels 0/1/2)."""
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                logger.info(f"LSTM Epoch {epoch+1}/{self.epochs} — loss={avg_loss:.4f}")

    def predict(self, X: np.ndarray) -> Prediction:
        """Génère une prédiction pour une séquence de shape (1, seq_len, input_size)."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            if X_t.dim() == 2:
                X_t = X_t.unsqueeze(0)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        action_idx = int(probs.argmax())
        return Prediction(
            action=Action(action_idx),
            confidence=float(probs[action_idx]),
            probabilities={"SELL": float(probs[0]), "HOLD": float(probs[1]), "BUY": float(probs[2])},
            model_name=self.name,
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"LSTM sauvegardé → {path}")

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info(f"LSTM chargé ← {path}")

    @staticmethod
    def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Découpe les données en séquences pour l'entraînement."""
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs), np.array(ys)
