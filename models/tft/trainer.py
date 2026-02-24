"""
models/tft/trainer.py — Entraînement du Temporal Fusion Transformer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from models.base import BaseModel, Action, Prediction
from models.tft.architecture import TemporalFusionTransformer


class TFTTrainer(BaseModel):
    """Entraîne et utilise le TFT pour prédire BUY/HOLD/SELL."""

    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        lr: float = 3e-4,
        epochs: int = 50,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = TemporalFusionTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            seq_len=seq_len,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.criterion = nn.CrossEntropyLoss()

    @property
    def name(self) -> str:
        return "TFT"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(self.device)

        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"TFT Epoch {epoch+1}/{self.epochs} — loss={total_loss/len(loader):.4f}")

    def predict(self, X: np.ndarray) -> Prediction:
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
        logger.info(f"TFT sauvegardé → {path}")

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
