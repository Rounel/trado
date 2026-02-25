"""
models/lstm/trainer.py — Entraînement LSTM.

Améliorations vs v1 :
  - Early stopping : restaure les meilleurs poids si la loss de validation stagne
  - Dropout adaptatif : augmente le dropout si val_loss > 2× train_loss (surajustement)
  - Gradient clipping (norme 1.0)
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from models.base import BaseModel, Action, Prediction
from models.lstm.architecture import LSTMModel

_DROPOUT_MIN  = 0.05
_DROPOUT_MAX  = 0.50
_DROPOUT_STEP = 0.05


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
        dropout: float = 0.2,
        device: str | None = None,
    ) -> None:
        self.seq_len    = seq_len
        self.epochs     = epochs
        self.batch_size = batch_size
        self._dropout   = dropout
        self.device     = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    @property
    def name(self) -> str:
        return "LSTM"

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        patience: int = 7,
    ) -> None:
        """
        Entraîne le LSTM.

        Args:
            X_train, y_train : données d'entraînement (séquences shape (N, seq_len, features))
            X_val, y_val     : données de validation (activent l'early stopping si fournis)
            patience         : nombre d'époques sans amélioration avant arrêt précoce
        """
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        has_val       = X_val is not None and y_val is not None
        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0

        for epoch in range(self.epochs):
            # --- Passe d'entraînement ---
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss   = self.criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(loader)

            if not has_val:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"LSTM Epoch {epoch+1}/{self.epochs} — train_loss={train_loss:.4f}")
                continue

            # --- Validation + early stopping ---
            val_loss = self._eval_loss(X_val, y_val)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"LSTM Epoch {epoch+1}/{self.epochs} — "
                    f"train={train_loss:.4f}  val={val_loss:.4f}  "
                    f"dropout={self._dropout:.2f}"
                )

            # Dropout adaptatif
            if val_loss > 2.0 * train_loss and self._dropout < _DROPOUT_MAX:
                self._dropout = min(self._dropout + _DROPOUT_STEP, _DROPOUT_MAX)
                self.model.set_dropout(self._dropout)
                logger.debug(
                    f"LSTM: dropout augmenté → {self._dropout:.2f} "
                    f"(val/train={val_loss/train_loss:.1f}x)"
                )
            elif val_loss < 1.1 * train_loss and self._dropout > _DROPOUT_MIN:
                self._dropout = max(self._dropout - _DROPOUT_STEP, _DROPOUT_MIN)
                self.model.set_dropout(self._dropout)
                logger.debug(f"LSTM: dropout réduit → {self._dropout:.2f}")

            # Sauvegarde du meilleur état
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state    = copy.deepcopy(self.model.state_dict())
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(
                        f"LSTM early stopping: epoch {epoch+1}  "
                        f"best_val_loss={best_val_loss:.4f}"
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("LSTM: meilleurs poids restaurés après early stopping")

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> Prediction:
        """Génère une prédiction pour une séquence de shape (1, seq_len, input_size)."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            if X_t.dim() == 2:
                X_t = X_t.unsqueeze(0)
            logits = self.model(X_t)
            probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        action_idx = int(probs.argmax())
        return Prediction(
            action=Action(action_idx),
            confidence=float(probs[action_idx]),
            probabilities={
                "SELL": float(probs[0]),
                "HOLD": float(probs[1]),
                "BUY":  float(probs[2]),
            },
            model_name=self.name,
        )

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"LSTM sauvegardé → {path}")

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info(f"LSTM chargé <- {path}")

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    @staticmethod
    def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Découpe les données en séquences pour l'entraînement."""
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs), np.array(ys)

    def _eval_loss(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Calcule la cross-entropy loss sur le jeu de validation (sans gradient)."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_t = torch.tensor(y_val, dtype=torch.long).to(self.device)
            logits = self.model(X_t)
            loss   = self.criterion(logits, y_t)
        return float(loss.item())
