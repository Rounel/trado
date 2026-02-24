"""
models/tft/architecture.py — Temporal Fusion Transformer (PyTorch).

Implémentation simplifiée du TFT pour la prédiction de séries temporelles.
Architecture basée sur : Lim et al., 2021 "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting".

Composants clés :
  - Gated Residual Network (GRN)
  - Variable Selection Network (VSN)
  - Temporal Self-Attention
  - Quantile output
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """GRN — Bloc de base du TFT avec gating et connexion résiduelle."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h) * torch.sigmoid(self.gate(h))
        return self.layer_norm(out + residual)


class TemporalFusionTransformer(nn.Module):
    """
    TFT simplifié — prend une séquence de features et prédit BUY/HOLD/SELL.

    Pour une implémentation complète multi-horizon avec covariables temporelles,
    utiliser la librairie pytorch-forecasting.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        output_size: int = 3,
        dropout: float = 0.1,
        seq_len: int = 60,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Variable Selection Network (simplifié)
        self.vsn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

        # Temporal Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # GRN de sortie + classifieur
        self.output_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, input_size)
        x = self.input_proj(x)           # → (batch, seq_len, hidden_size)
        x = self.vsn(x)                  # Variable selection
        x = self.transformer(x)          # Temporal attention
        x = x[:, -1, :]                  # Dernier pas de temps
        x = self.output_grn(x)
        return self.classifier(x)        # logits (batch, 3)
