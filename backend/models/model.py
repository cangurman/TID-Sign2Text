"""Sequence classification models for TID-Sign2Text.

Input:  (batch, seq_len, 258) normalized landmark sequences.
Output: (batch, num_classes) logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """Bi-LSTM over landmark sequences with mean+max pooling head."""

    def __init__(
        self,
        input_dim: int = 258,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        out, _ = self.lstm(h)                     # (B, T, 2H)
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)  # (B, 4H)
        return self.head(pooled)


class TransformerClassifier(nn.Module):
    """Small Transformer encoder alternative (try when data grows)."""

    def __init__(
        self,
        input_dim: int = 258,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        num_classes: int = 10,
        dropout: float = 0.2,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos_embed[:, : x.shape[1]]
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


def build_model(name: str, num_classes: int, input_dim: int = 258) -> nn.Module:
    if name == "bilstm":
        return BiLSTMClassifier(input_dim=input_dim, num_classes=num_classes)
    if name == "transformer":
        return TransformerClassifier(input_dim=input_dim, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name!r} (expected 'bilstm' or 'transformer')")
