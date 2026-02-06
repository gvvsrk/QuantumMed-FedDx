from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn

from models.classical.mini_encoder import MiniEncoder, MiniEncoderConfig
from models.quantum.vqc import QFeatEmbedVQC, VQCConfig

@dataclass
class VQMedNetConfig:
    num_classes: int
    q_qubits: int
    token_pool: Literal["mean"] = "mean"
    head_hidden: int = 64
    dropout: float = 0.1

class VQMedNet(nn.Module):
    def __init__(self, cfg: VQMedNetConfig, mini_cfg: MiniEncoderConfig, vqc_cfg: VQCConfig):
        super().__init__()
        if cfg.q_qubits != vqc_cfg.n_qubits:
            raise ValueError("q_qubits mismatch")
        self.cfg = cfg
        self.mini = MiniEncoder(mini_cfg, q_qubits=cfg.q_qubits)
        self.vqc = QFeatEmbedVQC(vqc_cfg)
        emb_dim = self.vqc.embedding_dim(d=mini_cfg.d_out)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.num_classes),
        )

    def forward(self, tokens: torch.Tensor, token_counts: Optional[torch.Tensor] = None):
        if tokens.dim() == 5:
            B, T, C, S, S2 = tokens.shape
            tokens_flat = tokens.view(B*T, C, S, S)
            f = self.mini(tokens_flat)
            z = self.vqc(f)
            z = z.view(B, T, -1)
            z_pool = z.mean(dim=1)
        elif tokens.dim() == 4:
            raise ValueError("Flattened token mode not enabled in this runner yet.")
        else:
            raise ValueError(f"Unexpected tokens shape: {tuple(tokens.shape)}")
        logits = self.head(z_pool)
        return logits, {"z_pool": z_pool}
