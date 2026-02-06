from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MiniEncoderConfig:
    encoder_type: Literal["cnn", "vit_stem"] = "cnn"
    in_channels: int = 4
    token_size: int = 32
    d_out: int = 6
    hidden_dim: int = 64
    vit_num_heads: int = 4
    vit_mlp_ratio: float = 2.0
    vit_num_layers: int = 1
    dropout: float = 0.1
    norm_type: Literal["l2", "layernorm"] = "l2"

class CNNMiniEncoder(nn.Module):
    def __init__(self, cfg: MiniEncoderConfig):
        super().__init__()
        C = cfg.in_channels
        H = cfg.hidden_dim
        self.features = nn.Sequential(
            nn.Conv2d(C, H, 3, padding=1, bias=False),
            nn.BatchNorm2d(H),
            nn.ReLU(inplace=True),
            nn.Conv2d(H, H, 3, padding=1, bias=False),
            nn.BatchNorm2d(H),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(H, 2*H, 3, padding=1, bias=False),
            nn.BatchNorm2d(2*H),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Linear(2*H, cfg.d_out)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        h = self.dropout(h)
        return self.proj(h)

class ViTStemMiniEncoder(nn.Module):
    def __init__(self, cfg: MiniEncoderConfig):
        super().__init__()
        S = cfg.token_size
        C = cfg.in_channels
        D = cfg.hidden_dim
        inner_patch = 8 if S >= 32 else 4
        if S % inner_patch != 0:
            raise ValueError("token_size must be divisible by inner_patch")
        n = (S // inner_patch) * (S // inner_patch)
        self.patch_embed = nn.Conv2d(C, D, kernel_size=inner_patch, stride=inner_patch, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.pos = nn.Parameter(torch.zeros(1, 1 + n, D))
        self.drop = nn.Dropout(cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=cfg.vit_num_heads,
            dim_feedforward=int(D * cfg.vit_mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=cfg.vit_num_layers)
        self.proj = nn.Linear(D, cfg.d_out)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch_embed(x)               # (B,D,Sh,Sw)
        z = z.flatten(2).transpose(1,2)       # (B,n,D)
        B = z.size(0)
        cls = self.cls.expand(B, -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = z + self.pos[:, :z.size(1), :]
        z = self.drop(z)
        z = self.enc(z)
        return self.proj(z[:,0,:])

class FeatureNormalizer(nn.Module):
    def __init__(self, norm_type: str, d_out: int):
        super().__init__()
        self.norm_type = norm_type
        self.ln = nn.LayerNorm(d_out) if norm_type == "layernorm" else None

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "layernorm":
            return self.ln(f)
        return F.normalize(f, p=2, dim=-1, eps=1e-8)

class MiniEncoder(nn.Module):
    def __init__(self, cfg: MiniEncoderConfig, q_qubits: int):
        super().__init__()
        if cfg.d_out > q_qubits:
            raise ValueError(f"d_out={cfg.d_out} must satisfy d_out <= q_qubits={q_qubits}")
        self.cfg = cfg
        if cfg.encoder_type == "cnn":
            self.enc = CNNMiniEncoder(cfg)
        elif cfg.encoder_type == "vit_stem":
            self.enc = ViTStemMiniEncoder(cfg)
        else:
            raise ValueError(f"Unknown encoder_type: {cfg.encoder_type}")
        self.norm = FeatureNormalizer(cfg.norm_type, cfg.d_out)

    def forward(self, token_batch: torch.Tensor) -> torch.Tensor:
        f = self.enc(token_batch)
        return self.norm(f)
