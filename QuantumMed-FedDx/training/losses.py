from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class LossConfig:
    use_class_weights: bool = False
    class_weights: Optional[torch.Tensor] = None
    use_focal: bool = False
    focal_gamma: float = 2.0
    focal_alpha: Optional[torch.Tensor] = None
    emb_loss_weight: float = 0.0

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        p = torch.exp(logp)
        pt = p.gather(1, targets.view(-1, 1)).squeeze(1)
        logpt = logp.gather(1, targets.view(-1, 1)).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            loss = at * loss
        return loss.mean()

def classification_loss(logits: torch.Tensor, y: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    if cfg.use_focal:
        return FocalLoss(gamma=cfg.focal_gamma, alpha=cfg.focal_alpha)(logits, y)
    if cfg.use_class_weights and cfg.class_weights is not None:
        return F.cross_entropy(logits, y, weight=cfg.class_weights)
    return F.cross_entropy(logits, y)

def embedding_separation_loss(z_pool: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    classes = torch.unique(y)
    centroids = []
    for c in classes:
        centroids.append(z_pool[y == c].mean(dim=0))
    centroids = torch.stack(centroids, dim=0)
    class_to_idx = {int(c.item()): i for i, c in enumerate(classes)}
    idx = torch.tensor([class_to_idx[int(t.item())] for t in y], device=y.device)
    z_c = centroids[idx]
    within = (z_pool - z_c).pow(2).mean()
    if centroids.size(0) > 1:
        dist = torch.cdist(centroids, centroids, p=2)
        mask = ~torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
        between = dist[mask].mean()
        return within - 0.1 * between
    return within

def total_loss(logits: torch.Tensor, y: torch.Tensor, aux: dict, cfg: LossConfig) -> torch.Tensor:
    l_cls = classification_loss(logits, y, cfg)
    if cfg.emb_loss_weight > 0.0:
        z_pool = aux.get("z_pool", None)
        if z_pool is None:
            raise ValueError("aux['z_pool'] required for embedding loss")
        l_emb = embedding_separation_loss(z_pool, y)
        return l_cls + cfg.emb_loss_weight * l_emb
    return l_cls
