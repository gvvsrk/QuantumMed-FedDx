from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.losses import LossConfig, total_loss

@dataclass
class TrainConfig:
    epochs: int = 20
    lr_classical: float = 1e-3
    lr_quantum: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    shot_schedule: Optional[List[Tuple[int, Optional[int]]]] = None
    early_stop_patience: int = 5

def _apply_shot_schedule(model: nn.Module, epoch: int, schedule: Optional[List[Tuple[int, Optional[int]]]]) -> None:
    if schedule is None:
        return
    shots = None
    for ep_start, sh in schedule:
        if epoch >= ep_start:
            shots = sh
    if hasattr(model, "vqc") and hasattr(model.vqc, "dev"):
        model.vqc.dev.shots = shots
        if hasattr(model.vqc, "cfg"):
            model.vqc.cfg.shots = shots

def _separate_param_groups(model: nn.Module):
    quantum_params = []
    classical_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "vqc.theta" in name or name.endswith("vqc.theta"):
            quantum_params.append(p)
        else:
            classical_params.append(p)
    return classical_params, quantum_params

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        y = batch["y"].to(device)
        logits, _ = model(tokens)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.size(0)
    return {"acc": correct / max(1, total), "loss": loss_sum / max(1, total)}

def train_vqmednet(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, train_cfg: TrainConfig, loss_cfg: LossConfig) -> Dict[str, float]:
    device = train_cfg.device
    model.to(device)

    classical_params, quantum_params = _separate_param_groups(model)
    optim = torch.optim.AdamW(
        [{"params": classical_params, "lr": train_cfg.lr_classical},
         {"params": quantum_params, "lr": train_cfg.lr_quantum}],
        weight_decay=train_cfg.weight_decay
    )

    best_val = -1.0
    best_state = None
    patience = 0

    for epoch in range(1, train_cfg.epochs + 1):
        _apply_shot_schedule(model, epoch, train_cfg.shot_schedule)
        model.train()
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            y = batch["y"].to(device)

            optim.zero_grad(set_to_none=True)
            logits, aux = model(tokens)
            loss = total_loss(logits, y, aux, loss_cfg)
            loss.backward()
            if train_cfg.grad_clip and train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optim.step()

        val_metrics = evaluate(model, val_loader, device=device)
        if val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_acc": best_val}
