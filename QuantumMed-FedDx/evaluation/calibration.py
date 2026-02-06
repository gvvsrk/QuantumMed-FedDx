from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch

@dataclass
class CalibrationConfig:
    n_bins: int = 15

def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    K = probs.shape[1]
    y_onehot = np.eye(K)[y]
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == y).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        ece += (np.sum(mask) / N) * abs(float(np.mean(acc[mask])) - float(np.mean(conf[mask])))
    return float(ece)

@torch.no_grad()
def calibration_metrics(model, loader, device: str, cfg: CalibrationConfig) -> Dict[str, float]:
    model.eval()
    probs_list, y_list = [], []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        y = batch["y"].to(device)
        logits, _ = model(tokens)
        probs = torch.softmax(logits, dim=-1)
        probs_list.append(probs.detach().cpu().numpy())
        y_list.append(y.detach().cpu().numpy())
    probs_np = np.concatenate(probs_list, 0)
    y_np = np.concatenate(y_list, 0)
    return {"ece": expected_calibration_error(probs_np, y_np, n_bins=cfg.n_bins),
            "brier": brier_score(probs_np, y_np)}
