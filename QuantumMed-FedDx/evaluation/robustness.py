from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

@dataclass
class RobustnessConfig:
    noise_std: float = 0.05
    intensity_scale: float = 0.10
    blur_kernel: int = 0
    seed: int = 42

def _gaussian_blur(tokens: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return tokens
    if k % 2 == 0:
        raise ValueError("blur_kernel must be odd.")
    C = tokens.size(-3)
    weight = torch.ones(C, 1, k, k, device=tokens.device, dtype=tokens.dtype) / (k * k)
    pad = k // 2
    if tokens.dim() == 5:
        B, T, C, S, _ = tokens.shape
        x = tokens.view(B*T, C, S, S)
        y = F.conv2d(x, weight, padding=pad, groups=C)
        return y.view(B, T, C, S, S)
    if tokens.dim() == 4:
        return F.conv2d(tokens, weight, padding=pad, groups=C)
    return tokens

def apply_perturbation(tokens: torch.Tensor, cfg: RobustnessConfig, mode: str) -> torch.Tensor:
    torch.manual_seed(cfg.seed)
    x = tokens
    if mode == "noise":
        return x + cfg.noise_std * torch.randn_like(x)
    if mode == "scale":
        scale = (1.0 - cfg.intensity_scale) + (2.0 * cfg.intensity_scale) * torch.rand(1, device=x.device, dtype=x.dtype)
        return x * scale
    if mode == "blur":
        return _gaussian_blur(x, cfg.blur_kernel)
    if mode == "noise+scale":
        x = x + cfg.noise_std * torch.randn_like(x)
        scale = (1.0 - cfg.intensity_scale) + (2.0 * cfg.intensity_scale) * torch.rand(1, device=x.device, dtype=x.dtype)
        return x * scale
    raise ValueError(f"Unknown robustness mode: {mode}")

@torch.no_grad()
def eval_classification_metrics(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    y_np = y.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    out = {"acc": float((pred == y).float().mean().item()),
           "macro_f1": float(f1_score(y_np, pred_np, average="macro"))}
    K = probs_np.shape[1]
    try:
        if K == 2:
            out["auc_roc"] = float(roc_auc_score(y_np, probs_np[:,1]))
            out["auc_pr"] = float(average_precision_score(y_np, probs_np[:,1]))
        else:
            out["auc_roc"] = float(roc_auc_score(y_np, probs_np, multi_class="ovr"))
            pr_list = []
            for k in range(K):
                yk = (y_np == k).astype(np.int32)
                pr_list.append(average_precision_score(yk, probs_np[:,k]))
            out["auc_pr"] = float(np.mean(pr_list))
    except Exception:
        pass
    return out

@torch.no_grad()
def robustness_suite(model, loader, device: str, cfg: RobustnessConfig) -> Dict[str, float]:
    model.eval()
    clean_logits, clean_y = [], []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        y = batch["y"].to(device)
        logits, _ = model(tokens)
        clean_logits.append(logits); clean_y.append(y)
    clean_logits = torch.cat(clean_logits, 0)
    clean_y = torch.cat(clean_y, 0)
    clean_metrics = eval_classification_metrics(clean_logits, clean_y)
    out = {f"clean_{k}": v for k, v in clean_metrics.items()}

    modes = ["noise", "scale"]
    if cfg.blur_kernel and cfg.blur_kernel > 1:
        modes.append("blur")
    modes.append("noise+scale")

    for mode in modes:
        pert_logits, pert_y = [], []
        for batch in loader:
            tokens = batch["tokens"].to(device)
            y = batch["y"].to(device)
            tokens_p = apply_perturbation(tokens, cfg, mode)
            logits, _ = model(tokens_p)
            pert_logits.append(logits); pert_y.append(y)
        pert_logits = torch.cat(pert_logits, 0)
        pert_y = torch.cat(pert_y, 0)
        m = eval_classification_metrics(pert_logits, pert_y)
        for k, v in m.items():
            out[f"{mode}_{k}"] = v
        if "macro_f1" in clean_metrics and "macro_f1" in m:
            out[f"deltaF1_{mode}"] = float(clean_metrics["macro_f1"] - m["macro_f1"])
    return out
