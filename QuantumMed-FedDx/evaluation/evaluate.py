from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import torch
from sklearn.metrics import f1_score, accuracy_score

from evaluation.separability import SeparabilityConfig, collect_embeddings, knn_separability, reduce_embeddings
from evaluation.robustness import RobustnessConfig, robustness_suite
from evaluation.calibration import CalibrationConfig, calibration_metrics
from evaluation.complexity import quantum_complexity

@dataclass
class EvalConfig:
    separability: SeparabilityConfig = SeparabilityConfig()
    robustness: RobustnessConfig = RobustnessConfig()
    calibration: CalibrationConfig = CalibrationConfig()

@torch.no_grad()
def test_metrics(model, loader, device: str) -> Dict[str, float]:
    model.eval()
    logits_list, y_list = [], []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        y = batch["y"].to(device)
        logits, _ = model(tokens)
        logits_list.append(logits); y_list.append(y)
    logits = torch.cat(logits_list, 0)
    y = torch.cat(y_list, 0)
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    y_np = y.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    return {"acc": float(accuracy_score(y_np, pred_np)),
            "macro_f1": float(f1_score(y_np, pred_np, average="macro")),
            "n": float(y.numel())}

def run_full_evaluation(model, train_loader, test_loader, device: str, cfg: EvalConfig) -> Dict[str, object]:
    model.to(device)
    out: Dict[str, object] = {}
    out.update({f"test_{k}": v for k, v in test_metrics(model, test_loader, device).items()})
    Z_tr, y_tr = collect_embeddings(model, train_loader, device)
    Z_te, y_te = collect_embeddings(model, test_loader, device)
    out.update(knn_separability(Z_tr, y_tr, Z_te, y_te, k=cfg.separability.knn_k))
    coords = reduce_embeddings(Z_te, cfg.separability)
    if coords is not None:
        out["embed_coords"] = coords
    out.update(robustness_suite(model, test_loader, device, cfg.robustness))
    out.update(calibration_metrics(model, test_loader, device, cfg.calibration))
    out["quantum_complexity"] = quantum_complexity(model)
    return out
