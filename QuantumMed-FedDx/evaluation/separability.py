from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    from sklearn.manifold import TSNE
except Exception:
    TSNE = None

try:
    import umap
except Exception:
    umap = None

@dataclass
class SeparabilityConfig:
    knn_k: int = 5
    reducer: str = "none"
    reducer_dim: int = 2
    reducer_seed: int = 42

@torch.no_grad()
def collect_embeddings(model, loader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    Z_list, y_list = [], []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        y = batch["y"].to(device)
        _, aux = model(tokens)
        z_pool = aux.get("z_pool")
        if z_pool is None:
            raise ValueError("aux['z_pool'] missing.")
        Z_list.append(z_pool.detach().cpu().numpy())
        y_list.append(y.detach().cpu().numpy())
    Z = np.concatenate(Z_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return Z, y

def knn_separability(Z_train: np.ndarray, y_train: np.ndarray, Z_test: np.ndarray, y_test: np.ndarray, k: int) -> Dict[str, float]:
    knn = KNeighborsClassifier(n_neighbors=int(k))
    knn.fit(Z_train, y_train)
    y_pred = knn.predict(Z_test)
    return {"knn_acc": float(accuracy_score(y_test, y_pred)),
            "knn_macro_f1": float(f1_score(y_test, y_pred, average="macro"))}

def reduce_embeddings(Z: np.ndarray, cfg: SeparabilityConfig) -> Optional[np.ndarray]:
    if cfg.reducer == "none":
        return None
    if cfg.reducer == "tsne":
        if TSNE is None:
            return None
        return TSNE(n_components=cfg.reducer_dim, random_state=cfg.reducer_seed, init="pca", learning_rate="auto").fit_transform(Z)
    if cfg.reducer == "umap":
        if umap is None:
            return None
        return umap.UMAP(n_components=cfg.reducer_dim, random_state=cfg.reducer_seed).fit_transform(Z)
    raise ValueError(f"Unknown reducer: {cfg.reducer}")
