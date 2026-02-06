from __future__ import annotations
from typing import Dict, List
import numpy as np
import nibabel as nib

def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)

def zscore_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mask = x != 0
    if not np.any(mask):
        return x
    mu = x[mask].mean()
    sigma = x[mask].std()
    return (x - mu) / (sigma + eps)

def preprocess_brats_modalities(paths: Dict[str, str]) -> np.ndarray:
    """Return volume shape (C,H,W,D) for modalities [t1,t1ce,t2,flair]."""
    vols = []
    for mod in ["t1", "t1ce", "t2", "flair"]:
        v = load_nifti(paths[mod])
        v = zscore_normalize(v)
        vols.append(v)
    return np.stack(vols, axis=0)

def extract_valid_slices(mask: np.ndarray, min_area: int = 50) -> List[int]:
    idxs = []
    for z in range(mask.shape[-1]):
        if np.sum(mask[:, :, z] > 0) >= min_area:
            idxs.append(z)
    return idxs

def extract_2p5d_slice(volume: np.ndarray, z: int, context: int = 1) -> np.ndarray:
    """volume: (C,H,W,D) -> return (C*(2c+1),H,W)"""
    slices = []
    for dz in range(-context, context + 1):
        zz = int(np.clip(z + dz, 0, volume.shape[-1] - 1))
        slices.append(volume[..., zz])
    return np.concatenate(slices, axis=0)

def extract_tokens(img: np.ndarray, token_size: int, stride: int, max_tokens: int) -> np.ndarray:
    """img: (C,H,W) -> (T,C,S,S)"""
    C, H, W = img.shape
    tokens = []
    for i in range(0, H - token_size + 1, stride):
        for j in range(0, W - token_size + 1, stride):
            tokens.append(img[:, i:i+token_size, j:j+token_size])
            if len(tokens) >= max_tokens:
                return np.stack(tokens).astype(np.float32)
    if tokens:
        return np.stack(tokens).astype(np.float32)
    return np.zeros((1, C, token_size, token_size), dtype=np.float32)
