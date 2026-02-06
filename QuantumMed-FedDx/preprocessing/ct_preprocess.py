from __future__ import annotations
from typing import List, Optional
import numpy as np
import nibabel as nib

def load_ct_volume(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)

def hu_window(x: np.ndarray, center: int = -600, width: int = 1500) -> np.ndarray:
    lo = center - width // 2
    hi = center + width // 2
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo)

def normalize_ct(x: np.ndarray) -> np.ndarray:
    mu = x.mean()
    std = x.std() + 1e-6
    return (x - mu) / std

def preprocess_ct(path: str, center: int = -600, width: int = 1500) -> np.ndarray:
    vol = load_ct_volume(path)
    vol = hu_window(vol, center=center, width=width)
    vol = normalize_ct(vol)
    return vol

def select_relevant_slices(mask: np.ndarray, min_pixels: int = 20) -> List[int]:
    idxs = []
    for z in range(mask.shape[-1]):
        if np.sum(mask[:, :, z] > 0) >= min_pixels:
            idxs.append(z)
    return idxs

def extract_slice_with_context(vol: np.ndarray, z: int, context: int = 1) -> np.ndarray:
    slices = []
    for dz in range(-context, context + 1):
        zz = int(np.clip(z + dz, 0, vol.shape[-1] - 1))
        slices.append(vol[:, :, zz])
    return np.stack(slices, axis=0).astype(np.float32)  # (C,H,W)

def extract_tokens(img: np.ndarray, token_size: int, stride: int, max_tokens: int) -> np.ndarray:
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
