from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

from preprocessing.ct_preprocess import (
    preprocess_ct, select_relevant_slices, extract_slice_with_context, extract_tokens
)

@dataclass
class LIDCConfig:
    root_dir: str
    split: str                      # train | val | test
    token_size: int = 32
    stride: int = 32
    max_tokens: int = 16
    context_slices: int = 1
    label_map: Optional[Dict[str, int]] = None

class LIDCDataset(Dataset):
    """Expected per-case files: ct.nii.gz and mask.nii.gz within each case directory."""
    def __init__(self, cfg: LIDCConfig):
        self.cfg = cfg
        self.cases = self._collect_cases()

    def _collect_cases(self) -> List[str]:
        split_dir = os.path.join(self.cfg.root_dir, self.cfg.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"LIDC split directory not found: {split_dir}")
        return sorted([os.path.join(split_dir, d) for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int):
        case_dir = self.cases[idx]
        ct_path = os.path.join(case_dir, "ct.nii.gz")
        mask_path = os.path.join(case_dir, "mask.nii.gz")

        vol = preprocess_ct(ct_path)                                 # (H,W,D)
        mask = nib.load(mask_path).get_fdata().astype(np.float32)    # (H,W,D)

        relevant = select_relevant_slices(mask, min_pixels=20)
        z = relevant[len(relevant)//2] if relevant else (mask.shape[-1] // 2)

        img = extract_slice_with_context(vol, z, context=self.cfg.context_slices)  # (C,H,W)
        tokens = extract_tokens(img, token_size=self.cfg.token_size, stride=self.cfg.stride, max_tokens=self.cfg.max_tokens)

        y = 1 if np.sum(mask > 0) > 0 else 0
        if self.cfg.label_map is not None:
            case_id = os.path.basename(case_dir)
            y = int(self.cfg.label_map.get(case_id, y))

        return {"tokens": torch.from_numpy(tokens).float(),
                "y": torch.tensor(y, dtype=torch.long)}
