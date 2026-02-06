from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

from preprocessing.mri_preprocess import (
    preprocess_brats_modalities, extract_valid_slices, extract_2p5d_slice, extract_tokens
)

@dataclass
class BraTSConfig:
    root_dir: str
    split: str                      # train | val | test
    token_size: int = 32
    stride: int = 32
    max_tokens: int = 16
    context_slices: int = 1
    label_map: Optional[Dict[str, int]] = None  # optional case->label

class BraTSDataset(Dataset):
    def __init__(self, cfg: BraTSConfig):
        self.cfg = cfg
        self.cases = self._collect_cases()

    def _collect_cases(self) -> List[str]:
        split_dir = os.path.join(self.cfg.root_dir, self.cfg.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"BraTS split directory not found: {split_dir}")
        return sorted([os.path.join(split_dir, d) for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int):
        case_dir = self.cases[idx]
        paths = {
            "t1": os.path.join(case_dir, "t1.nii.gz"),
            "t1ce": os.path.join(case_dir, "t1ce.nii.gz"),
            "t2": os.path.join(case_dir, "t2.nii.gz"),
            "flair": os.path.join(case_dir, "flair.nii.gz"),
        }
        seg_path = os.path.join(case_dir, "seg.nii.gz")

        volume = preprocess_brats_modalities(paths)               # (C,H,W,D)
        mask = nib.load(seg_path).get_fdata().astype(np.float32)  # (H,W,D)

        valid = extract_valid_slices(mask, min_area=50)
        z = valid[len(valid)//2] if valid else (mask.shape[-1] // 2)

        img = extract_2p5d_slice(volume, z, context=self.cfg.context_slices)  # (C*(2c+1),H,W)
        tokens = extract_tokens(img, token_size=self.cfg.token_size, stride=self.cfg.stride, max_tokens=self.cfg.max_tokens)

        y = 1 if np.sum(mask > 0) > 0 else 0
        if self.cfg.label_map is not None:
            case_id = os.path.basename(case_dir)
            y = int(self.cfg.label_map.get(case_id, y))

        return {"tokens": torch.from_numpy(tokens).float(),
                "y": torch.tensor(y, dtype=torch.long)}
