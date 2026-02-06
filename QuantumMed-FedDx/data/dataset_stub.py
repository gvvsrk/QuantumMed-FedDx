from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
from torch.utils.data import Dataset

@dataclass
class DatasetStubConfig:
    num_samples: int = 200
    num_classes: int = 2
    tokens_per_sample: int = 16
    in_channels: int = 4
    token_size: int = 32

class TokenDatasetStub(Dataset):
    def __init__(self, cfg: DatasetStubConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self._seed_shift = {"train": 0, "val": 1, "test": 2}.get(split, 0)

    def __len__(self):
        return self.cfg.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(idx + 10_000 * self._seed_shift)
        T, C, S = self.cfg.tokens_per_sample, self.cfg.in_channels, self.cfg.token_size
        tokens = torch.randn(T, C, S, S, generator=g)
        y = torch.randint(0, self.cfg.num_classes, (1,), generator=g).squeeze(0)
        return {"tokens": tokens, "y": y}
