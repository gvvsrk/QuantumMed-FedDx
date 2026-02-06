from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

@dataclass
class ROIBox:
    """ROI bounding box in (x, y, w, h) pixel coordinates."""
    x: int
    y: int
    w: int
    h: int

@dataclass
class TokenizationConfig:
    patch_size: int = 32
    stride: int = 32
    use_overlapping: bool = False
    flatten_tokens: bool = False
    roi_output_size: int = 64
    pad_if_needed: bool = True

def _ensure_chw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        if x.shape[0] in (1, 3, 4) and x.shape[1] > 8 and x.shape[2] > 8:
            return x
        return x.permute(2, 0, 1).contiguous()
    raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")

def _pad_to_grid(x_chw: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    _, H, W = x_chw.shape

    def _needed_pad(L: int) -> int:
        if L < patch_size:
            return patch_size - L
        steps = (L - patch_size) // stride + 1
        last_start = (steps - 1) * stride
        end = last_start + patch_size
        return max(0, end - L)

    pad_h = _needed_pad(H)
    pad_w = _needed_pad(W)
    if pad_h == 0 and pad_w == 0:
        return x_chw
    return F.pad(x_chw, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

def extract_patch_tokens(x: torch.Tensor, cfg: TokenizationConfig) -> torch.Tensor:
    x_chw = _ensure_chw(x)
    p = int(cfg.patch_size)
    s = int(cfg.stride if cfg.use_overlapping else cfg.patch_size)
    if cfg.pad_if_needed:
        x_chw = _pad_to_grid(x_chw, patch_size=p, stride=s)

    C, H, W = x_chw.shape
    if H < p or W < p:
        raise ValueError(f"Image too small for patch_size={p}, got H={H}, W={W}")

    patches = x_chw.unfold(1, p, s).unfold(2, p, s)             # (C,nH,nW,p,p)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()        # (nH,nW,C,p,p)
    nH, nW = patches.shape[0], patches.shape[1]
    tokens = patches.view(nH * nW, C, p, p)

    if cfg.flatten_tokens:
        tokens = tokens.view(tokens.size(0), -1)
    return tokens

def extract_roi_tokens(x: torch.Tensor, rois: Sequence[ROIBox], cfg: TokenizationConfig) -> torch.Tensor:
    x_chw = _ensure_chw(x)
    C, H, W = x_chw.shape
    out_s = int(cfg.roi_output_size)

    tokens: List[torch.Tensor] = []
    for b in rois:
        x0, y0 = int(b.x), int(b.y)
        x1, y1 = x0 + int(b.w), y0 + int(b.h)
        x0 = max(0, min(x0, W - 1))
        y0 = max(0, min(y0, H - 1))
        x1 = max(x0 + 1, min(x1, W))
        y1 = max(y0 + 1, min(y1, H))
        crop = x_chw[:, y0:y1, x0:x1].unsqueeze(0)  # (1,C,h,w)
        crop_resized = F.interpolate(crop, size=(out_s, out_s), mode="bilinear", align_corners=False)
        tokens.append(crop_resized.squeeze(0))

    if len(tokens) == 0:
        empty = torch.empty((0, C, out_s, out_s), dtype=x_chw.dtype, device=x_chw.device)
        return empty.view(0, -1) if cfg.flatten_tokens else empty

    out = torch.stack(tokens, dim=0)
    if cfg.flatten_tokens:
        out = out.view(out.size(0), -1)
    return out

def tokenize_image(x: torch.Tensor, cfg: TokenizationConfig, rois: Optional[Sequence[ROIBox]] = None) -> torch.Tensor:
    if rois is not None and len(rois) > 0:
        return extract_roi_tokens(x, rois=rois, cfg=cfg)
    return extract_patch_tokens(x, cfg=cfg)
