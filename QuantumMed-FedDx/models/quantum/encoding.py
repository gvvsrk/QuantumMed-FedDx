from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import pennylane as qml
import torch

@dataclass
class EncodingConfig:
    encoding_type: Literal["angle", "angle_reupload"] = "angle"
    use_rz: bool = True
    scale: float = 3.141592653589793

def angle_encoding(features: torch.Tensor, n_qubits: int, cfg: EncodingConfig) -> None:
    d = int(features.shape[0])
    for k in range(d):
        angle = cfg.scale * features[k]
        qml.RY(angle, wires=k)
        if cfg.use_rz:
            qml.RZ(angle, wires=k)

def encode(features: torch.Tensor, n_qubits: int, cfg: EncodingConfig) -> None:
    if cfg.encoding_type in ("angle", "angle_reupload"):
        angle_encoding(features, n_qubits=n_qubits, cfg=cfg)
    else:
        raise ValueError(f"Unsupported encoding_type: {cfg.encoding_type}")
