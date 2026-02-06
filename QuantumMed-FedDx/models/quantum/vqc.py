from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import pennylane as qml
import torch
import torch.nn as nn

from .encoding import EncodingConfig, encode
from .measurement import MeasurementConfig, build_observables

@dataclass
class VQCConfig:
    n_qubits: int = 6
    depth: int = 2
    entanglement: Literal["none", "nn", "ring"] = "ring"
    gate_set: Literal["ry_rz"] = "ry_rz"
    encoding: EncodingConfig = EncodingConfig()
    measurement: MeasurementConfig = MeasurementConfig(basis=("Z",), measure_all_qubits=True)
    shots: Optional[int] = 1024

def _entangle(entanglement: str, n_qubits: int) -> None:
    if entanglement == "none":
        return
    if entanglement == "nn":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return
    if entanglement == "ring":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        if n_qubits > 2:
            qml.CNOT(wires=[n_qubits - 1, 0])
        return
    raise ValueError(f"Unknown entanglement: {entanglement}")

class QFeatEmbedVQC(nn.Module):
    def __init__(self, cfg: VQCConfig):
        super().__init__()
        self.cfg = cfg
        self.theta = nn.Parameter(0.01 * torch.randn(cfg.depth, cfg.n_qubits, 2))
        self.dev = qml.device("default.qubit", wires=cfg.n_qubits, shots=cfg.shots)
        self._cached_obs_key = None
        self._cached_obs = None

        def circuit(features_1d: torch.Tensor, theta: torch.Tensor, d: int):
            encode(features_1d, n_qubits=cfg.n_qubits, cfg=cfg.encoding)
            for l in range(cfg.depth):
                for q in range(cfg.n_qubits):
                    qml.RY(theta[l, q, 0], wires=q)
                    qml.RZ(theta[l, q, 1], wires=q)
                _entangle(cfg.entanglement, cfg.n_qubits)
                if cfg.encoding.encoding_type == "angle_reupload" and l < cfg.depth - 1:
                    encode(features_1d, n_qubits=cfg.n_qubits, cfg=cfg.encoding)
            obs = self._get_observables(d)
            return [qml.expval(o) for o in obs]

        self.qnode = qml.QNode(circuit, self.dev, interface="torch", diff_method="parameter-shift")

    def _get_observables(self, d: int):
        key = (d, tuple(self.cfg.measurement.basis), self.cfg.measurement.measure_all_qubits, self.cfg.n_qubits)
        if self._cached_obs_key == key and self._cached_obs is not None:
            return self._cached_obs
        obs = build_observables(n_qubits=self.cfg.n_qubits, d=d, cfg=self.cfg.measurement)
        self._cached_obs_key = key
        self._cached_obs = obs
        return obs

    def embedding_dim(self, d: int) -> int:
        measured = self.cfg.n_qubits if self.cfg.measurement.measure_all_qubits else d
        return measured * len(self.cfg.measurement.basis)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(f"Expected (B,d), got {tuple(features.shape)}")
        B, d = features.shape
        if d > self.cfg.n_qubits:
            raise ValueError(f"d={d} must be <= n_qubits={self.cfg.n_qubits}")
        outs = []
        for i in range(B):
            expvals = self.qnode(features[i], self.theta, d)
            outs.append(torch.stack(expvals))
        return torch.stack(outs, dim=0)
