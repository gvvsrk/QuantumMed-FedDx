from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Literal
import pennylane as qml

@dataclass
class MeasurementConfig:
    basis: Sequence[Literal["Z", "X", "Y"]] = ("Z",)
    measure_all_qubits: bool = True

def _pauli_op(basis: str, wire: int):
    if basis == "Z":
        return qml.PauliZ(wires=wire)
    if basis == "X":
        return qml.PauliX(wires=wire)
    if basis == "Y":
        return qml.PauliY(wires=wire)
    raise ValueError(f"Unknown basis: {basis}")

def build_observables(n_qubits: int, d: int, cfg: MeasurementConfig):
    wires = list(range(n_qubits if cfg.measure_all_qubits else d))
    obs = []
    for b in cfg.basis:
        for w in wires:
            obs.append(_pauli_op(b, w))
    return obs
