from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ComplexityReport:
    n_qubits: int
    depth: int
    entanglement: str
    encoding_type: str
    measure_basis: str
    measure_all_qubits: bool
    shots: Optional[int]
    n_trainable_params: int
    approx_gate_count: int

def _approx_gate_count(n_qubits: int, depth: int, entanglement: str, encoding_use_rz: bool, reupload: bool) -> int:
    enc_per_qubit = 1 + (1 if encoding_use_rz else 0)
    enc_gates = n_qubits * enc_per_qubit
    rot_gates = depth * (2 * n_qubits)
    if entanglement == "none":
        ent_gates = 0
    elif entanglement == "nn":
        ent_gates = depth * (n_qubits - 1)
    elif entanglement == "ring":
        ent_gates = depth * (n_qubits if n_qubits > 2 else (n_qubits - 1))
    else:
        ent_gates = 0
    extra_enc = (depth - 1) * enc_gates if reupload and depth > 1 else 0
    return int(enc_gates + rot_gates + ent_gates + extra_enc)

def quantum_complexity(model) -> Dict[str, object]:
    if not hasattr(model, "vqc") or not hasattr(model.vqc, "cfg"):
        return {}
    cfg = model.vqc.cfg
    n_qubits = int(cfg.n_qubits)
    depth = int(cfg.depth)
    entanglement = str(cfg.entanglement)
    encoding_type = str(cfg.encoding.encoding_type)
    use_rz = bool(cfg.encoding.use_rz)
    reupload = encoding_type == "angle_reupload"
    basis = ",".join(cfg.measurement.basis)
    shots = cfg.shots
    n_trainable = int(model.vqc.theta.numel()) if hasattr(model.vqc, "theta") else 0
    gate_count = _approx_gate_count(n_qubits, depth, entanglement, use_rz, reupload)
    rep = ComplexityReport(
        n_qubits=n_qubits, depth=depth, entanglement=entanglement,
        encoding_type=encoding_type, measure_basis=basis,
        measure_all_qubits=bool(cfg.measurement.measure_all_qubits),
        shots=shots, n_trainable_params=n_trainable, approx_gate_count=gate_count
    )
    return rep.__dict__
