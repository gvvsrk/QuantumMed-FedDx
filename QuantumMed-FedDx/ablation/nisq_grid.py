from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional, Tuple
import json
import time

from models.classical.mini_encoder import MiniEncoderConfig
from models.quantum.encoding import EncodingConfig
from models.quantum.measurement import MeasurementConfig
from models.quantum.vqc import VQCConfig
from models.hybrid.vqmednet import VQMedNetConfig, VQMedNet

from training.train import TrainConfig, train_vqmednet
from training.losses import LossConfig
from evaluation.evaluate import EvalConfig, run_full_evaluation

@dataclass(frozen=True)
class AblationPoint:
    q: int
    depth: int
    encoding_type: Literal["angle", "angle_reupload"]
    entanglement: Literal["none", "nn", "ring"]
    shots: Optional[int]

@dataclass
class AblationRunConfig:
    out_jsonl: str = "ablation_results.jsonl"
    eval_cfg: EvalConfig = EvalConfig()
    perf_key: str = "test_macro_f1"

def build_model_for_point(point: AblationPoint, num_classes: int, in_channels: int, token_size: int, d_out: int,
                          head_hidden: int = 64, dropout: float = 0.1,
                          measure_basis: Tuple[str, ...] = ("Z",), measure_all_qubits: bool = True) -> VQMedNet:
    if d_out > point.q:
        raise ValueError(f"d_out={d_out} must be <= q={point.q}")
    mini_cfg = MiniEncoderConfig(encoder_type="cnn", in_channels=in_channels, token_size=token_size, d_out=d_out,
                                 hidden_dim=64, dropout=dropout, norm_type="l2")
    vqc_cfg = VQCConfig(n_qubits=point.q, depth=point.depth, entanglement=point.entanglement,
                        encoding=EncodingConfig(encoding_type=point.encoding_type, use_rz=True),
                        measurement=MeasurementConfig(basis=measure_basis, measure_all_qubits=measure_all_qubits),
                        shots=point.shots)
    net_cfg = VQMedNetConfig(num_classes=num_classes, q_qubits=point.q, token_pool="mean",
                             head_hidden=head_hidden, dropout=dropout)
    return VQMedNet(net_cfg, mini_cfg, vqc_cfg)

def generate_grid(q_list: List[int], depth_list: List[int], enc_list: List[Literal["angle","angle_reupload"]],
                  ent_list: List[Literal["none","nn","ring"]], shots_list: List[Optional[int]]) -> List[AblationPoint]:
    grid = []
    for q in q_list:
        for d in depth_list:
            for e in enc_list:
                for t in ent_list:
                    for s in shots_list:
                        grid.append(AblationPoint(q=q, depth=d, encoding_type=e, entanglement=t, shots=s))
    return grid

def run_ablation_grid(grid: List[AblationPoint], train_loader, val_loader, test_loader,
                      num_classes: int, in_channels: int, token_size: int, d_out: int,
                      train_cfg: TrainConfig, loss_cfg: LossConfig, run_cfg: AblationRunConfig) -> None:
    device = train_cfg.device
    with open(run_cfg.out_jsonl, "w", encoding="utf-8") as f:
        for idx, point in enumerate(grid, start=1):
            t0 = time.time()
            model = build_model_for_point(point, num_classes, in_channels, token_size, d_out)
            train_summary = train_vqmednet(model, train_loader, val_loader, train_cfg, loss_cfg)
            eval_out = run_full_evaluation(model, train_loader, test_loader, device=device, cfg=run_cfg.eval_cfg)
            elapsed = time.time() - t0
            record = {"run_id": idx, "ablation_point": asdict(point), "train_summary": train_summary,
                      "evaluation": eval_out, "elapsed_sec": elapsed}
            f.write(json.dumps(record) + "\n"); f.flush()
            print(f"[{idx}/{len(grid)}] point={point} {run_cfg.perf_key}={eval_out.get(run_cfg.perf_key)} elapsed={elapsed:.1f}s")
