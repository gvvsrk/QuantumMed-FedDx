from __future__ import annotations
import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, random_split

from utils.config import load_yaml
from utils.seed import set_seed
from utils.io import save_json

from data.dataset_stub import DatasetStubConfig, TokenDatasetStub
from data.brats_loader import BraTSConfig, BraTSDataset
from data.lidc_loader import LIDCConfig, LIDCDataset

from models.classical.mini_encoder import MiniEncoderConfig
from models.quantum.encoding import EncodingConfig
from models.quantum.measurement import MeasurementConfig
from models.quantum.vqc import VQCConfig
from models.hybrid.vqmednet import VQMedNetConfig, VQMedNet

from training.train import TrainConfig, train_vqmednet
from training.losses import LossConfig
from evaluation.evaluate import EvalConfig, run_full_evaluation
from evaluation.separability import SeparabilityConfig
from evaluation.robustness import RobustnessConfig
from evaluation.calibration import CalibrationConfig

from ablation.nisq_grid import generate_grid, run_ablation_grid, AblationRunConfig


def _resolve_device(s: str) -> str:
    if s == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return s


def build_loaders(cfg: Dict[str, Any]):
    ds = cfg["dataset"]
    dtype = ds.get("type", "stub")
    bs = int(cfg.get("batch_size", 4))

    if dtype == "stub":
        stub = ds["stub"]
        dsc = DatasetStubConfig(
            num_samples=int(stub["num_samples"]),
            num_classes=int(stub["num_classes"]),
            tokens_per_sample=int(stub["tokens_per_sample"]),
            in_channels=int(stub["in_channels"]),
            token_size=int(stub["token_size"]),
        )
        full = TokenDatasetStub(dsc, split="train")
        n = len(full)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        n_test = n - n_train - n_val
        train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(123))
    elif dtype == "brats":
        b = ds["brats"]
        train_ds = BraTSDataset(BraTSConfig(root_dir=b["root_dir"], split="train",
                                            token_size=b["token_size"], stride=b["stride"],
                                            max_tokens=b["max_tokens"], context_slices=b["context_slices"]))
        val_ds = BraTSDataset(BraTSConfig(root_dir=b["root_dir"], split="val",
                                          token_size=b["token_size"], stride=b["stride"],
                                          max_tokens=b["max_tokens"], context_slices=b["context_slices"]))
        test_ds = BraTSDataset(BraTSConfig(root_dir=b["root_dir"], split="test",
                                           token_size=b["token_size"], stride=b["stride"],
                                           max_tokens=b["max_tokens"], context_slices=b["context_slices"]))
    elif dtype == "lidc":
        l = ds["lidc"]
        train_ds = LIDCDataset(LIDCConfig(root_dir=l["root_dir"], split="train",
                                          token_size=l["token_size"], stride=l["stride"],
                                          max_tokens=l["max_tokens"], context_slices=l["context_slices"]))
        val_ds = LIDCDataset(LIDCConfig(root_dir=l["root_dir"], split="val",
                                        token_size=l["token_size"], stride=l["stride"],
                                        max_tokens=l["max_tokens"], context_slices=l["context_slices"]))
        test_ds = LIDCDataset(LIDCConfig(root_dir=l["root_dir"], split="test",
                                         token_size=l["token_size"], stride=l["stride"],
                                         max_tokens=l["max_tokens"], context_slices=l["context_slices"]))
    else:
        raise ValueError(f"Unknown dataset.type: {dtype}")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader


def build_model(cfg: Dict[str, Any]) -> VQMedNet:
    m = cfg["model"]
    q = cfg["quantum"]

    mini_cfg = MiniEncoderConfig(
        encoder_type=m.get("encoder_type", "cnn"),
        in_channels=int(m["in_channels"]),
        token_size=int(m["token_size"]),
        d_out=int(m["d_out"]),
        hidden_dim=64,
        dropout=float(m.get("dropout", 0.1)),
        norm_type="l2",
    )

    vqc_cfg = VQCConfig(
        n_qubits=int(m["q_qubits"]),
        depth=int(q["depth"]),
        entanglement=str(q["entanglement"]),
        encoding=EncodingConfig(encoding_type=str(q["encoding_type"]), use_rz=True),
        measurement=MeasurementConfig(
            basis=tuple(q.get("measure_basis", ["Z"])),
            measure_all_qubits=bool(q.get("measure_all_qubits", True)),
        ),
        shots=q.get("shots", 512),
    )

    net_cfg = VQMedNetConfig(
        num_classes=int(m["num_classes"]),
        q_qubits=int(m["q_qubits"]),
        token_pool="mean",
        head_hidden=int(m.get("head_hidden", 64)),
        dropout=float(m.get("dropout", 0.1)),
    )
    return VQMedNet(net_cfg, mini_cfg, vqc_cfg)


def build_train_cfg(cfg: Dict[str, Any]) -> TrainConfig:
    t = cfg["training"]
    return TrainConfig(
        epochs=int(t["epochs"]),
        lr_classical=float(t["lr_classical"]),
        lr_quantum=float(t["lr_quantum"]),
        weight_decay=float(t["weight_decay"]),
        grad_clip=float(t["grad_clip"]),
        device=_resolve_device(cfg.get("device", "auto")),
        shot_schedule=[(int(a), b) for a, b in t.get("shot_schedule", [])] if t.get("shot_schedule") else None,
        early_stop_patience=int(t["early_stop_patience"]),
    )


def build_loss_cfg(cfg: Dict[str, Any]) -> LossConfig:
    l = cfg["loss"]
    return LossConfig(
        use_class_weights=False,
        class_weights=None,
        use_focal=bool(l.get("use_focal", False)),
        focal_gamma=float(l.get("focal_gamma", 2.0)),
        focal_alpha=None,
        emb_loss_weight=float(l.get("emb_loss_weight", 0.0)),
    )


def build_eval_cfg(cfg: Dict[str, Any]) -> EvalConfig:
    e = cfg["evaluation"]
    return EvalConfig(
        separability=SeparabilityConfig(
            knn_k=int(e["separability"]["knn_k"]),
            reducer=str(e["separability"]["reducer"]),
        ),
        robustness=RobustnessConfig(
            noise_std=float(e["robustness"]["noise_std"]),
            intensity_scale=float(e["robustness"]["intensity_scale"]),
            blur_kernel=int(e["robustness"]["blur_kernel"]),
        ),
        calibration=CalibrationConfig(
            n_bins=int(e["calibration"]["n_bins"])
        )
    )


def main():
    cfg = load_yaml("configs/run.yaml")
    set_seed(int(cfg.get("seed", 42)))
    out_dir = cfg.get("output_dir", "outputs/run1")
    os.makedirs(out_dir, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders(cfg)
    model = build_model(cfg)

    task = cfg.get("task", "train_eval")

    if task == "train_eval":
        train_cfg = build_train_cfg(cfg)
        loss_cfg = build_loss_cfg(cfg)
        eval_cfg = build_eval_cfg(cfg)

        train_summary = train_vqmednet(model, train_loader, val_loader, train_cfg, loss_cfg)
        results = run_full_evaluation(model, train_loader, test_loader, device=train_cfg.device, cfg=eval_cfg)

        save_json(os.path.join(out_dir, "train_summary.json"), train_summary)
        save_json(os.path.join(out_dir, "evaluation.json"), results)
        print("Saved outputs to:", out_dir)

    elif task == "ablation":
        ab = cfg["ablation"]
        grid_cfg = ab["grid"]
        grid = generate_grid(
            q_list=[int(x) for x in grid_cfg["q_list"]],
            depth_list=[int(x) for x in grid_cfg["depth_list"]],
            enc_list=[str(x) for x in grid_cfg["enc_list"]],
            ent_list=[str(x) for x in grid_cfg["ent_list"]],
            shots_list=[None if x is None else int(x) for x in grid_cfg["shots_list"]],
        )

        train_cfg = build_train_cfg(cfg)
        loss_cfg = build_loss_cfg(cfg)
        eval_cfg = build_eval_cfg(cfg)

        run_ablation_grid(
            grid=grid,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=int(cfg["model"]["num_classes"]),
            in_channels=int(cfg["model"]["in_channels"]),
            token_size=int(cfg["model"]["token_size"]),
            d_out=int(cfg["model"]["d_out"]),
            train_cfg=train_cfg,
            loss_cfg=loss_cfg,
            run_cfg=AblationRunConfig(out_jsonl=ab.get("out_jsonl", os.path.join(out_dir, "ablation_results.jsonl")),
                                      eval_cfg=eval_cfg,
                                      perf_key="test_macro_f1")
        )
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
