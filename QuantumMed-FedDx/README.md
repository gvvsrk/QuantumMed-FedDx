# QuantumMed-FedDx (Objective-1 codebase)

This repository contains a modular implementation of:
- Patch/token extraction + classical feature compression (MiniEncoder)
- QFeatEmbed-VQC (variational quantum embedding with NISQ constraints)
- Hybrid model (VQMedNet) and training loop
- Evaluation: separability, robustness, calibration, complexity
- NISQ ablation grid + Pareto trade-off

## Install
pip install -r requirements.txt

## Configure
Edit `configs/run.yaml`:
- dataset.type: stub | brats | lidc
- quantum: q_qubits, depth, encoding_type, entanglement, shots
- training/evaluation/ablation params

## Run
python main.py

## Dataset folder expectations
### BraTS
data/BraTS/{train,val,test}/{CASE}/
  t1.nii.gz, t1ce.nii.gz, t2.nii.gz, flair.nii.gz, seg.nii.gz

### LIDC (prepared)
data/LIDC-IDRI/{train,val,test}/{CASE}/
  ct.nii.gz, mask.nii.gz
