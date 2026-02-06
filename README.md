# QuantumMed-FedDx
**Quantum-Enhanced Hybrid Deep Learning for Medical Image Diagnosis**

## Overview
**QuantumMed-FedDx** is a modular research codebase implementing a hybrid quantum–classical deep learning framework for MRI- and CT-based disease diagnosis. The system integrates **variational quantum circuits (VQCs)** with lightweight classical encoders to perform quantum feature embedding under **NISQ-era constraints**, enabling systematic evaluation of potential quantum advantages in medical imaging.

The current implementation corresponds to the **non-federated configuration** of the framework, focusing on quantum-enhanced representation learning and diagnostic performance analysis for:

- Brain tumor diagnosis from MRI (BraTS)
- Lung nodule diagnosis from CT (LIDC-IDRI)

---

## Key Features
- Hybrid CNN / ViT stem combined with Variational Quantum Circuits (VQC)
- Token-based patch and ROI image representation
- NISQ-aware quantum circuit design (bounded qubits, depth, shots)
- End-to-end differentiable quantum–classical training
- Comprehensive evaluation framework:
  - Classification performance
  - Embedding separability
  - Robustness to noise and domain shift
  - Calibration reliability
  - Quantum circuit complexity
- Automated NISQ feasibility ablation and Pareto analysis

---



## Repository Structure

QuantumMed-FedDx/
│
├── ablation/               # NISQ ablation grid and Pareto analysis
├── configs/                # YAML configuration files
├── data/                   # Dataset loaders (BraTS, LIDC-IDRI, stubs)
├── evaluation/             # Separability, robustness, calibration, complexity
├── models/
│   ├── classical/          # Mini CNN / ViT feature encoder
│   ├── quantum/            # VQC, encoding, measurement modules
│   └── hybrid/             # VQMedNet hybrid architecture
├── preprocessing/          # MRI/CT preprocessing and tokenization
├── training/               # Losses and training loops
├── utils/                  # Configuration, seeding, I/O utilities
│
├── main.py                 # Entry point for training / evaluation / ablation
├── requirements.txt
└── README.md


---

## Supported Datasets

### BraTS (Brain Tumor MRI)
Expected directory structure:


data/BraTS/{train,val,test}/{CASE_ID}/
├── t1.nii.gz
├── t1ce.nii.gz
├── t2.nii.gz
├── flair.nii.gz
└── seg.nii.gz


### LIDC-IDRI (Lung CT)
Prepared per-case structure:


data/LIDC-IDRI/{train,val,test}/{CASE_ID}/
├── ct.nii.gz
└── mask.nii.gz


**Note:** Dataset preparation scripts are not included. Users must ensure datasets are preprocessed and split consistently with the experimental protocol.

---

## Installation
Install all dependencies using:
```bash
pip install -r requirements.txt

Core Dependencies

PyTorch

PennyLane

NumPy

scikit-learn

nibabel

PyYAML

Configuration

All experiments are controlled via:

configs/run.yaml


Key configurable components include:

Dataset selection (stub / BraTS / LIDC)

Token size and number of tokens

Quantum circuit parameters (qubits, depth, encoding, entanglement, shots)

Training and optimization hyperparameters

Evaluation and robustness settings

NISQ ablation grids

Running Experiments
Train and Evaluate
python main.py


Outputs are saved to:

outputs/<run_name>/
  ├── train_summary.json
  └── evaluation.json

NISQ Ablation Study

Set the following in configs/run.yaml:

task: ablation


Then run:

python main.py


Results are logged as:

ablation_results.jsonl

Research Scope

This implementation is designed for:

Studying quantum feature embeddings in medical imaging

Evaluating performance–complexity trade-offs under NISQ constraints

Benchmarking against strong classical baselines

Supporting SCI / SCOPUS-standard experimental rigor

Federated learning extensions are planned in subsequent modules.

Reproducibility

Deterministic random seeds supported

Explicit circuit parameterization

Fully config-driven experiments

Clear dataset expectations

Complete ablation and evaluation logging

License

This project is intended for academic and research use.
Please cite appropriately if used in publications.

Contact

For research collaboration, extensions, or clarification, please contact the authors via the corresponding publication.


---

This version will:
- Render perfectly on GitHub
- Pass reviewer scrutiny
- Look professional for **SCI / SCOPUS submissions**
- Be future-proof for federated extensions

If you want, next I can:
- Add a **BibTeX citation block**
- Convert this to **IEEE-style README**
- Add **GitHub badges**
- Split into `README.md` + `docs/EXPERIMENTS.md`

Just say which one.
