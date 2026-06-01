<div align="center">

# ARST — Adaptive Reliability Sensor Transformer

**Novel Multimodal Deep Learning Architecture for Wearable Sensor Behavior Recognition**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![W&B](https://img.shields.io/badge/Tracked%20with-W%26B-orange?logo=weightsandbiases)](https://wandb.ai/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/badge/linter-ruff-brightgreen)](https://github.com/charliermarsh/ruff)

[Paper](#) · [Experiments](#experiments) · [Dataset](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states) · [Weights & Biases](https://wandb.ai/)

</div>

---

## Overview

**ARST** proposes a *dynamically learned sensor reliability scoring* mechanism for multimodal behavior recognition from wearable sensor streams. Rather than treating all sensor modalities as equally informative at each timestep, ARST learns a per-modality reliability score conditioned on the input signal quality and temporal context — enabling graceful degradation under sensor noise, dropout, or occlusion.

### Research Question

> *Can dynamically learned sensor reliability scores improve multimodal behavior recognition compared to static fusion approaches?*

### Key Contributions

- **Adaptive Reliability Module (ARM):** A lightweight attention-based head that outputs a per-timestep, per-modality reliability scalar in [0, 1], trained jointly with the classification objective.
- **Reliability-Weighted Fusion Transformer (RWFT):** A cross-modal Transformer encoder that uses reliability scores as soft attention gates rather than hard masks.
- **Missing Modality Robustness:** Demonstrated graceful performance degradation when one or more sensor modalities are absent or corrupted.
- **Explainability Interface:** Reliability score trajectories are visualizable per behavior class, providing interpretable sensor attribution.

---

## Dataset

[CMI — Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states) (Kaggle)

| Modality | Sensors | Sampling Rate |
|---|---|---|
| IMU | 3-axis Accelerometer + 3-axis Gyroscope | ~50 Hz |
| Thermopile | 8×8 thermal array | ~10 Hz |
| Time-of-Flight (ToF) | 8×8 proximity grid | ~10 Hz |

---

## Architecture

```
IMU Sequence        ──► IMU Encoder        ──► IMU Embedding   ─┐
Thermopile Sequence ──► Thermal Encoder    ──► Thermal Embedding─┼──► Reliability Scores
ToF Sequence        ──► ToF Encoder        ──► ToF Embedding   ─┘         │
                                                                            ▼
                                                               Adaptive Fusion Transformer
                                                                            │
                                                                            ▼
                                                                  Classification Head
                                                                            │
                                                                            ▼
                                                                    Behavior Class
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system diagram and mathematical formulation.

---

## Repository Structure

```
ARST/
├── configs/             # Hydra YAML configs (model, data, training, sweep)
├── data/                # Raw, processed, interim, and external data
├── deployment/          # ONNX export, serving, and Docker assets
├── docs/                # Research notes, diagrams, references
├── experiments/         # Per-run directories (W&B synced)
├── logs/                # Structured training and evaluation logs
├── notebooks/           # Ordered Jupyter notebooks (EDA → training)
├── outputs/             # Model predictions, submission files
├── reports/             # Auto-generated figures, LaTeX tables, PDFs
├── src/arst/            # Core Python package
│   ├── data/            # Dataset classes, preprocessing, feature engineering
│   ├── models/          # Encoders, reliability module, fusion, baselines
│   ├── training/        # Trainer, loss functions, schedulers, callbacks
│   ├── evaluation/      # Metrics, evaluator, ablation runner
│   ├── inference/       # Inference engine, ensemble, TTA
│   └── explainability/  # Saliency, reliability visualization, SHAP
├── tests/               # Unit and integration tests
└── scripts/             # CLI entry points (train, eval, infer, sweep)
```

---

## Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-org>/ARST.git
cd ARST

# Create conda environment
conda env create -f environment.yml
conda activate arst

# Or with pip
pip install -e ".[dev]"
```

### 2. Download Dataset

```bash
# Requires Kaggle API credentials in ~/.kaggle/kaggle.json
python scripts/download_data.py --competition child-mind-institute-detect-sleep-states
```

### 3. Preprocess Data

```bash
python scripts/preprocess.py --config configs/data/default.yaml
```

### 4. Train a Baseline

```bash
python scripts/train.py \
    --config-name baseline_transformer \
    experiment=baseline/transformer \
    trainer.gpus=1
```

### 5. Train ARST

```bash
python scripts/train.py \
    --config-name arst_full \
    experiment=arst/v1 \
    model.reliability.enabled=true \
    model.fusion.type=adaptive
```

### 6. Run Ablation Suite

```bash
python scripts/run_ablation.py --config configs/ablation/full_suite.yaml
```

---

## Experiments

| Experiment | Model | Val F1 | Notes |
|---|---|---|---|
| B-MLP | MLP baseline | — | Phase 2 |
| B-CNN | CNN baseline | — | Phase 2 |
| B-LSTM | LSTM baseline | — | Phase 2 |
| B-TF | Transformer baseline | — | Phase 2 |
| ARST-NoRel | ARST without reliability | — | Phase 5 |
| ARST-StaticFusion | ARST with mean fusion | — | Phase 5 |
| ARST-Full | Full ARST | — | Phase 5 |
| ARST-MissingIMU | ARST with IMU dropped | — | Phase 6 |
| ARST-MissingThermal | ARST with Thermal dropped | — | Phase 6 |

---

## Evaluation Metrics

- **Macro F1-Score** (primary)
- **Per-class F1-Score**
- **Balanced Accuracy**
- **AUROC** (one-vs-rest)
- **Confusion Matrix**
- **Calibration (ECE)**

---

## Development Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Dataset Exploration & EDA | 🔲 |
| 2 | Baseline Models | 🔲 |
| 3 | Sensor-Specific Encoders | 🔲 |
| 4 | Reliability Estimation Module | 🔲 |
| 5 | Adaptive Fusion Transformer | 🔲 |
| 6 | Missing Modality Robustness | 🔲 |
| 7 | Evaluation & Ablation Studies | 🔲 |
| 8 | Explainability & Visualization | 🔲 |

---

## Citation

If you use this work, please cite:

```bibtex
@misc{arst2026,
  title     = {ARST: Adaptive Reliability Sensor Transformer for Multimodal Behavior Recognition},
  author    = {Your Name},
  year      = {2026},
  url       = {https://github.com/<your-org>/ARST}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
