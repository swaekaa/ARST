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

> **Phase 1 verified schema** — values below are confirmed from `data/raw/train.csv` (574,945 rows).

| Modality | Sensors | Channels | Notes |
|---|---|---|---|
| IMU | 3-axis Accelerometer + Quaternion | **7** (`acc_x/y/z`, `rot_w/x/y/z`) | Not gyroscope; quaternion orientation |
| Thermopile | 5 infrared channels | **5** (`thm_1`…`thm_5`) | Linear array, not 8×8 grid |
| Time-of-Flight (ToF) | 5 sensors × 64 pixels | **320** (`tof_1_v0`…`tof_5_v63`) | ~59% readings invalid (encoded as −1.0) |

**Dataset statistics:**
- **574,945** total timestep rows (flat CSV — _not_ per-sequence parquet)
- **8,151** unique sequences across multiple subjects
- **4** behavior classes (see below)
- Overall missing rate: ~1.8% NaN + ~59% ToF sentinel invalidity

**Behavior classes:**

| Index | Class |
|---|---|
| 0 | Hand at target location |
| 1 | Moves hand to target location |
| 2 | Performs gesture |
| 3 | Relaxes and moves hand to target location |

---

## Architecture

```
IMU Sequence   [B,T,7]  ──► IMU Encoder   ──► IMU Embedding   [B,T,D] ─┐
Thermal Seq.   [B,T,5]  ──► Therm Encoder ──► Therm Embedding [B,T,D] ─┼──► Reliability Scores
ToF Sequence   [B,T,320]──► ToF Encoder   ──► ToF Embedding   [B,T,D] ─┘         │
               + mask[B,T,320]                (mask used inside encoder)           ▼
                                                                   Adaptive Fusion Transformer
                                                                                   │
                                                                                   ▼
                                                                         Classification Head
                                                                                   │
                                                                                   ▼
                                                                     Behavior Class (C=4)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system diagram and mathematical formulation.

---

## Phase 1 Findings & Architectural Implications

> Phase 1 EDA (see `reports/phase1_summary.md`) revealed several discrepancies between the initial architecture assumptions and the actual dataset schema.

| Property | Initially Assumed | **Actual (Phase 1 Confirmed)** | Impact |
|---|---|---|---|
| Storage format | Per-sequence `.parquet` files | **Single flat CSV (1.1 GB)** | Chunked reading + HDF5 conversion required |
| IMU channels | acc + gyroscope (6) | **acc + quaternion = 7** | IMU encoder input dim: 6 → **7** |
| Thermopile | 8×8 grid = 64 channels | **5 linear channels** | Thermal encoder drastically simplified |
| ToF | Single 8×8 map = 64 channels | **5 sensors × 64 = 320 channels** | ToF encoder input dim: 64 → **320** |
| ToF invalidity | Unknown | **~59% avg (encoded as −1.0)** | Mask channel is mandatory; primary ARM motivation |
| Behavior classes | Unknown | **4 classes** | Classification head output dim = 4 |
| Total sequences | Unknown | **8,151** | Window strategy: T=128, 50% overlap |

**Architectural changes required by Phase 1:**
1. IMU encoder: `[B,T,6]` → `[B,T,7]` input projection
2. Thermal encoder: `[B,T,64]` → `[B,T,5]` — simplified from spatial to linear
3. ToF encoder: `[B,T,64]` → `[B,T,320]` + explicit mask channel `[B,T,320]`
4. Classification head output: `C` → `4`
5. Data pipeline: CSV-first, no parquet loading logic required

---

## Repository Structure

```
ARST/
├── configs/             # Hydra YAML configs (model, data, training, sweep)
│   └── sensor_groups.yaml   # Phase 1 verified sensor column assignments
├── data/                # Raw, processed, interim, and external data
├── deployment/          # ONNX export, serving, and Docker assets
├── docs/                # Research notes, diagrams, references
├── experiments/         # Per-run directories (W&B synced)
├── logs/                # Structured training and evaluation logs
├── notebooks/           # Ordered Jupyter notebooks (EDA → training)
│   └── phase1_dataset_exploration.ipynb  # Phase 1 EDA
├── outputs/             # Model predictions, EDA figures, submission files
│   └── eda/             # Phase 1 plots and CSVs
├── reports/             # Auto-generated figures, LaTeX tables, PDFs
│   ├── dataset_inventory.md
│   ├── dataset_profile.md
│   ├── class_analysis.md
│   ├── sequence_analysis.md
│   ├── missing_data_analysis.md
│   ├── reliability_motivation.md
│   ├── preprocessing_recommendations.md
│   └── phase1_summary.md
├── src/arst/            # Core Python package
│   ├── data/            # Dataset classes, preprocessing, feature engineering
│   ├── models/          # Encoders, reliability module, fusion, baselines
│   ├── training/        # Trainer, loss functions, schedulers, callbacks
│   ├── evaluation/      # Metrics, evaluator, ablation runner
│   ├── inference/       # Inference engine, ensemble, TTA
│   └── explainability/  # Saliency, reliability visualization, SHAP
├── tests/               # Unit and integration tests
└── scripts/             # CLI entry points (train, eval, infer, sweep)
    ├── phase1_eda.py    # Phase 1 EDA script (run to regenerate reports)
    └── test_phase1.py   # Dataloader smoke test
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

### 3. Run Phase 1 EDA (already complete)

```bash
# Regenerate all Phase 1 reports and figures
python scripts/phase1_eda.py

# Verify dataloader batch shapes
python scripts/test_phase1.py
```

### 4. Preprocess Data

```bash
python scripts/preprocess.py --config configs/data/default.yaml
```

### 5. Train a Baseline

```bash
python scripts/train.py \
    --config-name baseline_transformer \
    experiment=baseline/transformer \
    trainer.gpus=1
```

### 6. Train ARST

```bash
python scripts/train.py \
    --config-name arst_full \
    experiment=arst/v1 \
    model.reliability.enabled=true \
    model.fusion.type=adaptive
```

### 7. Run Ablation Suite

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
| 1 | Dataset Exploration & EDA | ✅ Complete |
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
