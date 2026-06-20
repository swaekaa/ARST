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

**Current Stage:** PHASE 2.6 — BASELINE REPAIR
**Status:** ACTIVE DEVELOPMENT
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

### 4. Run Phase 2 Baseline Training

```bash
# Train individual baselines (Hydra-driven, from repo root)
python train.py model=mlp
python train.py model=cnn
python train.py model=lstm
python train.py model=transformer

# Run full Phase 2.5 benchmark pipeline (trains all 3, generates reports)
.\scripts\run_phase25.ps1              # PowerShell (Windows)
python scripts/run_phase25_full.py     # Python cross-platform

# Or train without Hydra (simpler)
python scripts/train_baseline.py --model cnn
python scripts/train_baseline.py --model lstm
python scripts/train_baseline.py --model transformer
```

### 5. Generate Benchmark Reports

```bash
# After all baselines are trained:
python scripts/generate_benchmark_report.py
```

Outputs: `reports/baseline_benchmark_results.md`, `outputs/benchmarks/*.png`

---



## Current Project Status

**Phase 1**
* Complete

**Phase 2**
* Complete

**Phase 2.5**
* Complete

**Phase 2.6**
* In Progress

**Phase 3**
* Not Started

---

## 📊 Baseline Benchmark Results (Phase 2.5)

> All models trained with identical experimental conditions (seed=42, AdamW, Focal Loss, 100 epochs).

| Model | Accuracy | Macro F1 |
| ----------- | -------: | -------: |
| Random      |   0.7850 |   0.2199 |
| Majority    |   0.7850 |   0.2199 |
| MLP         |   0.6410 |   0.3179 |
| CNN         |   0.1807 |   0.1894 |
| LSTM        |   0.0482 |   0.0390 |
| Transformer |   0.2527 |   0.2007 |

**Explanations:**
* **Macro F1 is the primary metric.**
* **Majority accuracy is high because the dataset is highly imbalanced.**
* **MLP is currently the strongest validated baseline.**

---

## Known Issues and Ongoing Investigation

### Severe Class Imbalance
* Majority class accounts for ~78.5% of samples.
* Accuracy alone is misleading.
* Macro F1 is used as the principal metric.

### CNN Underperformance
**Current:**
Macro F1 ≈ 0.189

**Expected:**
Should exceed MLP.

**Status:**
Under investigation.

### LSTM Failure
**Current:**
Macro F1 ≈ 0.039

This is significantly worse than expected.

**Possible causes:**
* sequence alignment
* normalization
* hidden-state handling
* class imbalance effects

**Status:**
Under investigation.

### Transformer Underperformance
**Current:**
Macro F1 ≈ 0.201

**Expected:**
Comparable to LSTM.

**Status:**
Under investigation.

### Phase 2.6
**Goal:**
Repair and validate CNN, LSTM and Transformer baselines before beginning Phase 3.

---

## Research Notes

Tiny-overfit tests succeeded.

Therefore:
* Training infrastructure is correct.
* Dataloaders are functional.
* Architectures are expressive enough.

The poor benchmark results indicate an optimization or data-related issue rather than a capacity issue.

---

> Full benchmark: [`reports/baseline_benchmark_results.md`](reports/baseline_benchmark_results.md)
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
| 2 | Baseline Models (infrastructure) | ✅ Complete |
| 2.5 | Baseline Benchmark Validation | ✅ Complete |
| 2.6 | Architecture Validation and Baseline Repair | 🔄 IN PROGRESS |
| 3 | Modality-Specific Encoders | 🔲 NOT STARTED |
| 4 | Adaptive Reliability Module | 🔲 |
| 5 | Adaptive Fusion Transformer | 🔲 |

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
