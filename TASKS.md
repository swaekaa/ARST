# ARST — Project Task Tracker

> **Adaptive Reliability Sensor Transformer**
> Living task list — update as work progresses

---

## Status Key
- `[ ]` Not started
- `[/]` In progress
- `[x]` Complete
- `[!]` Blocked

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MILESTONE 1 — Repository & Environment Setup
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Repository Structure
- [x] Design complete folder tree
- [x] Create all source directories
- [x] Create all data directories
- [x] Create all experiment directories
- [x] Create placeholder `__init__.py` files

### Documentation
- [x] Write `README.md`
- [x] Write `CONTEXT.md`
- [x] Write `ARCHITECTURE.md`
- [x] Write `DEVELOPMENT_ROADMAP.md`
- [x] Write `TASKS.md`

### Configuration System
- [x] Design Hydra config directory structure
- [x] Write `configs/data/default.yaml`
- [x] Write `configs/model/arst_full.yaml`
- [x] Write `configs/model/baseline_transformer.yaml`
- [x] Write `configs/training/default.yaml`
- [x] Write `configs/experiment/` examples

### Environment
- [x] Write `environment.yml`
- [x] Write `pyproject.toml`
- [x] Write `.gitignore`
- [x] Write `.pre-commit-config.yaml`
- [ ] Test environment creation on target machine
- [ ] Verify CUDA availability (`torch.cuda.is_available()`)
- [ ] Run `python scripts/verify_environment.py`

### W&B Integration
- [ ] Set up W&B project: `arst-behavior-recognition`
- [ ] Verify W&B API key configuration
- [ ] Test W&B run creation

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MILESTONE 2 — Phase 1: Dataset EDA
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Data Download
- [ ] Configure Kaggle API credentials
- [ ] Run `python scripts/download_data.py`
- [ ] Verify all parquet files downloaded
- [ ] Check total dataset size

### Data Loading Pipeline
- [ ] Implement `src/arst/data/raw_loader.py`
- [ ] Implement `src/arst/data/preprocessor.py` (skeleton)
- [ ] Test: load one sequence successfully
- [ ] Test: handle missing modality file gracefully

### EDA Notebooks
- [ ] `notebooks/01_data_loading.ipynb` — structure audit
- [ ] `notebooks/02_imu_eda.ipynb` — signal plots, FFT, autocorrelation
- [ ] `notebooks/03_thermal_eda.ipynb` — thermal maps, dead pixel analysis
- [ ] `notebooks/04_tof_eda.ipynb` — depth maps, invalid reading analysis
- [ ] `notebooks/05_class_imbalance_analysis.ipynb`
- [ ] `notebooks/06_sequence_length_analysis.ipynb`
- [ ] `notebooks/07_missing_data_analysis.ipynb`

### Preprocessing Decisions
- [ ] Determine optimal window size (T)
- [ ] Choose normalization strategy
- [ ] Design missing value handling
- [ ] Define train/val/test split

### Deliverable
- [ ] EDA summary report: `reports/phase1_eda_report.md`
- [ ] Preprocessing config finalized: `configs/data/default.yaml`

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MILESTONE 3 — Data Pipeline
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Preprocessing
- [ ] Implement `src/arst/data/preprocessor.py`
  - [ ] IMU: bandpass filter, gravity removal, normalization
  - [ ] Thermal: dead pixel fix, background subtraction, normalization
  - [ ] ToF: invalid masking, inpainting, normalization
- [ ] Run preprocessing: `python scripts/preprocess.py`
- [ ] Verify processed data shape and statistics

### Feature Engineering
- [ ] Implement `src/arst/data/feature_engineer.py`
  - [ ] IMU: ENMO, jerk, tilt angles
  - [ ] Thermal: centroid, spread, entropy
  - [ ] ToF: depth gradients, motion flow

### Dataset Class
- [ ] Implement `src/arst/data/dataset.py`
  - [ ] `ARSTDataset(Dataset)`: load windowed samples
  - [ ] Window slicing with configurable overlap
  - [ ] Online augmentation interface
  - [ ] Variable-length handling and padding

### Augmentation
- [ ] Implement `src/arst/data/augmentation.py`
  - [ ] `RandomTimeWarp`
  - [ ] `RandomChannelNoise`
  - [ ] `RandomModalityDrop`
  - [ ] `RandomTimeShift`
  - [ ] `MagnitudeScaling`

### DataModule
- [ ] Implement `src/arst/data/datamodule.py`
  - [ ] `ARSTDataModule`: train/val/test dataloaders
  - [ ] Stratified sampling support
  - [ ] Class weight computation
- [ ] Unit tests: `tests/test_dataset.py`

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MILESTONE 4 — Phase 2: Baseline Models
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### MLP Baseline
- [x] Implement `src/arst/models/baselines/mlp.py`
- [x] Config: `configs/model/baseline_mlp.yaml`
- [ ] Train: `python scripts/train.py --config-name baseline_mlp`
- [ ] Evaluate and log to W&B

### CNN Baseline
- [x] Implement `src/arst/models/baselines/cnn.py`
- [x] Config: `configs/model/baseline_cnn.yaml`
- [ ] Train and evaluate

### LSTM Baseline
- [x] Implement `src/arst/models/baselines/lstm.py`
- [x] Config: `configs/model/baseline_lstm.yaml`
- [ ] Train and evaluate

### Transformer Baseline
- [x] Implement `src/arst/models/baselines/transformer.py`
- [x] Config: `configs/model/baseline_transformer.yaml`
- [ ] Train and evaluate

### Baseline Comparison
- [ ] `notebooks/12_baseline_comparison.ipynb`
- [ ] Leaderboard table: F1, Balanced Acc, AUROC, Latency
- [x] Report: `reports/baseline_benchmark_template.md`

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MILESTONE 5 — Phase 3: Sensor Encoders
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### IMU Encoder
- [ ] Implement `src/arst/models/encoders/imu_encoder.py`
- [ ] Unimodal training and evaluation
- [ ] Unit tests: `tests/models/test_imu_encoder.py`

### Thermal Encoder
- [ ] Implement `src/arst/models/encoders/thermal_encoder.py`
  - [ ] Evaluate Option A (linear + transformer) vs Option B (2D CNN + transformer)
- [ ] Unimodal training and evaluation

### ToF Encoder
- [ ] Implement `src/arst/models/encoders/tof_encoder.py`
- [ ] Evaluate shared vs independent weights with ThermalEncoder
- [ ] Unimodal training and evaluation

### Simple Multimodal Fusion
- [ ] Implement `src/arst/models/fusion/concat_fusion.py`
- [ ] Implement `src/arst/models/fusion/mean_fusion.py`
- [ ] Train multimodal (no reliability) and compare to baselines

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MILESTONE 6 — Phase 4: Reliability Module
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### ARM Implementation
- [ ] Implement `src/arst/models/reliability/arm.py`
  - [ ] Sigmoid variant
  - [ ] Softmax variant
- [ ] Implement `src/arst/training/losses/reliability_loss.py`
  - [ ] Entropy regularization
  - [ ] Diversity regularization
- [ ] W&B reliability logging hooks

### Reliability Training
- [ ] Train with sigmoid ARM; sweep λ_rel
- [ ] Train with softmax ARM; sweep λ_rel
- [ ] Plot reliability histograms per epoch

### Reliability Validation
- [ ] Implement corruption probe experiment
- [ ] Compute Reliability-Accuracy Correlation (RAC)
- [ ] `notebooks/19_reliability_validation.ipynb`
- [ ] Report: `reports/phase4_reliability_report.md`

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MILESTONE 7 — Phase 5: ARST Full Model
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### AFT Implementation
- [ ] Implement `src/arst/models/fusion/adaptive_fusion_transformer.py`
  - [ ] Reliability bias matrix computation
  - [ ] Modal token embeddings
  - [ ] CLS token pooling
- [ ] Memory profiling on RTX 3060
- [ ] Enable gradient checkpointing if needed

### ARST Model Integration
- [ ] Implement `src/arst/models/arst.py`
  - [ ] `ARSTModel(nn.Module)`: full end-to-end model
  - [ ] Forward pass with reliability logging
- [ ] Config: `configs/model/arst_full.yaml`
- [ ] Unit tests: `tests/models/test_arst.py`

### ARST Training
- [ ] End-to-end training run
- [ ] W&B sweep: LR, D, L, λ_rel
- [ ] Select best hyperparameters
- [ ] Final evaluation vs all baselines

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MILESTONE 8 — Phase 6: Robustness
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- [ ] Implement modality dropout augmentation
- [ ] Train ARST with dropout augmentation
- [ ] Evaluate: single modality missing (×3 conditions)
- [ ] Evaluate: two modalities missing (×3 conditions)
- [ ] Plot performance degradation curves
- [ ] `notebooks/24_robustness_evaluation.ipynb`

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MILESTONE 9 — Phase 7: Ablation Studies
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- [ ] Implement `src/arst/evaluation/ablation_runner.py`
- [ ] Run full ablation suite (see CONTEXT.md §11)
- [ ] Bootstrap confidence intervals (5 seeds per variant)
- [ ] Statistical significance tests
- [ ] Generate LaTeX ablation table
- [ ] Error analysis and confusion matrix clustering
- [ ] Efficiency analysis (FLOPs, latency, memory)

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MILESTONE 10 — Phase 8: Explainability
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- [ ] Implement `src/arst/explainability/reliability_visualizer.py`
- [ ] Implement `src/arst/explainability/saliency.py`
- [ ] Implement `src/arst/explainability/attention_visualizer.py`
- [ ] Implement `src/arst/explainability/shap_explainer.py`
- [ ] Generate all paper-ready figures
- [ ] Write `reports/arst_analysis_report.md`

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MILESTONE 11 — Deployment & Paper
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- [ ] ONNX export of ARST-Full
- [ ] Validate ONNX output vs PyTorch output
- [ ] Dynamic INT8 quantization
- [ ] Write model card: `deployment/MODEL_CARD.md`
- [ ] Write paper draft
- [ ] Submit to Kaggle competition
