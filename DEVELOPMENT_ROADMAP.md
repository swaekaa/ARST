# ARST — Development Roadmap

> **Adaptive Reliability Sensor Transformer**
> Phased Research Development Plan

---

## Roadmap Overview

```
Phase 1          Phase 2         Phase 3          Phase 4
Dataset EDA   →  Baselines   →  Encoders      →  Reliability
(Weeks 1–2)      (Weeks 3–4)    (Weeks 5–6)      (Weeks 7–8)
     │
     ▼
Phase 5          Phase 6         Phase 7          Phase 8
ARST Fusion   →  Robustness  →  Evaluation    →  Explainability
(Weeks 9–10)     (Week 11)       (Weeks 12–13)    (Weeks 14–15)
```

---

## Phase 1 — Dataset Exploration & EDA

**Duration:** 2 weeks
**Deliverable:** EDA report, data quality assessment, preprocessing decisions

### 1.1 Data Loading & Structure Analysis

- [ ] Download CMI competition data via Kaggle API
- [ ] Inspect raw parquet structure (columns, dtypes, timestamps)
- [ ] Audit sequence counts per subject per class
- [ ] Identify missing sequence files (modalities missing entirely)
- [ ] Audit sensor sampling rates and timestamp alignment

### 1.2 Exploratory Data Analysis

**IMU:**
- [ ] Time-series plots of acc/gyro per class
- [ ] Signal magnitude distribution per class
- [ ] FFT spectrum analysis per class
- [ ] Autocorrelation analysis

**Thermopile:**
- [ ] Thermal map visualizations (8×8 frames)
- [ ] Temporal thermal heatmap animations
- [ ] Dead pixel map across dataset

**ToF:**
- [ ] Depth map visualizations (8×8 frames)
- [ ] Invalid reading frequency map
- [ ] Temporal depth profile per class

### 1.3 Statistical Analysis

- [ ] Class distribution analysis (counts, percentages, imbalance ratio)
- [ ] Sequence length distribution (min, max, mean, std, percentiles)
- [ ] Missing value analysis (per modality, per class, per subject)
- [ ] Inter-subject variability analysis (signal statistics per subject)
- [ ] Class-conditional feature distribution plots

### 1.4 Preprocessing Decisions

- [ ] Determine optimal window size (T) via autocorrelation + class duration analysis
- [ ] Choose normalization strategy (global, per-subject, per-channel)
- [ ] Design missing value handling strategy per modality
- [ ] Define train/validation/test split (stratified by class + subject)

**Notebooks:**
- `notebooks/01_data_loading.ipynb`
- `notebooks/02_imu_eda.ipynb`
- `notebooks/03_thermal_eda.ipynb`
- `notebooks/04_tof_eda.ipynb`
- `notebooks/05_class_imbalance_analysis.ipynb`
- `notebooks/06_sequence_length_analysis.ipynb`
- `notebooks/07_missing_data_analysis.ipynb`

---

## Phase 2 — Baseline Models

**Duration:** 2 weeks
**Deliverable:** 4 baseline models trained and evaluated; baseline leaderboard

### 2.1 Feature Extraction Baseline (MLP)

- [ ] Implement statistical feature extraction (mean, std, min, max, skewness, kurtosis per channel)
- [ ] Implement `MLPBaseline`: flat feature vector → MLP → classification
- [ ] Train with standard CE loss + class weights
- [ ] Evaluate on val set (Macro F1, Balanced Accuracy, AUROC)

### 2.2 Convolutional Baseline (CNN)

- [ ] Implement `CNNBaseline`: multi-scale 1D CNN per modality + concat fusion
- [ ] Kernel sizes: [3, 7, 15] → feature pyramid → global avg pool → MLP
- [ ] Train and evaluate

### 2.3 Recurrent Baseline (BiLSTM)

- [ ] Implement `LSTMBaseline`: bidirectional LSTM per modality + concat fusion
- [ ] Variable-length support via packing/masking
- [ ] Train and evaluate

### 2.4 Transformer Baseline

- [ ] Implement `TransformerBaseline`: concatenate all modalities + Transformer encoder
- [ ] Sinusoidal or learned positional encoding
- [ ] [CLS] token for classification
- [ ] Train and evaluate

### 2.5 Analysis

- [ ] Compare baselines on leaderboard
- [ ] Identify which modality contributes most (unimodal ablations)
- [ ] Identify dominant error patterns (confusion matrix analysis)
- [ ] Profile training speed and memory (RTX 3060 4GB constraints)

**Notebooks:**
- `notebooks/08_baseline_mlp.ipynb`
- `notebooks/09_baseline_cnn.ipynb`
- `notebooks/10_baseline_lstm.ipynb`
- `notebooks/11_baseline_transformer.ipynb`
- `notebooks/12_baseline_comparison.ipynb`

---

## Phase 3 — Sensor-Specific Encoders

**Duration:** 2 weeks
**Deliverable:** Three specialized encoders; modality-specific performance benchmarks

### 3.1 IMU Encoder

- [ ] Design `IMUEncoder`: multi-scale 1D CNN → Transformer blocks
- [ ] Incorporate domain knowledge: filter scales for different motion types
- [ ] Optional: spectrogram input branch (STFT features as 2D CNN input)
- [ ] Unimodal evaluation

### 3.2 Thermal Encoder

- [ ] Design `ThermalEncoder`:
  - Option A: Linear projection of 64-dim vector + temporal Transformer
  - Option B: Per-frame 2D CNN (treating 8×8 as image) + temporal Transformer
- [ ] Handle missing/dead pixels with learned masking
- [ ] Unimodal evaluation

### 3.3 ToF Encoder

- [ ] Design `ToFEncoder` (mirror ThermalEncoder architecture)
- [ ] Add invalid-reading mask as auxiliary input
- [ ] Ablate: shared vs independent weights with ThermalEncoder
- [ ] Unimodal evaluation

### 3.4 Simple Multimodal Fusion (No Reliability)

- [ ] Implement `ConcatFusionModel`: all three encoders + concat + MLP
- [ ] Implement `MeanFusionModel`: mean pool embeddings per modality
- [ ] Evaluate: do specialized encoders outperform generic baseline encoders?

**Notebooks:**
- `notebooks/13_imu_encoder_design.ipynb`
- `notebooks/14_thermal_encoder_design.ipynb`
- `notebooks/15_tof_encoder_design.ipynb`
- `notebooks/16_encoder_comparison.ipynb`

---

## Phase 4 — Reliability Estimation Module

**Duration:** 2 weeks
**Deliverable:** ARM implementation; validation that reliability scores are meaningful

### 4.1 ARM Implementation

- [ ] Implement `AdaptiveReliabilityModule` (lightweight MLP head)
- [ ] Implement reliability regularization losses (entropy, diversity)
- [ ] Add reliability histograms to W&B logging
- [ ] Design reliability probe experiment (corruption test)

### 4.2 Reliability Training

- [ ] Train ARM with sigmoid output (independent per-modality)
- [ ] Train ARM with softmax output (competitive across modalities)
- [ ] Sweep `λ_rel` hyperparameter (0.0, 0.01, 0.1, 1.0)
- [ ] Monitor reliability collapse / saturation

### 4.3 Reliability Validation

- [ ] Run corruption probe: inject Gaussian noise, check reliability drops
- [ ] Compute Reliability-Accuracy Correlation (RAC)
- [ ] Visualize reliability timelines per behavior class
- [ ] Check if reliability generalizes across subjects

**Notebooks:**
- `notebooks/17_reliability_module_design.ipynb`
- `notebooks/18_reliability_training.ipynb`
- `notebooks/19_reliability_validation.ipynb`

---

## Phase 5 — Adaptive Fusion Transformer

**Duration:** 2 weeks
**Deliverable:** Full ARST model; comparison against all baselines

### 5.1 AFT Implementation

- [ ] Implement `AdaptiveFusionTransformer` with reliability bias
- [ ] Implement modal tokens ([MODAL_IMU], [MODAL_THM], [MODAL_TOF])
- [ ] Implement [CLS] token pooling
- [ ] Memory profiling on RTX 3060 4GB (tune sequence length accordingly)

### 5.2 ARST Integration

- [ ] Implement `ARSTModel`: integrate encoders + ARM + AFT + ClassHead
- [ ] End-to-end training (all components jointly)
- [ ] Implement gradient checkpointing if needed for memory
- [ ] Benchmark: ARST-Full vs all Phase 2/3 baselines

### 5.3 Hyperparameter Optimization

- [ ] W&B Sweep on:
  - Learning rate
  - Batch size (with gradient accumulation)
  - Transformer depth (L = 2, 4, 6)
  - Embedding dim (D = 128, 256)
  - λ_rel values
- [ ] Select best configuration as ARST-Full

**Notebooks:**
- `notebooks/20_aft_implementation.ipynb`
- `notebooks/21_arst_full_training.ipynb`
- `notebooks/22_hyperparameter_sweep.ipynb`

---

## Phase 6 — Missing Modality Robustness

**Duration:** 1 week
**Deliverable:** Robustness analysis; ARST performance under sensor failure

### 6.1 Missing Modality Simulation

- [ ] Implement `ModalityDropAugmentation`: zero-fill, Gaussian noise, learned null token
- [ ] Train ARST-Full with random modality dropout (p=0.2 per modality per batch)
- [ ] Evaluate: ARST-Full vs ARST-NoDrop under missing modality conditions

### 6.2 Robustness Benchmarks

- [ ] Evaluate all models with each modality dropped individually
- [ ] Evaluate with two modalities dropped simultaneously
- [ ] Plot: performance degradation curve vs fraction of missing data

### 6.3 Reliability Under Missing Modality

- [ ] Verify: dropped modality receives r ≈ 0.0
- [ ] Check: remaining modalities compensate (reliability increases)

**Notebooks:**
- `notebooks/23_missing_modality_simulation.ipynb`
- `notebooks/24_robustness_evaluation.ipynb`

---

## Phase 7 — Evaluation & Ablation Studies

**Duration:** 2 weeks
**Deliverable:** Full ablation table; statistical significance tests; final leaderboard

### 7.1 Full Ablation Suite

- [ ] Run all variants defined in CONTEXT.md §11
- [ ] Bootstrap confidence intervals (5 seeds per variant)
- [ ] Paired t-tests for significance
- [ ] Generate LaTeX ablation table

### 7.2 Error Analysis

- [ ] Per-class performance breakdown
- [ ] Confusion matrix analysis (cluster errors)
- [ ] Identify failure modes (which behaviors are hardest?)
- [ ] Analyze high-error samples (what's different about them?)

### 7.3 Efficiency Analysis

- [ ] Measure FLOPs per forward pass per model
- [ ] Measure inference latency (ms/sample on CPU and GPU)
- [ ] Measure training time and peak GPU memory
- [ ] Pareto frontier: F1 vs latency

**Notebooks:**
- `notebooks/25_ablation_suite.ipynb`
- `notebooks/26_error_analysis.ipynb`
- `notebooks/27_efficiency_analysis.ipynb`

---

## Phase 8 — Explainability & Visualization

**Duration:** 2 weeks
**Deliverable:** Explainability report; visualizations for paper figures

### 8.1 Reliability Visualization

- [ ] Temporal reliability heatmap per modality per class
- [ ] Per-class average reliability bar chart
- [ ] Reliability distribution plots (per modality, per class)

### 8.2 Saliency Analysis

- [ ] Gradient saliency maps for all modalities
- [ ] Integrated Gradients attribution
- [ ] Compare: saliency alignment with reliability scores

### 8.3 Attention Visualization

- [ ] Cross-modal attention heatmaps from AFT
- [ ] Layer-by-layer attention evolution
- [ ] Head diversity analysis

### 8.4 SHAP Analysis

- [ ] SHAP values for modality importance
- [ ] Summary plot (feature importance ranking)
- [ ] Interaction effects between modalities

### 8.5 Report Generation

- [ ] Generate paper-ready figures (300 DPI, LaTeX-compatible)
- [ ] Write `reports/arst_analysis_report.pdf`

**Notebooks:**
- `notebooks/28_reliability_visualization.ipynb`
- `notebooks/29_saliency_analysis.ipynb`
- `notebooks/30_attention_visualization.ipynb`
- `notebooks/31_shap_analysis.ipynb`
- `notebooks/32_paper_figures.ipynb`

---

## Hardware Constraints & Mitigation

### RTX 3060 4GB Limitations

| Issue | Mitigation Strategy |
|---|---|
| GPU OOM at batch_size=32 | Use gradient accumulation (steps=4) |
| Long sequences (T=256+) | Gradient checkpointing in Transformer |
| Large ablation sweeps | Run serially; use CPU for small models |
| Mixed precision | Enable `torch.cuda.amp` (AMP) |
| Multiple simultaneous experiments | Use W&B sweep agents sequentially |

### Memory Budget Estimate

| Model Component | Approx GPU Memory |
|---|---|
| ARST-Full (D=256, L=4, T=256) | ~2.5 GB |
| Batch of 32 samples | ~0.5 GB |
| Optimizer states (AdamW) | ~0.8 GB |
| **Total** | **~3.8 GB** (within 4 GB) |

**Fallback:** If memory is tight, reduce D=128 or T=128 and verify performance degradation is minimal.

---

## Milestones & Success Criteria

| Milestone | Target | Success Criteria |
|---|---|---|
| Phase 1 complete | Week 2 | EDA notebook published; preprocessing pipeline coded |
| Phase 2 complete | Week 4 | Best baseline F1 > random chance by 2× |
| Phase 3 complete | Week 6 | Encoder models beat simple baselines |
| Phase 4 complete | Week 8 | Reliability probe correlation > 0.3 |
| Phase 5 complete | Week 10 | ARST-Full > best baseline (statistically significant) |
| Phase 6 complete | Week 11 | ARST-Full degrades gracefully (< 10% drop with 1 modality missing) |
| Phase 7 complete | Week 13 | Full ablation table with CIs; effect sizes reported |
| Phase 8 complete | Week 15 | Paper-ready figures; explainability report |
