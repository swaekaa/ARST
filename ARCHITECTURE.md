    # ARST — System Architecture

> **Adaptive Reliability Sensor Transformer**
> Full Architecture Reference Document

---

## Table of Contents

1. [Complete System Architecture](#1-complete-system-architecture)
2. [Data Flow](#2-data-flow)
3. [Training Pipeline](#3-training-pipeline)
4. [Inference Pipeline](#4-inference-pipeline)
5. [Reliability Module](#5-reliability-module)
6. [Fusion Module](#6-fusion-module)
7. [Explainability Module](#7-explainability-module)

---

## 1. Complete System Architecture

### 1.1 Bird's-Eye View

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         ARST SYSTEM ARCHITECTURE                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                        DATA LAYER                               │    ║
║  │  Raw Parquet → Preprocessing → Windowing → DataLoader           │    ║
║  └─────────────────────────┬───────────────────────────────────────┘    ║
║                            │                                            ║
║  ┌─────────────────────────▼───────────────────────────────────────┐    ║
║  │                     ENCODER LAYER                               │    ║
║  │  IMU Encoder │ Thermal Encoder │ ToF Encoder                    │    ║
║  │  [B,T,6]→[B,T,D] │ [B,T,64]→[B,T,D] │ [B,T,64]→[B,T,D]       │    ║
║  └─────────────────────────┬───────────────────────────────────────┘    ║
║                            │                                            ║
║  ┌─────────────────────────▼───────────────────────────────────────┐    ║
║  │                   RELIABILITY LAYER                             │    ║
║  │  ARM_imu | ARM_thm | ARM_tof                                    │    ║
║  │  [B,T,D]→[B,T,1] per modality                                  │    ║
║  └─────────────────────────┬───────────────────────────────────────┘    ║
║                            │                                            ║
║  ┌─────────────────────────▼───────────────────────────────────────┐    ║
║  │                   ADAPTIVE FUSION LAYER                         │    ║
║  │  Reliability-Biased Cross-Modal Transformer                     │    ║
║  │  [B,3T,D] + bias → [B,D]                                       │    ║
║  └─────────────────────────┬───────────────────────────────────────┘    ║
║                            │                                            ║
║  ┌─────────────────────────▼───────────────────────────────────────┐    ║
║  │                  CLASSIFICATION LAYER                           │    ║
║  │  MLP Head → Softmax → [B,C]                                     │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 1.2 Module Inventory

| Module | Class | File | Parameters (est.) |
|---|---|---|---|
| IMU Encoder | `IMUEncoder` | `models/encoders/imu_encoder.py` | ~2M |
| Thermal Encoder | `ThermalEncoder` | `models/encoders/thermal_encoder.py` | ~2M |
| ToF Encoder | `ToFEncoder` | `models/encoders/tof_encoder.py` | ~2M |
| Reliability Head (×3) | `ReliabilityHead` | `models/reliability/arm.py` | ~50K each |
| Adaptive Fusion Transformer | `AdaptiveFusionTransformer` | `models/fusion/aft.py` | ~5M |
| Classification Head | `ClassificationHead` | `models/heads/classification.py` | ~100K |
| **Total (ARST-Full)** | `ARSTModel` | `models/arst.py` | **~11M** |

---

## 2. Data Flow

### 2.1 Offline Preprocessing Pipeline

```
Raw Data (Parquet files)
        │
        ▼
┌─────────────────────────┐
│  SensorDataLoader       │
│  - Load parquet files   │
│  - Align timestamps     │
│  - Handle missing files │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  PerModalityPreprocessor│
│  IMU:                   │
│    - Bandpass filter     │
│    - Gravity removal     │
│    - Normalization       │
│  Thermal:               │
│    - Dead pixel fix      │
│    - Background sub.     │
│    - Normalization       │
│  ToF:                   │
│    - Invalid mask        │
│    - Inpainting          │
│    - Normalization       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  FeatureEngineer        │
│  (optional)             │
│  - ENMO, jerk, STFT     │
│  - Thermal centroid      │
│  - ToF gradients         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  WindowSlicer           │
│  - Fixed window: T=256  │
│  - Stride: 50% overlap  │
│  - Pad short sequences  │
│  - Label assignment     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Save to HDF5 / NPZ     │
│  data/processed/        │
└─────────────────────────┘
```

### 2.2 Online Data Flow (Training)

```
HDF5 Dataset
     │
     ▼
ARSTDataset (torch.utils.data.Dataset)
     │
     ├── __getitem__(idx)
     │     ├── Load window: X_imu [T,6], X_thm [T,64], X_tof [T,64]
     │     ├── Apply online augmentation (if training)
     │     │     ├── RandomTimeWarp
     │     │     ├── RandomChannelNoise
     │     │     └── RandomModalityDrop (for MM robustness training)
     │     └── Return (X_imu, X_thm, X_tof, label, metadata)
     │
     ▼
DataLoader (batch_size=32, num_workers=4, pin_memory=True)
     │
     ▼
     [B,T,6]    [B,T,64]   [B,T,64]
     X_imu      X_thm      X_tof
```

### 2.3 Model Forward Pass Data Flow

```
X_imu [B,T,6] ──► IMUEncoder ──────────────► H_imu [B,T,D] ──► ARM_imu ──► r_imu [B,T,1]
                                                │                                │
X_thm [B,T,64] ─► ThermalEncoder ─────────────► H_thm [B,T,D] ──► ARM_thm ──► r_thm [B,T,1]
                                                │                                │
X_tof [B,T,64] ─► ToFEncoder ─────────────────► H_tof [B,T,D] ──► ARM_tof ──► r_tof [B,T,1]
                                                │                                │
                                                ▼                                ▼
                                    Gate: Ĥ_m = r_m ⊙ H_m
                                                │
                                                ▼
                                  Concat: [Ĥ_imu; Ĥ_thm; Ĥ_tof] [B,3T,D]
                                                │
                                                ▼  ← reliability bias matrix
                                  AdaptiveFusionTransformer
                                                │
                                                ▼
                                    H_fused [B,D]  (CLS token or mean pool)
                                                │
                                                ▼
                                   ClassificationHead
                                                │
                                                ▼
                                        logits [B,C]
```

---

## 3. Training Pipeline

### 3.1 Trainer Architecture

```
Trainer
├── setup()
│   ├── Build model (ARSTModel or BaselineModel)
│   ├── Build optimizer (AdamW + cosine schedule)
│   ├── Build loss function (FocalLoss + ReliabilityLoss)
│   ├── Build dataloaders (train/val/test)
│   └── Initialize W&B run
│
├── train()
│   └── For each epoch:
│       ├── train_epoch()
│       │   └── For each batch:
│       │       ├── Forward pass
│       │       ├── Compute loss (L_cls + λ*L_rel)
│       │       ├── Backward pass (loss.backward())
│       │       ├── Gradient clipping (max_norm=1.0)
│       │       ├── Optimizer step
│       │       ├── LR scheduler step
│       │       └── Log to W&B (loss, lr, reliability histograms)
│       │
│       └── validate_epoch()
│           ├── Forward pass (no grad)
│           ├── Compute metrics (F1, AUROC, ECE)
│           ├── Save best checkpoint
│           └── Log validation metrics to W&B
│
└── finish()
    ├── Load best checkpoint
    ├── Run test evaluation
    └── Save final metrics and artifacts
```

### 3.2 Loss Computation

```python
# Pseudocode
def compute_loss(logits, labels, reliability_scores, config):
    # Primary classification loss
    if config.loss.type == "focal":
        L_cls = focal_loss(logits, labels, gamma=2.0, alpha=class_weights)
    else:
        L_cls = cross_entropy(logits, labels, weight=class_weights)

    # Reliability regularization
    L_rel = 0.0
    for r_m in reliability_scores:  # r_m: [B, T, 1]
        # Entropy regularization (anti-collapse)
        L_rel -= config.lambda_ent * binary_entropy(r_m).mean()

    # Total loss
    L_total = L_cls + config.lambda_rel * L_rel
    return L_total, {"L_cls": L_cls, "L_rel": L_rel}
```

### 3.3 Optimizer and Scheduler

```yaml
optimizer:
  type: AdamW
  lr: 1e-4
  weight_decay: 1e-2
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  type: cosine_with_warmup
  warmup_epochs: 5
  total_epochs: 100
  min_lr: 1e-6

gradient_clipping:
  max_norm: 1.0
  norm_type: 2
```

### 3.4 Training Configuration

```yaml
training:
  epochs: 100
  batch_size: 32
  accumulation_steps: 4        # Effective batch = 128
  mixed_precision: true        # AMP for RTX 3060
  compile: false               # torch.compile (optional)

  early_stopping:
    patience: 15
    monitor: val_f1_macro
    mode: max

  checkpointing:
    save_top_k: 3
    monitor: val_f1_macro
    dirpath: experiments/{run_id}/checkpoints/
```

---

## 4. Inference Pipeline

### 4.1 Standard Inference

```
Input: Raw sensor files (parquet)
        │
        ▼
PreprocessingPipeline (identical to training, no augmentation)
        │
        ▼
Window generation (stride = window_size → no overlap at inference)
        │
        ▼
ARSTInferenceEngine
    ├── Load model from checkpoint
    ├── model.eval()
    ├── torch.no_grad()
    └── For each window:
        ├── Forward pass → logits, reliability_scores
        ├── Softmax → probabilities
        └── Store: (window_idx, probs, reliability_scores)
        │
        ▼
SequenceAggregator
    ├── Aggregate window-level predictions → sequence-level prediction
    ├── Strategies: majority vote, max-pooling, learned aggregation
    └── Return: final_label, confidence, per_window_reliability
        │
        ▼
Output: predictions.csv + reliability_maps/
```

### 4.2 Test-Time Augmentation (TTA)

```
For each window:
    ├── Original window
    ├── Time-reversed window
    ├── Gaussian noise augmented (×2)
    └── TimeWarp augmented (×2)
    │
    ▼
Ensemble: average softmax probabilities across TTA variants
```

### 4.3 Ensemble Inference

```
K trained ARST models (different seeds or architectures)
        │
        ▼
For each model k:
    └── Forward pass → probs_k [B, C]
        │
        ▼
Ensemble aggregation:
    ├── Mean ensemble: mean(probs_k)
    ├── Reliability-weighted ensemble: Σ_k R_k * probs_k / Σ_k R_k
    └── Stacking: meta-learner on top of [probs_1, ..., probs_K]
```

### 4.4 ONNX Export Pipeline

```
PyTorch ARSTModel
        │
        ▼ torch.onnx.export()
ONNX Model (arst_model.onnx)
        │
        ├── Validate: onnxruntime vs PyTorch output comparison
        ├── Optimize: onnxruntime-tools graph optimization
        └── Quantize: dynamic INT8 quantization (optional)
        │
        ▼
Deployment Assets:
    deployment/
    ├── arst_model.onnx
    ├── arst_model_int8.onnx
    ├── preprocessor_config.json
    └── label_encoder.json
```

---

## 5. Reliability Module

### 5.1 Adaptive Reliability Module (ARM) Architecture

```
Input: H_m ∈ ℝ^{B×T×D}   (encoder output for modality m)
    │
    ▼
Linear(D → D_h)   where D_h = D // 4
    │
    ▼
ReLU
    │
    ▼
Dropout(p=0.1)
    │
    ▼
Linear(D_h → 1)
    │
    ▼
Sigmoid
    │
    ▼
r_m ∈ ℝ^{B×T×1}     (reliability score: 0=unreliable, 1=reliable)
```

### 5.2 ARM Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Output activation | Sigmoid | Continuous [0,1] output; gradient flows at all values |
| Alternative | Softmax (cross-modality) | Competitive; tested in ablation |
| Bottleneck dim | D//4 | Reduce parameters; reliability is low-dimensional signal |
| Gradient flow | Direct (no stop-gradient) | Allow reliability to influence encoder training |
| Initialization | Bias = 0 → σ(0) = 0.5 | Start with equal reliability; learn from data |

### 5.3 Reliability Score Monitoring

During training, log the following to W&B:

```python
# Per-modality reliability statistics (per batch)
wandb.log({
    "reliability/imu_mean": r_imu.mean().item(),
    "reliability/imu_std": r_imu.std().item(),
    "reliability/thm_mean": r_thm.mean().item(),
    "reliability/thm_std": r_thm.std().item(),
    "reliability/tof_mean": r_tof.mean().item(),
    "reliability/tof_std": r_tof.std().item(),
    "reliability/imu_hist": wandb.Histogram(r_imu.cpu().numpy()),
    "reliability/thm_hist": wandb.Histogram(r_thm.cpu().numpy()),
    "reliability/tof_hist": wandb.Histogram(r_tof.cpu().numpy()),
})
```

**Watch for:**
- Reliability collapse: all scores → 0 or 1 (indicates loss weight imbalance)
- Modality dominance: one modality always scores r≈1 (may indicate data imbalance)
- Temporal patterns: reliability should vary across timesteps within a window

### 5.4 Reliability Probe (Validation)

To validate reliability is meaningful, run post-hoc analysis:

```python
# For each validation sample:
# 1. Compute reliability scores
# 2. Artificially corrupt one modality (add Gaussian noise σ=3.0)
# 3. Re-compute reliability scores
# 4. Check: corrupted modality should have lower reliability

delta_r_imu = r_imu_corrupted - r_imu_clean  # Should be negative
```

---

## 6. Fusion Module

### 6.1 Adaptive Fusion Transformer (AFT) Architecture

```
Inputs:
    Ĥ_imu ∈ ℝ^{B×T×D}   (reliability-gated IMU embedding)
    Ĥ_thm ∈ ℝ^{B×T×D}   (reliability-gated thermal embedding)
    Ĥ_tof ∈ ℝ^{B×T×D}   (reliability-gated ToF embedding)
    r_imu, r_thm, r_tof ∈ ℝ^{B×T×1}   (reliability scores)

Step 1: Prepend [MODAL] tokens
    Ĥ_imu_cls = Concat([modal_token_imu; Ĥ_imu])  → [B, T+1, D]
    (similarly for thm, tof)

Step 2: Concatenate all modalities
    H_cat = Concat(Ĥ_imu_cls, Ĥ_thm_cls, Ĥ_tof_cls)  → [B, 3(T+1), D]

Step 3: Build reliability bias matrix
    R_all = Concat(r_imu, r_thm, r_tof) → [B, 3T, 1]
    # Bias: tokens from unreliable modalities receive less attention
    attn_bias = log(R_all + eps)  # [B, 3T] added to key attention scores

Step 4: L-layer Transformer with reliability bias
    For l in range(L):
        H_cat = TransformerBlock(H_cat, attn_bias=attn_bias)

Step 5: Extract fused representation
    H_fused = H_cat[:, 0, :]   # CLS token (or mean pool)

Output:
    H_fused ∈ ℝ^{B×D}
```

### 6.2 AFT Hyperparameters

```yaml
fusion_transformer:
  num_layers: 4
  num_heads: 8
  d_model: 256
  d_ff: 1024
  dropout: 0.1
  attention_dropout: 0.1
  use_reliability_bias: true
  use_modal_tokens: true
  pool_type: cls              # 'cls' or 'mean'
```

### 6.3 Fusion Variants (Ablation)

| Variant | Description | Complexity |
|---|---|---|
| `concat` | Flatten all embeddings; single MLP | O(1) |
| `mean_pool` | Average per-modality embeddings; MLP | O(1) |
| `weighted_sum` | Learned static weights; weighted sum | O(M) |
| `arst_simple` | Reliability-weighted sum | O(M×T) |
| `arst_full` | AFT with reliability bias | O(M²×T²) |

---

## 7. Explainability Module

### 7.1 Reliability Score Visualization

```
For a given sequence:
    ├── Run forward pass → r_imu [T], r_thm [T], r_tof [T]
    └── Plot: temporal reliability heatmap per modality
        │
        ▼
    ┌──────────────────────────────────────────────────────┐
    │  t:   0    50   100  150  200  250                   │
    │ IMU: ████ ████ ░░░░ ███ ████ ████  (high→low→high)  │
    │ THM: ████ ████ ████ ████ ███ ██░░  (gradual decay)  │
    │ ToF: ░░░░ ░░░░ ████ ████ ████ ████  (late onset)    │
    └──────────────────────────────────────────────────────┘
```

### 7.2 Saliency Maps

**Gradient-based saliency (per modality):**

```python
saliency = autograd.grad(logits[:, target_class].sum(), X_imu)[0]
# Visualize: |saliency| per timestep and channel
```

**Integrated Gradients:**

```python
# Baseline: zero input (silent sensor)
# Interpolation path: baseline → actual input
# Attribution: IG(x_i) = (x_i - x_i_baseline) × ∫_0^1 ∂F/∂x_i dt
```

### 7.3 Attention Visualization

Extract attention weights from AFT layers:

```python
# For each attention head h in each layer l:
A_lh ∈ ℝ^{B×3T×3T}   # Full cross-modal attention matrix
# Block structure reveals which modality attends to which
# Block (imu→thm): A_lh[0:T, T:2T]
```

### 7.4 SHAP-based Modality Attribution

```python
import shap
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_inputs)
# shap_values[class_idx][modality_idx] → [B, T, feature_dim]
# Aggregate: mean |SHAP| per modality → global modality importance
```

### 7.5 Reliability-Accuracy Analysis

```python
# Compute per-sample:
# - Aggregate reliability: R_m = mean(r_m) per modality per sample
# - Correctness: correct = (predicted_label == true_label)

# Analysis:
# 1. Scatter plot: R_m vs correctness (expect positive correlation)
# 2. Calibration: mean(R_m | correct) vs mean(R_m | incorrect)
# 3. Reliability histograms per class
# 4. Per-modality reliability distribution per behavior class
```

### 7.6 Class Activation Mapping (CAM) for Temporal Data

Adapted GradCAM for 1D temporal sequences:

```python
# For each target class c:
# 1. Compute gradients of logit_c w.r.t. final encoder feature map
# 2. Global average pool gradients → weights alpha_k^c
# 3. CAM_c = ReLU(Σ_k alpha_k^c * A_k)   (A_k: activation map)
# 4. CAM_c ∈ [0,1]^T → temporal importance per class
```

### 7.7 Explainability Output Structure

```
reports/explainability/
├── reliability_maps/
│   ├── per_class/          # Avg reliability per class per modality
│   └── per_sample/         # Individual sample reliability timelines
├── saliency/
│   ├── gradient_saliency/
│   └── integrated_gradients/
├── attention/
│   ├── cross_modal_attn/   # Heatmaps of AFT attention blocks
│   └── layer_by_layer/
├── shap/
│   ├── summary_plots/
│   └── beeswarm_plots/
└── activation_maps/
    └── temporal_cam/
```
