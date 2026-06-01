# ARST — Technical Research Context

> **Adaptive Reliability Sensor Transformer**
> Multimodal Behavior Recognition from Wearable Sensor Streams

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Motivation](#2-research-motivation)
3. [Dataset Description](#3-dataset-description)
4. [Sensor Modality Description](#4-sensor-modality-description)
5. [Proposed ARST Architecture](#5-proposed-arst-architecture)
6. [Mathematical Formulation](#6-mathematical-formulation)
7. [Reliability Score Equations](#7-reliability-score-equations)
8. [Adaptive Fusion Equations](#8-adaptive-fusion-equations)
9. [Planned Experiments](#9-planned-experiments)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Ablation Strategy](#11-ablation-strategy)
12. [Future Work](#12-future-work)

---

## 1. Project Overview

ARST (Adaptive Reliability Sensor Transformer) is a research-grade deep learning system designed for **multimodal behavior recognition** using heterogeneous wearable sensor streams. The system targets the Kaggle CMI competition dataset, which contains synchronized recordings from IMU, thermopile, and time-of-flight (ToF) sensors.

The central hypothesis is that **not all sensor modalities are equally informative at every timestep**. A sensor may be degraded due to:

- Physical occlusion or poor skin contact
- Subject motion artifacts
- Environmental interference
- Hardware limitations (e.g., thermal drift, ToF ambiguity at close range)

Traditional fusion approaches (concatenation, mean pooling, learned weighted sum) apply **static** modality weights that cannot adapt to per-sample or per-timestep signal quality. ARST addresses this with a **dynamic reliability estimation module** that learns to score modality trustworthiness from the signal itself.

---

## 2. Research Motivation

### 2.1 Limitations of Existing Approaches

| Approach | Limitation |
|---|---|
| Concatenation fusion | Treats all modalities equally; noisy modality pollutes fused representation |
| Static weighted sum | Weights are global; cannot adapt to signal quality at inference time |
| Hard modality dropout | Binary; doesn't capture partial degradation |
| Attention-based fusion | Attention is over tokens, not over modality trustworthiness |
| Missing modality imputation | Introduces distribution shift; relies on strong priors |

### 2.2 Opportunity

Wearable sensor deployments in naturalistic settings exhibit frequent partial data quality degradation — precisely the condition under which fixed fusion breaks down. By learning a **continuous, differentiable reliability score** per modality per timestep, we enable:

1. **Graceful degradation**: Corrupted modalities are downweighted, not discarded.
2. **Interpretability**: Reliability scores provide per-modality, per-timestep attribution.
3. **Robustness**: The model implicitly learns when to trust each sensor.
4. **Data efficiency**: Training does not require paired "clean" and "corrupted" data.

### 2.3 Connection to Literature

ARST draws from and extends:

- **MulT** (Tsai et al., 2019): Cross-modal Transformer; extended with reliability gating.
- **Perceiver IO** (Jaegle et al., 2021): Latent array cross-attention; adapted for sensor streams.
- **Confident Learning** (Northcutt et al., 2021): Label noise estimation; analogous to modality noise estimation.
- **MC Dropout** (Gal & Ghahramani, 2016): Uncertainty quantification; reliability scores approximate epistemic confidence.
- **CAFe** (Tran et al., 2023): Calibrated fusion; our reliability head is conceptually a learned calibration.

---

## 3. Dataset Description

### 3.1 Competition

**CMI — Detect Behavior with Sensor Data**
Kaggle URL: `https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states`

### 3.2 Task

Multi-class behavior classification from synchronized, multi-modal wearable sensor recordings. Each sample is a fixed-length time window; the label is a discrete behavior class.

### 3.3 Data Structure

```
data/raw/
├── train.csv                  # Training labels and metadata
├── test.csv                   # Test sequences (no labels)
├── sample_submission.csv      # Submission format
└── sensor_data/
    ├── train/
    │   └── <sequence_id>/
    │       ├── accel.parquet  # Accelerometer (x, y, z)
    │       ├── gyro.parquet   # Gyroscope (x, y, z)
    │       ├── thermo.parquet # Thermopile (64-channel)
    │       └── tof.parquet    # Time-of-Flight (64-channel)
    └── test/
        └── <sequence_id>/
            └── ...
```

### 3.4 Class Distribution

The dataset exhibits significant class imbalance — a key challenge requiring:
- Weighted loss functions (focal loss, class-weighted CE)
- Stratified sampling
- Macro-averaged evaluation metrics

### 3.5 Sequence Statistics

| Property | Value |
|---|---|
| Total sequences | TBD (post-EDA) |
| Sequence duration | Variable; windowed to fixed length |
| Window size (target) | 128–512 timesteps |
| Overlap | 50% (configurable) |
| Missing value rate | TBD (per-modality) |

---

## 4. Sensor Modality Description

### 4.1 IMU — Inertial Measurement Unit

**Sensors:** 3-axis accelerometer + 3-axis gyroscope
**Raw features:** `[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]` → 6 channels
**Sampling rate:** ~50 Hz
**Key behaviors captured:** Arm movements, walking, sitting, gestures

**Preprocessing pipeline:**
1. Bandpass filtering (0.5–20 Hz for motion; 0.1–3.5 Hz for activity)
2. Gravity removal (high-pass filter or orientation estimation)
3. Signal magnitude calculation: `|a| = sqrt(acc_x² + acc_y² + acc_z²)`
4. Jerk computation: `j_t = a_t - a_{t-1}`
5. Normalization (z-score per channel, per subject)

**Engineered features (optional):**
- ENMO (Euclidean Norm Minus One)
- Tilt angles (roll, pitch)
- Spectral features via STFT
- Wavelet decomposition coefficients

**Reliability challenges:**
- Motion artifacts from cable/strap movement
- Sensor saturation during high-impact events (clipping)
- Orientation ambiguity without magnetometer

### 4.2 Thermopile Array

**Sensors:** 8×8 grid of infrared thermopile elements
**Raw features:** 64 channels (one per pixel) → reshaped as `[8, 8]` spatial thermal maps
**Sampling rate:** ~10 Hz
**Key behaviors captured:** Body positioning, proximity to objects, thermal activity patterns

**Preprocessing pipeline:**
1. Dead pixel correction (median filter over neighbors)
2. Background subtraction (running mean baseline removal)
3. Spatial smoothing (Gaussian blur, σ=0.5)
4. Temperature normalization (relative to ambient)
5. Temporal stacking → `[T, 8, 8]` volume

**Engineered features (optional):**
- Centroid of thermal mass
- Max-min temperature spread
- Thermal entropy (spatial uniformity)

**Reliability challenges:**
- Thermal drift over long recording sessions
- Ambient temperature interference
- Limited spatial resolution (8×8)
- Slow thermal response time (thermal inertia)

### 4.3 Time-of-Flight (ToF) Sensor

**Sensors:** 8×8 distance-sensing array using infrared light
**Raw features:** 64 channels → `[8, 8]` depth maps (distance in mm)
**Sampling rate:** ~10 Hz
**Key behaviors captured:** Limb proximity, gesture patterns, body silhouette

**Preprocessing pipeline:**
1. Invalid reading masking (out-of-range returns → -1 flag)
2. Inpainting (bilinear interpolation for masked pixels)
3. Depth range clipping (0–1000 mm operational range)
4. Normalization to [0, 1]
5. Temporal stacking → `[T, 8, 8]` volume

**Engineered features (optional):**
- Depth histogram statistics
- Motion flow estimation (frame differencing)
- Distance gradients (edge detection in depth space)

**Reliability challenges:**
- Multi-path interference (reflective surfaces)
- Distance ambiguity at very close/far range (<5 cm, >2 m)
- Invalid returns on non-reflective surfaces (dark clothing)
- High sensitivity to sensor orientation

---

## 5. Proposed ARST Architecture

### 5.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ARST FULL ARCHITECTURE                             │
├───────────────┬──────────────────┬──────────────────────────────────────────┤
│  MODALITY 1   │    MODALITY 2    │           MODALITY 3                     │
│               │                  │                                          │
│ IMU Sequence  │ Thermo Sequence  │          ToF Sequence                    │
│ [B, T, 6]     │ [B, T, 64]       │          [B, T, 64]                      │
│      │        │       │          │               │                          │
│  IMU Encoder  │ Thermo Encoder   │          ToF Encoder                     │
│      │        │       │          │               │                          │
│  [B, T, D]    │  [B, T, D]       │          [B, T, D]                       │
│      │        │       │          │               │                          │
│  Reliability  │  Reliability     │          Reliability                     │
│    Head       │    Head          │            Head                          │
│      │        │       │          │               │                          │
│  r_imu ∈ [0,1]│ r_thm ∈ [0,1]   │          r_tof ∈ [0,1]                   │
└──────┬────────┴───────┬──────────┴───────────────┬──────────────────────────┘
       │                │                           │
       └────────────────┼───────────────────────────┘
                        │ Reliability-Gated Embeddings
                        ▼
            ┌───────────────────────┐
            │  Adaptive Fusion      │
            │  Transformer (AFT)    │
            │  (Cross-modal attn    │
            │   + reliability gate) │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Classification Head  │
            │  (MLP + Softmax)      │
            └───────────┬───────────┘
                        │
                        ▼
                  Behavior Class
```

### 5.2 Component Summary

| Component | Input Shape | Output Shape | Purpose |
|---|---|---|---|
| IMU Encoder | `[B, T, 6]` | `[B, T, D]` | Temporal feature extraction from motion data |
| Thermal Encoder | `[B, T, 64]` | `[B, T, D]` | Spatial-temporal thermal feature extraction |
| ToF Encoder | `[B, T, 64]` | `[B, T, D]` | Spatial-temporal depth feature extraction |
| Reliability Head (×3) | `[B, T, D]` | `[B, T, 1]` | Per-timestep modality reliability score |
| Adaptive Fusion Transformer | `[B, T, D]×3 + scores×3` | `[B, D]` | Reliability-gated cross-modal fusion |
| Classification Head | `[B, D]` | `[B, C]` | Final behavior class prediction |

### 5.3 Encoder Architectures

**IMU Encoder (Temporal)**
- 1D CNN feature extractor: kernel sizes [3, 5, 7] → multi-scale temporal features
- Positional encoding (sinusoidal or learned)
- Stacked Transformer encoder blocks (L=4, H=8, D=256)

**Thermal Encoder (Spatial-Temporal)**
- Spatial: Linear projection of 64-dim flattened thermal map
- Temporal: Transformer encoder or 1D convolutions over T
- Optional: 2D CNN per frame (if treating each timestep as 8×8 image)

**ToF Encoder (Spatial-Temporal)**
- Architecture mirrors Thermal Encoder (shared or independent weights)
- Additional masking layer for invalid returns (learned null embedding for masked pixels)

---

## 6. Mathematical Formulation

### 6.1 Input Definitions

Let the input multimodal sequence be:

```
X = {X_imu, X_thm, X_tof}
```

Where:
- `X_imu ∈ ℝ^{B×T×6}` — IMU: batch B, timesteps T, 6 channels
- `X_thm ∈ ℝ^{B×T×64}` — Thermopile: 64 flattened pixels
- `X_tof ∈ ℝ^{B×T×64}` — ToF: 64 flattened pixels

### 6.2 Encoder Forward Pass

For modality `m ∈ {imu, thm, tof}`:

```
H_m = Encoder_m(X_m; θ_m)     H_m ∈ ℝ^{B×T×D}
```

Where `θ_m` are modality-specific encoder parameters and `D` is the shared embedding dimension.

### 6.3 Reliability Score Computation

See §7 for full equations.

### 6.4 Reliability-Gated Embeddings

```
Ĥ_m = r_m ⊙ H_m     Ĥ_m ∈ ℝ^{B×T×D}
```

Where `r_m ∈ ℝ^{B×T×1}` is broadcast multiplied (Hadamard product).

### 6.5 Adaptive Fusion

```
H_fused = AFT(Ĥ_imu, Ĥ_thm, Ĥ_tof; r_imu, r_thm, r_tof)
```

The AFT performs cross-modal attention with reliability-biased attention weights. See §8.

### 6.6 Classification

```
ẑ = ClassHead(Pooled(H_fused))     ẑ ∈ ℝ^{B×C}
ŷ = Softmax(ẑ)
```

### 6.7 Training Objective

```
L_total = L_cls + λ_rel * L_rel + λ_reg * L_reg
```

Where:
- `L_cls`: Cross-entropy (or focal loss for class imbalance)
- `L_rel`: Reliability regularization (see §7.4)
- `L_reg`: L2 weight decay
- `λ_rel`, `λ_reg`: Hyperparameters

---

## 7. Reliability Score Equations

### 7.1 Reliability Head Architecture

The reliability head for modality `m` is a lightweight MLP applied per timestep:

```
r_m^t = σ(W_2 · ReLU(W_1 · h_m^t + b_1) + b_2)
```

Where:
- `h_m^t ∈ ℝ^D` — encoder output at timestep t
- `W_1 ∈ ℝ^{D_h×D}`, `W_2 ∈ ℝ^{1×D_h}` — learnable weights
- `D_h = D/4` — bottleneck dimension (reduce computation)
- `σ(·)` — sigmoid activation → output in (0, 1)
- `r_m^t ∈ (0, 1)` — scalar reliability at timestep t

**Full temporal reliability vector:**

```
r_m = [r_m^1, r_m^2, ..., r_m^T]     r_m ∈ ℝ^{T}
```

### 7.2 Normalized Reliability (Optional Variant)

Apply softmax normalization across modalities to enforce a competition between modalities:

```
r̃_m^t = exp(logit_m^t) / Σ_{m'} exp(logit_{m'}^t)
```

This ensures `Σ_m r̃_m^t = 1` at each timestep — modalities compete for influence.

**Trade-off:** Softmax forces one modality to dominate; sigmoid allows all modalities to be equally trusted. Both variants will be evaluated as ablation conditions.

### 7.3 Sequence-Level Reliability Aggregation

For tasks requiring a single reliability score per modality (e.g., visualization, coarse ablations):

```
R_m = (1/T) Σ_t r_m^t     (temporal mean)
```

or

```
R_m = min_t r_m^t     (temporal minimum — conservative)
```

### 7.4 Reliability Regularization Loss

To prevent reliability collapse (all scores → 0 or 1):

**Entropy regularization** (maximize entropy → encourage uncertainty):

```
L_rel = -λ_ent * Σ_m (1/T) Σ_t [r_m^t * log(r_m^t) + (1 - r_m^t) * log(1 - r_m^t)]
```

**Diversity regularization** (discourage all modalities from having identical scores):

```
L_div = λ_div * ||Cov([r_imu, r_thm, r_tof])||_F
```

**Sparsity regularization** (optional; encourage selective modality use):

```
L_sparse = λ_sparse * Σ_m ||r_m||_1
```

### 7.5 Reliability Score Interpretation

| Score Range | Interpretation |
|---|---|
| r ≈ 1.0 | Modality is highly informative; signal quality is high |
| r ≈ 0.5 | Uncertain; modality provides ambiguous information |
| r ≈ 0.0 | Modality is uninformative or corrupted; effectively dropped |

---

## 8. Adaptive Fusion Equations

### 8.1 Reliability-Gated Cross-Modal Attention

The Adaptive Fusion Transformer (AFT) extends standard multi-head attention with reliability gating.

**Standard Multi-Head Attention:**

```
Attn(Q, K, V) = softmax(QK^T / sqrt(D_k)) V
```

**Reliability-Biased Attention:**

For a query from modality `m_q` attending to a key-value from modality `m_k`:

```
Attn_rel(Q_mq, K_mk, V_mk) = softmax((Q_mq K_mk^T / sqrt(D_k)) + log(r_mk + ε)) V_mk
```

The `log(r_mk + ε)` term acts as a **log-domain bias**:
- High reliability `r_mk → 1` → bias ≈ 0 → attention unaffected
- Low reliability `r_mk → 0` → bias → -∞ → attention weight → 0

This is mathematically equivalent to masking but fully differentiable.

### 8.2 Full AFT Forward Pass

**Step 1: Concatenate modality embeddings**

```
H_cat = Concat(Ĥ_imu, Ĥ_thm, Ĥ_tof)     H_cat ∈ ℝ^{B×3T×D}
```

**Step 2: Construct reliability bias matrix**

```
R_bias ∈ ℝ^{B×3T×1}     (concatenate per-timestep reliability scores)
Bias_attn ∈ ℝ^{B×3T×3T} = R_bias · R_bias^T  (reliability compatibility matrix)
```

Or alternatively, use additive bias directly on the attention logit matrix.

**Step 3: Apply reliability-biased self-attention**

```
H_fused_seq = TransformerEncoder(H_cat, attn_bias=log(Bias_attn + ε))
```

**Step 4: Temporal pooling**

```
H_fused = MeanPool(H_fused_seq)     H_fused ∈ ℝ^{B×D}
```

Or use a learned [CLS] token prepended to the sequence (BERT-style).

### 8.3 Modality-Level Fusion Alternative (Simpler Variant)

If per-timestep attention is too expensive:

**Aggregate each modality to a sequence-level embedding:**

```
e_m = MeanPool(H_m)     e_m ∈ ℝ^{B×D}
```

**Reliability-weighted sum:**

```
R_m = MeanPool(r_m)     R_m ∈ ℝ^{B×1}   (scalar reliability per modality)
H_fused = (R_imu * e_imu + R_thm * e_thm + R_tof * e_tof) / (R_imu + R_thm + R_tof)
```

This is a simpler baseline variant (ARST-SimpleWeighted) used in ablation.

### 8.4 Classification Head

```
H_fused ∈ ℝ^{B×D}
z = W_cls * LayerNorm(H_fused) + b_cls     z ∈ ℝ^{B×D_cls}
z = GeLU(z)
z = Dropout(z, p=0.1)
ẑ = W_out * z + b_out                      ẑ ∈ ℝ^{B×C}
ŷ = Softmax(ẑ)
```

---

## 9. Planned Experiments

### 9.1 Phase 2 — Baseline Models

| ID | Model | Fusion | Notes |
|---|---|---|---|
| B-MLP | MLP | Concatenation | Flat feature vector; no temporal modeling |
| B-CNN | 1D CNN | Concatenation | Multi-scale convolution over each modality |
| B-LSTM | BiLSTM | Concatenation | Bidirectional recurrent; handles variable length |
| B-TF | Transformer | Concatenation | Full self-attention baseline |

### 9.2 Phase 3 — Sensor-Specific Encoders

| ID | Variant | Notes |
|---|---|---|
| E-IMU | IMU Encoder only | Unimodal upper bound for IMU |
| E-THM | Thermal Encoder only | Unimodal upper bound for thermal |
| E-TOF | ToF Encoder only | Unimodal upper bound for ToF |
| E-ALL-CONCAT | All encoders + concat | Multimodal without fusion transformer |

### 9.3 Phase 5 — ARST Variants

| ID | Model | Notes |
|---|---|---|
| ARST-NoRel | ARST without reliability heads | Tests if reliability adds value |
| ARST-StaticW | ARST with fixed equal weights | Reliability = 1.0 everywhere |
| ARST-LearnedW | ARST with global learned weights | Static per-modality weights |
| ARST-SigmoidRel | ARST with sigmoid reliability | Independent per-modality |
| ARST-SoftmaxRel | ARST with softmax reliability | Competitive across modalities |
| ARST-Full | Full ARST (primary model) | Best configuration |

### 9.4 Phase 6 — Missing Modality

| ID | Missing Modality | Notes |
|---|---|---|
| MM-NoIMU | IMU dropped | Zero-fill or learned null embedding |
| MM-NoThm | Thermal dropped | |
| MM-NoToF | ToF dropped | |
| MM-TwoMissing | Only IMU remains | Extreme degradation case |

### 9.5 Phase 7 — Ablations

| Ablation Axis | Variants |
|---|---|
| Encoder type | CNN / LSTM / Transformer per modality |
| Fusion type | Concat / Sum / Attention / ARST |
| Reliability head | None / Sigmoid / Softmax / Learned threshold |
| Reliability loss | None / Entropy / Diversity / Combined |
| Window size | 64 / 128 / 256 / 512 timesteps |
| Embedding dim | 64 / 128 / 256 / 512 |
| Transformer depth | L = 1 / 2 / 4 / 6 |

---

## 10. Evaluation Metrics

### 10.1 Primary Metric

**Macro F1-Score:**

```
F1_macro = (1/C) Σ_c F1_c = (1/C) Σ_c [2 * P_c * R_c / (P_c + R_c)]
```

Macro averaging weights all classes equally, making it robust to class imbalance.

### 10.2 Secondary Metrics

| Metric | Formula / Description |
|---|---|
| Balanced Accuracy | Mean of per-class recall |
| Per-Class F1 | F1 score for each behavior class separately |
| AUROC (OvR) | Area under ROC curve; one-vs-rest per class |
| ECE | Expected Calibration Error; measures probability calibration |
| Confusion Matrix | Full C×C confusion matrix (normalized) |
| Reliability Correlation | Correlation between reliability scores and per-sample correctness |

### 10.3 Reliability-Specific Metrics

- **Reliability-Accuracy Correlation (RAC):** `corr(R_m, correct)` — high r_m should correlate with correct predictions
- **Reliability Under Corruption (RUC):** How reliability scores change when sensor is artificially corrupted
- **Modality Attribution Agreement:** Comparison with SHAP-based modality attribution

---

## 11. Ablation Strategy

### 11.1 Design Principles

1. **Single-factor ablations first:** Vary one component at a time to isolate its contribution.
2. **Factorial design for interactions:** Cross-product of key hyperparameter combinations.
3. **Bootstrap confidence intervals:** Report mean ± 95% CI over 5 random seeds.
4. **Fixed evaluation protocol:** Same train/val/test split across all ablations.

### 11.2 Ablation Tree

```
ARST-Full
├── Remove reliability heads → ARST-NoRel
│   └── Remove adaptive fusion → ARST-NoFusion (= B-TF baseline)
├── Change reliability activation (sigmoid → softmax) → ARST-SoftmaxRel
├── Remove reliability loss (L_rel = 0) → ARST-NoRelLoss
├── Change fusion type
│   ├── Mean pooling → ARST-MeanFusion
│   ├── Concatenation → ARST-ConcatFusion
│   └── Weighted sum (static) → ARST-StaticWeights
├── Change encoder type
│   ├── IMU: CNN → ARST-IMUCNN
│   ├── IMU: LSTM → ARST-IMULSTM
│   └── All: shared weights → ARST-SharedEncoders
└── Missing modality
    ├── Zero imputation → ARST-ZeroFill
    └── Learned null embedding → ARST-NullEmb
```

### 11.3 Statistical Testing

- Use paired t-test (α=0.05) for comparing model variants
- Report Cohen's d effect size
- Bonferroni correction for multiple comparisons in large ablation suites

---

## 12. Future Work

### 12.1 Architectural Extensions

1. **Hierarchical Reliability:** Estimate reliability at multiple temporal scales (fine/coarse).
2. **Inter-Modality Reliability Propagation:** Allow one modality's reliability to inform another's (e.g., if IMU is noisy but ToF is clean, boost ToF reliability).
3. **Uncertainty-Aware Reliability:** Replace point estimate with a distribution over reliability (via normalizing flows or deep ensembles).
4. **Temporal Reliability Smoothing:** Apply exponential moving average to reliability scores to reduce jitter.

### 12.2 Training Improvements

1. **Curriculum Learning:** Start with high-reliability samples; gradually include degraded samples.
2. **Auxiliary Self-Supervised Tasks:** Predict masked timesteps per modality (MAE-style pre-training).
3. **Contrastive Reliability:** Train reliability head to score clean and corrupted versions differently.
4. **Knowledge Distillation:** Distill ARST-Full into a single-modality student for edge deployment.

### 12.3 Dataset Extensions

1. **Cross-Dataset Transfer:** Test ARST trained on CMI data on other HAR datasets (PAMAP2, OPPORTUNITY, USC-HAD).
2. **Synthetic Corruption Augmentation:** Augment training data with realistic sensor failures (spike injection, zero-out, Gaussian noise).
3. **Multi-Subject Generalization:** Leave-one-subject-out evaluation to test personalization vs. generalization.

### 12.4 Deployment Path

1. **ONNX Export:** Export encoder + fusion module for edge deployment.
2. **TensorRT Optimization:** For NVIDIA embedded platforms (Jetson).
3. **Streaming Inference:** Implement causal (non-future-looking) encoder variants.
4. **Quantization:** INT8 post-training quantization with reliability-aware calibration.
