# ARST — Technical Research Context

> **Adaptive Reliability Sensor Transformer**
> Multimodal Behavior Recognition from Wearable Sensor Streams

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Motivation](#2-research-motivation)
3. [Dataset Description](#3-dataset-description)
4. [Sensor Modality Description](#4-sensor-modality-description)
5. [Phase 1 Findings and Architectural Implications](#5-phase-1-findings-and-architectural-implications)
6. [Proposed ARST Architecture](#6-proposed-arst-architecture)
7. [Mathematical Formulation](#7-mathematical-formulation)
8. [Reliability Score Equations](#8-reliability-score-equations)
9. [Adaptive Fusion Equations](#9-adaptive-fusion-equations)
10. [Planned Experiments](#10-planned-experiments)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Ablation Strategy](#12-ablation-strategy)
13. [Future Work](#13-future-work)

---

## 1. Project Overview

ARST (Adaptive Reliability Sensor Transformer) is a research-grade deep learning system designed for **multimodal behavior recognition** using heterogeneous wearable sensor streams. The system targets the Kaggle CMI competition dataset, which contains synchronized recordings from IMU, thermopile, and time-of-flight (ToF) sensors.

The central hypothesis is that **not all sensor modalities are equally informative at every timestep**. A sensor may be degraded due to:

- Physical occlusion or poor skin contact
- Subject motion artifacts
- Environmental interference
- Hardware limitations (e.g., thermal drift, ToF ambiguity at close range or on dark/non-reflective surfaces)

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

### 2.3 Empirical Motivation (Phase 1 Findings)

Phase 1 EDA on the actual dataset (`data/raw/train.csv`) provides **quantitative justification** for the ARM design:

| Modality | Mean Missing % | Max Missing % | Primary Cause |
|---|---|---|---|
| IMU | ~0.4% | ~0.6% | Near-complete; rarely fails |
| Thermopile | ~2.1% | ~5.8% | Sensor dropout; occasional hardware fault |
| **ToF** | **~59.4%** | **~74.4%** | Invalid returns (−1.0 sentinel); non-reflective surfaces, occlusion, range limits |

> **Key insight:** ToF has ~150× more invalidity than IMU. A static fusion that treats all three modalities equally will be heavily polluted by invalid ToF readings. The ARM must learn to suppress ToF contribution at timesteps where the sensor reports are mostly invalid.

This differential invalidity rate — discovered empirically in Phase 1 — is the **primary quantitative motivation** for the Adaptive Reliability Module.

### 2.4 Connection to Literature

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

### 3.3 Actual Data Structure (Phase 1 Verified)

> **IMPORTANT:** The dataset is **not** organized as per-sequence parquet files. It is a **single flat CSV** with one row per timestep.

```
data/raw/
├── train.csv              # 574,945 rows × 341 columns — ALL sensor data + labels
├── test.csv               # Test sequences (no behavior labels)
├── sample_submission.csv  # Submission format
└── [no sensor_data/ directory — all data is in train.csv]
```

**Column schema of `train.csv`:**

| Column Group | Columns | Count | Description |
|---|---|---|---|
| Metadata | `row_id`, `sequence_type`, `sequence_id`, `sequence_counter` | 4 | Row identifiers and sequence membership |
| Subject | `subject`, `orientation` | 2 | Participant ID and body orientation |
| Labels | `behavior`, `phase`, `gesture` | 3 | Multi-level behavior annotations |
| IMU | `acc_x`, `acc_y`, `acc_z`, `rot_w`, `rot_x`, `rot_y`, `rot_z` | **7** | Accelerometer + quaternion |
| Thermopile | `thm_1`, `thm_2`, `thm_3`, `thm_4`, `thm_5` | **5** | Linear thermopile array |
| ToF | `tof_1_v0`…`tof_1_v63`, …, `tof_5_v0`…`tof_5_v63` | **320** | 5 sensors × 64 pixels each |

### 3.4 Class Distribution

The dataset contains **4 behavior classes** with significant class imbalance:

| Rank | Behavior | Notes |
|---|---|---|
| 1 | Moves hand to target location | Most frequent |
| 2 | Performs gesture | |
| 3 | Relaxes and moves hand to target location | |
| 4 | Hand at target location | Least frequent |

See `reports/class_analysis.md` for exact counts and imbalance ratio.

**Recommended mitigations:**
- Weighted loss functions (Focal Loss, class-weighted CE)
- Stratified train/val/test splits (by subject to prevent data leakage)
- Macro-averaged evaluation metrics (Macro F1 as primary)

### 3.5 Sequence Statistics (Phase 1 Verified)

| Property | Value |
|---|---|
| Total rows | 574,945 |
| Total columns | 341 |
| Total sequences | **8,151** |
| Sequence length | Variable (see distribution in `reports/sequence_analysis.md`) |
| Recommended window size | **128 timesteps** (≤ P25 of sequence lengths) |
| Recommended stride | 50% overlap (64 timesteps) |
| Overall missing % (NaN) | ~1.8% |
| ToF invalidity (−1.0 sentinel) | **~59.4% average** across 320 ToF features |

---

## 4. Sensor Modality Description

> All channel counts and feature names below are **Phase 1 verified** from the actual `train.csv` schema.

### 4.1 IMU — Inertial Measurement Unit

**Sensors:** 3-axis accelerometer + quaternion orientation (NOT gyroscope)
**Raw features:** `acc_x`, `acc_y`, `acc_z`, `rot_w`, `rot_x`, `rot_y`, `rot_z` → **7 channels**
**Key behaviors captured:** Arm movements, gesture kinematics, body posture changes

**Actual signal statistics (Phase 1):**

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| acc_x | ~0.0 | ~0.9 | variable | variable |
| acc_y | ~0.0 | ~0.9 | variable | variable |
| acc_z | ~0.0 | ~0.9 | variable | variable |
| rot_w | ~0.9 | ~0.1 | −1.0 | 1.0 |
| rot_x | ~0.0 | ~0.1 | −1.0 | 1.0 |
| rot_y | ~0.0 | ~0.1 | −1.0 | 1.0 |
| rot_z | ~0.0 | ~0.1 | −1.0 | 1.0 |

**Preprocessing pipeline:**
1. Z-score normalization per channel (data is near-complete, ~0.4% missing)
2. Quaternion-to-Euler conversion (optional — for interpretability only)
3. Windowing to T=128 timesteps

**Encoder input:** `X_imu ∈ ℝ^{B×T×7}` ← _updated from 6 in original docs_

**Reliability challenges:**
- Motion artifacts during high-velocity transitions
- Sensor saturation during high-impact gestures (clipping)
- Quaternion gimbal lock edge cases

### 4.2 Thermopile Array

**Sensors:** 5 individual infrared thermopile channels (NOT an 8×8 grid)
**Raw features:** `thm_1`, `thm_2`, `thm_3`, `thm_4`, `thm_5` → **5 channels**
**Key behaviors captured:** Thermal proximity patterns, body heat signatures near sensor

**Actual signal statistics (Phase 1):**

| Property | Value |
|---|---|
| Mean across channels | ~27°C |
| Std | ~0.4°C |
| Max observed | ~39.6°C |
| Missing % | ~2.1% average (up to ~5.8% for `thm_5`) |

**Preprocessing pipeline:**
1. Z-score normalization per channel
2. NaN imputation (linear interpolation for ~2% missing)
3. Windowing to T=128 timesteps

**Encoder input:** `X_thm ∈ ℝ^{B×T×5}` ← _updated from 64 in original docs_

**Architecture implication:** No 2D spatial CNN required — a simple linear projection or 1D temporal encoder is sufficient for 5 channels.

**Reliability challenges:**
- Thermal drift over long recording sessions
- Ambient temperature interference
- `thm_5` has highest missing rate (~5.8%) — may reflect hardware placement

### 4.3 Time-of-Flight (ToF) Sensors

**Sensors:** 5 independent ToF sensor banks, each with a 64-pixel depth array
**Raw features:** `tof_1_v0`…`tof_1_v63`, …, `tof_5_v0`…`tof_5_v63` → **320 channels**
**Key behaviors captured:** Limb proximity, gesture spatial envelope, body silhouette depth

**Invalidity encoding:**
- Invalid readings are encoded as **−1.0** (sentinel value, NOT NaN)
- Valid readings represent distance in mm (confirmed range: 0–249 mm in training data)
- A **binary validity mask** `M_tof ∈ {0,1}^{B×T×320}` is extracted at preprocessing time

**Actual invalidity statistics (Phase 1):**

| Property | Value |
|---|---|
| Mean invalidity rate (across 320 features) | **~59.4%** |
| Max invalidity rate (single feature) | **~74.4%** |
| Min invalidity rate (single feature) | **~43.9%** |
| Sensor 5 NaN missing % (hardware) | ~5.2% |

**Preprocessing pipeline:**
1. Extract binary validity mask: `M = (X_tof != -1.0).float()`
2. Replace invalid −1.0 values with 0.0: `X_tof_clean = X_tof.clamp(min=0.0)`
3. Z-score normalization on valid readings only
4. Always return `(X_tof_clean, M_tof)` together — mask is mandatory

**Encoder input:**
- `X_tof ∈ ℝ^{B×T×320}` (cleaned, invalid → 0.0) ← _updated from 64 in original docs_
- `M_tof ∈ ℝ^{B×T×320}` (validity mask: 1=valid, 0=invalid) ← _new, mandatory_

**Architecture implication:** The ToF encoder must explicitly consume the mask tensor. The ~59% invalidity rate is the **primary empirical justification** for the Adaptive Reliability Module — see §5.

**Reliability challenges:**
- Multi-path interference (reflective surfaces)
- Distance invalidity at close range (<5 cm) and far range (>2 m)
- Dark/non-reflective surfaces yield invalid returns
- Per-sensor invalidity varies independently → per-sensor reliability estimation needed

---

## 5. Phase 1 Findings and Architectural Implications

> This section documents the concrete changes to the ARST architecture based on discoveries from Phase 1 EDA (`scripts/phase1_eda.py`, commit `e502b2b`).

### 5.1 Summary of Discrepancies

| Property | Pre-Phase-1 Assumption | Phase 1 Ground Truth | Δ Impact |
|---|---|---|---|
| Storage format | Per-sequence `.parquet` files | **Single flat CSV (1.1 GB)** | Data pipeline redesigned — no parquet loader |
| IMU channels | 6 (acc + gyro) | **7 (acc + quaternion)** | Input projection: 6→7 |
| Thermopile channels | 64 (8×8 spatial) | **5 (linear array)** | Encoder architecture simplified |
| ToF channels | 64 (8×8 spatial) | **320 (5×64)** | Encoder input dim: 64→320 |
| ToF invalidity rate | Unknown | **~59% (−1.0 sentinel)** | Mask channel is mandatory; primary ARM motivation |
| Behavior classes | Unknown | **4 classes** | Classification head output: C=4 |
| Total sequences | Unknown | **8,151** | Windowing: T=128, stride=64 |

### 5.2 Encoder Dimension Updates

| Encoder | Old Input Dim | **New Input Dim** | Notes |
|---|---|---|---|
| IMU Encoder | `[B,T,6]` | **`[B,T,7]`** | `rot_w` is the 7th channel (quaternion scalar) |
| Thermal Encoder | `[B,T,64]` | **`[B,T,5]`** | Linear array; no 2D spatial processing needed |
| ToF Encoder | `[B,T,64]` | **`[B,T,320]` + `[B,T,320]` mask** | 5 sensor banks; mask is mandatory input |

### 5.3 Data Pipeline Revision

The original design assumed a `SensorDataLoader` that reads per-sequence `.parquet` files and aligns timestamps. This approach is **not applicable** to the actual dataset.

**Revised pipeline:**
```
Raw: data/raw/train.csv (flat CSV, 574,945 rows)
        │
        ▼
Chunked reading (chunksize=50,000) → Sequence grouping by `sequence_id`
        │
        ▼
Window extraction (T=128, stride=64) per sequence
        │
        ▼
Per-modality normalization:
  IMU:    z-score (7 channels)
  Therm:  z-score (5 channels) + linear interpolation of NaN
  ToF:    extract mask M=(X!=-1), zero-fill invalid, z-score on valid
        │
        ▼
Save to HDF5 (data/processed/train_windows.h5)
Structure:
  /windows/imu      [N, 128, 7]   float32
  /windows/thermo   [N, 128, 5]   float32
  /windows/tof      [N, 128, 320] float32
  /windows/tof_mask [N, 128, 320] float32
  /windows/labels   [N]           int64
```

### 5.4 Reliability Module — Empirical Justification

The Phase 1 missing data analysis provides **four quantitative arguments** for the ARM:

**Argument 1 — Differential invalidity rates (inter-modality):**
- IMU: ~0.4% missing → highly reliable
- Thermopile: ~2.1% missing → mostly reliable
- ToF: ~59.4% missing → frequently unreliable
- A static fusion would weight these equally, polluting the fused representation with ~59% invalid ToF signals.

**Argument 2 — Intra-ToF sensor variation:**
- Individual ToF sensors have invalidity rates ranging from ~44% to ~74%.
- This variation is sensor- and timestep-specific, requiring dynamic (not static) reliability estimation.

**Argument 3 — Scale mismatch:**
- IMU values: range ~ [−5, +5] (normalized accelerations + unit quaternions)
- Thermopile values: range ~ [0, 40] (degrees Celsius)
- ToF values: range ~ [0, 249] mm
- Without per-modality reliability weighting, the large-magnitude ToF values dominate naive concatenation fusion.

**Argument 4 — Behavior-dependent sensor utility:**
- For static gestures (arm held at target), IMU shows little motion → low discriminability.
- For dynamic transitions, IMU is highly informative.
- The ARM should learn this temporal, behavior-dependent reliability pattern.

### 5.5 Recommended Preprocessing Changes vs. Original Design

| Step | Original Plan | Revised Plan (Phase 1) |
|---|---|---|
| Source format | Load `.parquet` per sequence | Read flat CSV, group by `sequence_id` |
| IMU preprocessing | Bandpass filter + gravity removal | Z-score normalization (quaternion is already filtered) |
| Thermal preprocessing | Dead pixel fix + background subtraction + Gaussian blur | Z-score + linear interpolation for NaN |
| ToF preprocessing | Inpainting of masked pixels | Zero-fill + explicit binary mask channel |
| Window size | 256 (assumed) | **128** (≤ P25 of actual sequence lengths) |
| Output format | NPZ per sequence | **HDF5** (`data/processed/train_windows.h5`) |

---

## 6. Proposed ARST Architecture

### 6.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ARST FULL ARCHITECTURE                             │
│                    (Phase 1 verified dimensions)                            │
├───────────────┬──────────────────┬──────────────────────────────────────────┤
│  MODALITY 1   │    MODALITY 2    │           MODALITY 3                     │
│               │                  │                                          │
│ IMU Sequence  │ Thermo Sequence  │          ToF Sequence                    │
│ [B, T, 7]     │ [B, T, 5]        │    [B, T, 320] + mask [B, T, 320]        │
│      │        │       │          │               │                          │
│  IMU Encoder  │ Thermo Encoder   │          ToF Encoder                     │
│  (1D-CNN/TF)  │ (Linear+TF)     │     (Linear+masked-attn)                 │
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
            │  Output: C = 4        │
            └───────────┬───────────┘
                        │
                        ▼
                  Behavior Class (0–3)
```

### 6.2 Component Summary (Phase 1 Updated)

| Component | Input Shape | Output Shape | Purpose |
|---|---|---|---|
| IMU Encoder | `[B, T, 7]` | `[B, T, D]` | Temporal feature extraction from acc + quaternion |
| Thermal Encoder | `[B, T, 5]` | `[B, T, D]` | Linear feature extraction (no 2D spatial needed) |
| ToF Encoder | `[B, T, 320]` + `[B, T, 320]` mask | `[B, T, D]` | Masked feature extraction; 5 sensor banks |
| Reliability Head (×3) | `[B, T, D]` | `[B, T, 1]` | Per-timestep modality reliability score |
| Adaptive Fusion Transformer | `[B, T, D]×3 + scores×3` | `[B, D]` | Reliability-gated cross-modal fusion |
| Classification Head | `[B, D]` | `[B, 4]` | Final behavior class prediction (C=4) |

### 6.3 Encoder Architectures (Phase 1 Updated)

**IMU Encoder (Temporal)**
- Input: `[B, T, 7]` — 3 acc channels + 4 quaternion channels
- 1D CNN feature extractor: kernel sizes [3, 5, 7] → multi-scale temporal features
- Positional encoding (sinusoidal or learned)
- Stacked Transformer encoder blocks (L=4, H=8, D=256)

**Thermal Encoder (Linear-Temporal)**
- Input: `[B, T, 5]` — 5 thermopile channels
- Simple linear projection: `Linear(5 → D)`
- Temporal Transformer encoder blocks (L=2, D=256)
- No 2D spatial processing (linear array, not 8×8 grid)

**ToF Encoder (Masked Spatial-Temporal)**
- Input: `[B, T, 320]` features + `[B, T, 320]` binary mask
- Per-sensor decomposition: reshape to `[B, T, 5, 64]`, process each sensor bank
- Learned null embedding for invalid pixels (mask=0)
- Or: zero-fill + reliability head naturally learns to suppress invalid timesteps
- Linear projection per sensor: `Linear(64 → D_sensor)` → aggregate across 5 sensors → `Linear(5*D_sensor → D)`

---

## 7. Mathematical Formulation

### 7.1 Input Definitions (Phase 1 Updated)

Let the input multimodal sequence be:

```
X = {X_imu, X_thm, X_tof, M_tof}
```

Where:
- `X_imu ∈ ℝ^{B×T×7}` — IMU: batch B, timesteps T, 7 channels (acc_xyz + quaternion)
- `X_thm ∈ ℝ^{B×T×5}` — Thermopile: 5 linear channels
- `X_tof ∈ ℝ^{B×T×320}` — ToF: 5 sensors × 64 pixels (invalid pixels zero-filled)
- `M_tof ∈ {0,1}^{B×T×320}` — ToF validity mask: 1=valid, 0=invalid (−1.0 sentinel)

### 7.2 Encoder Forward Pass

For modality `m ∈ {imu, thm, tof}`:

```
H_m = Encoder_m(X_m; θ_m)     H_m ∈ ℝ^{B×T×D}
```

For ToF specifically:
```
H_tof = Encoder_tof(X_tof, M_tof; θ_tof)
```

Where `θ_m` are modality-specific encoder parameters and `D` is the shared embedding dimension.

### 7.3 Reliability Score Computation

See §8 for full equations.

### 7.4 Reliability-Gated Embeddings

```
Ĥ_m = r_m ⊙ H_m     Ĥ_m ∈ ℝ^{B×T×D}
```

Where `r_m ∈ ℝ^{B×T×1}` is broadcast multiplied (Hadamard product).

### 7.5 Adaptive Fusion

```
H_fused = AFT(Ĥ_imu, Ĥ_thm, Ĥ_tof; r_imu, r_thm, r_tof)
```

The AFT performs cross-modal attention with reliability-biased attention weights. See §9.

### 7.6 Classification

```
ẑ = ClassHead(Pooled(H_fused))     ẑ ∈ ℝ^{B×4}
ŷ = Softmax(ẑ)
```

### 7.7 Training Objective

```
L_total = L_cls + λ_rel * L_rel + λ_reg * L_reg
```

Where:
- `L_cls`: Focal Loss (recommended; class imbalance confirmed in Phase 1)
- `L_rel`: Reliability regularization (see §8.4)
- `L_reg`: L2 weight decay
- `λ_rel`, `λ_reg`: Hyperparameters

---

## 8. Reliability Score Equations

### 8.1 Reliability Head Architecture

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

**Expected behavior (from Phase 1 motivation):**
- `r_tof` should be low at timesteps where `M_tof` is predominantly 0 (invalid)
- `r_imu` should remain high across most timesteps (~0.4% missing)
- `r_thm` should be slightly lower than IMU but generally high (~2.1% missing)

### 8.2 Normalized Reliability (Optional Variant)

Apply softmax normalization across modalities to enforce a competition between modalities:

```
r̃_m^t = exp(logit_m^t) / Σ_{m'} exp(logit_{m'}^t)
```

This ensures `Σ_m r̃_m^t = 1` at each timestep — modalities compete for influence.

**Trade-off:** Softmax forces one modality to dominate; sigmoid allows all modalities to be equally trusted. Both variants will be evaluated as ablation conditions.

### 8.3 Sequence-Level Reliability Aggregation

For tasks requiring a single reliability score per modality (e.g., visualization, coarse ablations):

```
R_m = (1/T) Σ_t r_m^t     (temporal mean)
```

or

```
R_m = min_t r_m^t     (temporal minimum — conservative)
```

### 8.4 Reliability Regularization Loss

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

### 8.5 Reliability Score Interpretation

| Score Range | Interpretation |
|---|---|
| r ≈ 1.0 | Modality is highly informative; signal quality is high |
| r ≈ 0.5 | Uncertain; modality provides ambiguous information |
| r ≈ 0.0 | Modality is uninformative or corrupted; effectively dropped |

**Expected prior distribution (from Phase 1):**
- `r_imu` → biased toward 1.0 (data is nearly complete)
- `r_thm` → mostly near 1.0 with occasional drops
- `r_tof` → heavily variable; should be near 0.0 for ~59% of timesteps on average

---

## 9. Adaptive Fusion Equations

### 9.1 Reliability-Gated Cross-Modal Attention

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

### 9.2 Full AFT Forward Pass

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

### 9.3 Modality-Level Fusion Alternative (Simpler Variant)

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

### 9.4 Classification Head

```
H_fused ∈ ℝ^{B×D}
z = W_cls * LayerNorm(H_fused) + b_cls     z ∈ ℝ^{B×D_cls}
z = GeLU(z)
z = Dropout(z, p=0.1)
ẑ = W_out * z + b_out                      ẑ ∈ ℝ^{B×4}
ŷ = Softmax(ẑ)
```

---

## 10. Planned Experiments

### 10.1 Phase 2 — Baseline Models

| ID | Model | Fusion | Notes |
|---|---|---|---|
| B-MLP | MLP | Concatenation | Flat feature vector; no temporal modeling |
| B-CNN | 1D CNN | Concatenation | Multi-scale convolution over each modality |
| B-LSTM | BiLSTM | Concatenation | Bidirectional recurrent; handles variable length |
| B-TF | Transformer | Concatenation | Full self-attention baseline |

### 10.2 Phase 3 — Sensor-Specific Encoders

| ID | Variant | Notes |
|---|---|---|
| E-IMU | IMU Encoder only (`[B,T,7]`) | Unimodal upper bound for IMU |
| E-THM | Thermal Encoder only (`[B,T,5]`) | Unimodal upper bound for thermal |
| E-TOF | ToF Encoder only (`[B,T,320]+mask`) | Unimodal upper bound for ToF |
| E-ALL-CONCAT | All encoders + concat | Multimodal without fusion transformer |

### 10.3 Phase 5 — ARST Variants

| ID | Model | Notes |
|---|---|---|
| ARST-NoRel | ARST without reliability heads | Tests if reliability adds value |
| ARST-StaticW | ARST with fixed equal weights | Reliability = 1.0 everywhere |
| ARST-LearnedW | ARST with global learned weights | Static per-modality weights |
| ARST-SigmoidRel | ARST with sigmoid reliability | Independent per-modality |
| ARST-SoftmaxRel | ARST with softmax reliability | Competitive across modalities |
| ARST-Full | Full ARST (primary model) | Best configuration |

### 10.4 Phase 6 — Missing Modality

| ID | Missing Modality | Notes |
|---|---|---|
| MM-NoIMU | IMU dropped | Zero-fill or learned null embedding |
| MM-NoThm | Thermal dropped | |
| MM-NoToF | ToF dropped | Expected minimal impact given ~59% invalidity |
| MM-TwoMissing | Only IMU remains | Extreme degradation case |

### 10.5 Phase 7 — Ablations

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

## 11. Evaluation Metrics

### 11.1 Primary Metric

**Macro F1-Score:**

```
F1_macro = (1/C) Σ_c F1_c = (1/C) Σ_c [2 * P_c * R_c / (P_c + R_c)]
```

Macro averaging weights all 4 classes equally, making it robust to the class imbalance confirmed in Phase 1.

### 11.2 Secondary Metrics

| Metric | Formula / Description |
|---|---|
| Balanced Accuracy | Mean of per-class recall |
| Per-Class F1 | F1 score for each of the 4 behavior classes separately |
| AUROC (OvR) | Area under ROC curve; one-vs-rest per class |
| ECE | Expected Calibration Error; measures probability calibration |
| Confusion Matrix | Full 4×4 confusion matrix (normalized) |
| Reliability Correlation | Correlation between reliability scores and per-sample correctness |

### 11.3 Reliability-Specific Metrics

- **Reliability-Accuracy Correlation (RAC):** `corr(R_m, correct)` — high r_m should correlate with correct predictions
- **Reliability Under Corruption (RUC):** How reliability scores change when sensor is artificially corrupted
- **ToF Mask Correlation:** Correlation between `r_tof` and `mean(M_tof)` — should be positive if ARM is working correctly
- **Modality Attribution Agreement:** Comparison with SHAP-based modality attribution

---

## 12. Ablation Strategy

### 12.1 Design Principles

1. **Single-factor ablations first:** Vary one component at a time to isolate its contribution.
2. **Factorial design for interactions:** Cross-product of key hyperparameter combinations.
3. **Bootstrap confidence intervals:** Report mean ± 95% CI over 5 random seeds.
4. **Fixed evaluation protocol:** Same train/val/test split across all ablations.

### 12.2 Ablation Tree

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

### 12.3 Statistical Testing

- Use paired t-test (α=0.05) for comparing model variants
- Report Cohen's d effect size
- Bonferroni correction for multiple comparisons in large ablation suites

---

## 13. Future Work

### 13.1 Architectural Extensions

1. **Hierarchical Reliability:** Estimate reliability at multiple temporal scales (fine/coarse).
2. **Inter-Modality Reliability Propagation:** Allow one modality's reliability to inform another's (e.g., if IMU is noisy but ToF is clean, boost ToF reliability).
3. **Uncertainty-Aware Reliability:** Replace point estimate with a distribution over reliability (via normalizing flows or deep ensembles).
4. **Temporal Reliability Smoothing:** Apply exponential moving average to reliability scores to reduce jitter.
5. **Per-Sensor ToF Reliability:** Estimate reliability independently for each of the 5 ToF sensor banks (not as a single r_tof score).

### 13.2 Training Improvements

1. **Curriculum Learning:** Start with high-reliability samples; gradually include degraded samples.
2. **Auxiliary Self-Supervised Tasks:** Predict masked timesteps per modality (MAE-style pre-training).
3. **Contrastive Reliability:** Train reliability head to score clean and corrupted versions differently.
4. **Knowledge Distillation:** Distill ARST-Full into a single-modality student for edge deployment.
5. **ToF Mask Supervision:** Use the known ToF validity mask (M_tof) as a weak supervision signal for the reliability head during early training.

### 13.3 Dataset Extensions

1. **Cross-Dataset Transfer:** Test ARST trained on CMI data on other HAR datasets (PAMAP2, OPPORTUNITY, USC-HAD).
2. **Synthetic Corruption Augmentation:** Augment training data with realistic sensor failures (spike injection, zero-out, Gaussian noise).
3. **Multi-Subject Generalization:** Leave-one-subject-out evaluation to test personalization vs. generalization.

### 13.4 Deployment Path

1. **ONNX Export:** Export encoder + fusion module for edge deployment.
2. **TensorRT Optimization:** For NVIDIA embedded platforms (Jetson).
3. **Streaming Inference:** Implement causal (non-future-looking) encoder variants.
4. **Quantization:** INT8 post-training quantization with reliability-aware calibration.
