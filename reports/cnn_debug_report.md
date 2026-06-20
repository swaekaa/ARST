# Phase 2.6 — CNN Debug Report

> **Status:** 🔧 Fix Applied
> **Date:** 2026-06-20

---

## Architecture Audit

### Tensor Shape Analysis ✅

| Operation | Shape |
|---|---|
| Input (imu) | `[B, T, 7]` |
| Input (thermo) | `[B, T, 5]` |
| Input (tof) | `[B, T, 320]` |
| After `.permute(0,2,1)` | `[B, F, T]` ← correct for Conv1d |
| After Conv1d(k=3) + BN + GELU | `[B, 64, T]` |
| After Conv1d(k=7) + BN + GELU | `[B, 64, T]` |
| After AdaptiveAvgPool1d(1) | `[B, 64, 1]` |
| After `.squeeze(-1)` | `[B, 64]` |
| After concat (3 modalities) | `[B, 192]` |
| After head | `[B, 4]` ← correct |

**Verdict:** Tensor shapes are correct. The transpose `[B,T,F] → [B,F,T]` for Conv1d is properly handled.

### Kernel Sizes ✅

- Kernel 3 with `padding=1`: preserves time dimension ✅
- Kernel 7 with `padding=3`: preserves time dimension ✅
- No dimension mismatch possible.

### Pooling ✅

- `AdaptiveAvgPool1d(1)` correctly collapses time dimension to 1.
- `.squeeze(-1)` removes the trailing dimension.

### Final Logits ✅

- Head: `Linear(192→256) → GELU → Dropout → Linear(256→4)` ✅

---

## Root Causes Identified

### 1. Excessive Dropout (0.3) — CRITICAL

The CNN baseline has only **172,356 parameters** — the smallest model in the benchmark. With 0.3 dropout applied in the classification head, combined with BatchNorm in every conv layer and Global Average Pooling (which already acts as a regularizer), the model was severely under-parameterized relative to regularization.

**Evidence:** MLP (337K params) and LSTM (2M params) both use 0.3 dropout successfully because they have 2-12× more parameters.

**Fix:** Reduced dropout from 0.3 → 0.1.

### 2. Missing LayerNorm Pre-Head — CRITICAL

The LSTM baseline (F1=0.39, the best performer) uses `nn.LayerNorm(fused_dim)` before its classification head. This normalizes the concatenated branch outputs from different modalities, ensuring consistent scale.

The CNN was missing this normalization. The three CNN branches (IMU, Thermal, ToF) operate on data with very different scales:
- IMU: 7 channels, small values (accelerometer/quaternion)
- Thermal: 5 channels, temperature values
- ToF: 320→64 channels, distance values

After GAP, the feature magnitudes can be wildly different across modalities. Without normalization, the classification head receives inputs at inconsistent scales, making learning unstable.

**Fix:** Added `nn.LayerNorm(fused_dim)` as the first layer in the classification head.

### 3. BatchNorm + AMP Interaction — MINOR

BatchNorm statistics computed in mixed precision (FP16) can be less stable for small batch sizes. This is a minor concern but could contribute to instability.

**Mitigation:** The dropout reduction and LayerNorm addition should be sufficient. No BN changes needed.

---

## Changes Applied

### File: `src/arst/models/baselines/cnn.py`

```diff
-        dropout: float = 0.3,
+        dropout: float = 0.1,

         # Classification head
         self.head = nn.Sequential(
+            nn.LayerNorm(fused_dim),
             nn.Linear(fused_dim, head_hidden_dim),
             nn.GELU(),
             nn.Dropout(dropout),
             nn.Linear(head_hidden_dim, num_classes),
         )
```

### File: `configs/model/cnn.yaml`

```diff
-dropout: 0.3
+dropout: 0.1
```

---

## Verification Plan

1. **Tiny overfit test (32 samples):** CNN should reach >95% training accuracy
2. **Full benchmark:** CNN Macro F1 should exceed MLP (0.32)
