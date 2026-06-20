# Phase 2.6 — Transformer Debug Report

> **Status:** 🔧 Fix Applied
> **Date:** 2026-06-20

---

## Architecture Audit

### Input Projection Dimensions ✅

| Projection | Shape |
|---|---|
| `imu_proj` | `Linear(7 → 128)` |
| `thermo_proj` | `Linear(5 → 128)` |
| `tof_proj` | `Linear(320 → 128)` |

All projections correctly map to `d_model=128`.

### Positional Encoding ✅

- Uses `SinusoidalPositionalEncoding` from `arst.models.encoders.imu_encoder`
- Operates on `[B, T, d_model]` — correct shape ✅
- `max_len=5000` — sufficient for T=193 ✅

### Sequence Ordering ✅

- Modalities concatenated along time axis: `[B, T*3, 128]` = `[B, 192, 128]`
- CLS token prepended: `[B, 193, 128]`
- Order is consistent (always IMU → Thermo → ToF)

### Attention Mask ✅

- No padding mask needed (fixed-length sequences)
- No causal mask needed (classification, not generation)

### Pooling Strategy ✅

- CLS pooling: `x[:, 0, :]` correctly extracts CLS token output
- Mean pooling: `x.mean(dim=1)` correctly averages over all tokens

### Pre-LN Configuration ✅

- `norm_first=True` in `nn.TransformerEncoderLayer` — correct for Pre-LN
- Final `self.norm = nn.LayerNorm(d_model)` applied after encoder — correct (ViT-style)

### batch_first=True ✅

- Set correctly in `nn.TransformerEncoderLayer`
- All tensor operations assume `[B, T, d_model]` — consistent

### Output Shape ✅

| Operation | Shape |
|---|---|
| After concat | `[B, 192, 128]` |
| After CLS prepend | `[B, 193, 128]` |
| After Transformer | `[B, 193, 128]` |
| CLS pooling | `[B, 128]` |
| After head | `[B, 4]` ← correct |

---

## Root Causes Identified

### 1. `_init_weights()` Destroys Input Projections — CRITICAL 🔴

**This was the primary root cause of the Transformer collapse (F1 = 0.035).**

The `_init_weights()` method iterated over ALL `nn.Linear` layers and applied `trunc_normal_(std=0.02)`. This includes the input projection layers (`imu_proj`, `thermo_proj`, `tof_proj`).

**Why this is catastrophic:**

PyTorch's default initialization for `nn.Linear` is Kaiming uniform:
```
std ≈ 1 / sqrt(fan_in)
```

For `tof_proj (320→128)`:
- **Default Kaiming std:** `1/sqrt(320) ≈ 0.056`
- **After _init_weights:** `std = 0.02` (2.8× smaller)

For `imu_proj (7→128)`:
- **Default Kaiming std:** `1/sqrt(7) ≈ 0.378`
- **After _init_weights:** `std = 0.02` (19× smaller!)

The input projections need larger weights to transform raw sensor values (which span different ranges) into the `d_model` embedding space. With `std=0.02`, the projected activations are near-zero, effectively starving the Transformer of meaningful input signal. The model then collapses to predicting a uniform distribution.

**Fix:** Modified `_init_weights()` to skip `imu_proj`, `thermo_proj`, and `tof_proj`. These layers now keep their default Kaiming initialization.

### 2. Missing `sqrt(d_model)` Scaling — IMPORTANT ⚠️

The standard Transformer architecture (Vaswani et al., "Attention Is All You Need") scales embeddings by `sqrt(d_model)` before adding positional encoding:

```
x = x * sqrt(d_model) + PE(x)
```

Without this scaling, the sinusoidal positional encoding (which has values in `[-1, 1]`) can dominate the actual content embeddings, especially when the content embeddings are small (as they were due to issue #1).

With `d_model=128`, the scale factor is `sqrt(128) ≈ 11.3`, which ensures content embeddings are significantly larger than positional encodings.

**Fix:** Added `self._embed_scale = math.sqrt(d_model)` and `x = self.input_norm(x) * self._embed_scale` in the forward pass.

### 3. Missing Input LayerNorm — MODERATE ⚠️

Three different modality projections produce features at different scales. A shared `LayerNorm(d_model)` normalizes these before feeding to the Transformer.

**Fix:** Added `self.input_norm = nn.LayerNorm(d_model)` applied after concatenation, before scaling.

### 4. Head Hidden Dim Too Small — MINOR

The original head used `d_model // 2 = 64` as hidden dimension. This is unnecessarily narrow for a 4-class problem with 128-dim input.

**Fix:** Changed head hidden dim from `d_model // 2` to `d_model` (128).

---

## Changes Applied

### File: `src/arst/models/baselines/transformer.py`

```diff
+import math
+
 # Per-modality input projections
+# NOTE: These use default Kaiming init — _init_weights() must NOT
+# override them with trunc_normal_ (that was the Phase 2.5 bug).

+# Shared input norm
+self.input_norm = nn.LayerNorm(d_model)
+self._embed_scale = math.sqrt(d_model)

 # Classification head
 self.head = nn.Sequential(
-    nn.Linear(d_model, d_model // 2),
+    nn.Linear(d_model, d_model),
     nn.GELU(),
     nn.Dropout(dropout),
-    nn.Linear(d_model // 2, num_classes),
+    nn.Linear(d_model, num_classes),
 )

 def _init_weights(self) -> None:
+    # Collect input projection layers that must be skipped
+    skip_modules = set()
+    for name in ("imu_proj", "thermo_proj", "tof_proj"):
+        if hasattr(self, name):
+            skip_modules.add(id(getattr(self, name)))
+
     for module in self.modules():
+        if id(module) in skip_modules:
+            continue  # keep Kaiming init for input projections
         if isinstance(module, nn.Linear):
             nn.init.trunc_normal_(module.weight, std=0.02)

 # In forward():
+x = self.input_norm(x) * self._embed_scale
```

---

## Verification Plan

1. **Tiny overfit test (32 samples):** Transformer should reach >95% training accuracy
2. **Full benchmark:** Transformer Macro F1 should match or exceed LSTM (0.39)
