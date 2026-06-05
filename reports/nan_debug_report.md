# ARST Phase 2 — NaN Training Instability Debug Report

## Root Cause

**Primary cause: NaN propagation from raw CSV first-row extraction.**

`build_csv_loaders()` in `src/arst/data/dataloader.py` called `make_window_df()`, which extracted the feature value for each sensor column using:

```python
row[col] = window_data[col].values[0] if len(window_data) > 0 else 0.0
```

This takes only the **first row** of each sequence. The raw dataset has a **2.3% NaN rate** in Thermal and ToF columns.
The full-dataset audit confirmed that **556 out of 8,151 sequences (6.8%)** have NaN in their first row.

These NaN values were passed as-is into `float32` tensors. When the model computed `mean(dim=1)` over the time dimension (in `MLPBaseline.forward`), the NaN propagated:

```
NaN in input -> NaN in mean -> NaN in logits -> NaN in cross_entropy -> NaN loss
```

Since the loss NaN appeared from epoch 0, batch 0, no learning occurred.

---

## Evidence

### 1. Full-dataset NaN audit
```
Total sequences: 8151
Sequences with NaN in first row: 556
Rate: 6.8%
```

### 2. Column-level NaN counts (first 5000 rows)
```
thm_1     67
thm_2     67
thm_3     67
thm_4     67
thm_5    301  ← especially high
```

### 3. Propagation path
```
CSV row[0] = NaN
  -> ARSTRawCSVDataset._extract_cols -> tile NaN across [T, 5]
    -> thermo.mean(dim=1) = NaN
      -> torch.cat([...], dim=-1) = NaN in flat vector
        -> MLP forward = NaN logits
          -> F.cross_entropy = NaN loss
```

### 4. Observed symptom
```
Epoch 000/99  train_loss=nan  val_loss=nan  val_f1=0.0000
```

---

## Secondary Issues Found

### 2a. AMP GradScaler deprecated API
**File:** `src/arst/training/trainer.py`, line 28
**Old:** `from torch.cuda.amp import GradScaler`
**New:** `torch.amp.GradScaler('cuda', enabled=...)`
PyTorch 2.x emits a `FutureWarning` for the old API.

### 2b. Mixed precision silently disabled on CPU
The trainer already correctly gates AMP on CUDA availability:
```python
self.mixed_precision = mixed_precision and torch.cuda.is_available()
```
So AMP NaN (fp16 underflow) was **not** a factor — confirmed by the training logs
showing `amp=False` when running on CPU.

---

## Fix Applied

### Fix 1 — NaN root cause (`dataloader.py`)

```diff
-   row[col] = window_data[col].values[0] if len(window_data) > 0 else 0.0
+   col_vals = window_data[col].values
+   mean_val = float(np.nanmean(col_vals)) if len(col_vals) > 0 else 0.0
+   # nanmean returns nan if ALL values are nan; fall back to 0.0
+   row[col] = mean_val if np.isfinite(mean_val) else 0.0
```

`np.nanmean` skips NaN values when computing the column mean. If an entire
column is NaN (pathological case), the explicit `isfinite` guard substitutes `0.0`.

### Fix 2 — NaN guard in training loop (`trainer.py`)

Added a fail-fast assertion after the forward pass:

```python
if not torch.isfinite(loss):
    raise RuntimeError(
        f"NaN/Inf loss detected at epoch step {step}: "
        f"loss={loss.item():.6f}  "
        f"logits_nan={logits.isnan().any().item()}  "
        f"imu_nan={imu.isnan().any().item()}  ..."
    )
```

This ensures future NaN regressions are caught immediately with a diagnostic
message, rather than silently training on NaN for all 100 epochs.

### Fix 3 — Updated GradScaler API (`trainer.py`)

```diff
- from torch.cuda.amp import GradScaler
- self.scaler = GradScaler(enabled=self.mixed_precision)
+ _scaler_device = "cuda" if self.mixed_precision else "cpu"
+ self.scaler = torch.amp.GradScaler(_scaler_device, enabled=self.mixed_precision)
```

---

## Validation

After applying the fix, a quick smoke run confirmed:

- No NaN loss at any epoch
- `train_loss` decreases from epoch 0
- `val_f1_macro` increases from 0.0
- All 8,151 sequences are available with finite values

---

## Class Weights Safety Audit

```
class_weights range=[0.039, 2.024]
```

These are **inverse-frequency weights normalized so weights.sum() == n_classes**.
The ratio of 2.024 / 0.039 = 51.9× is large but not numerically unsafe because:
- `F.cross_entropy` with `weight` applies per-sample weighting to log-probabilities,
  not to raw logits, so no exponentiation amplification occurs
- Focal Loss additionally multiplies by `(1 - p_t)^gamma` which is ≤ 1.0 for any p_t

Class weights were **not** the root cause of NaN loss.

---

## Remaining Risk

The `ARSTRawCSVDataset` is a "flat-row" fallback for Phase 2 only: it tiles
a single aggregated row `T` times rather than loading true temporal sequences.
This means the model sees **identical timesteps** across all T positions.
For the Phase 3 transition, implement `ARSTSequenceDataset` (already scaffolded in
`dataset.py`) to load true 128-step windows from the CSV per sequence.
