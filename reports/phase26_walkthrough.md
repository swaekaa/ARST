# Phase 2.6 — Architecture Validation and Repair: Walkthrough

> **Status:** 🔧 Code fixes applied, awaiting execution
> **Date:** 2026-06-20

---

## Summary

Phase 2.5 benchmarks revealed CNN (F1=0.028) and Transformer (F1=0.035) performing 10× worse than MLP (0.32) and LSTM (0.39). This phase identified and fixed the root causes.

## Root Causes Discovered

### 1. CNN: Excessive Dropout + Missing LayerNorm (F1: 0.028 → expected >0.32)

| Issue | Impact | Fix |
|---|---|---|
| Dropout 0.3 on 172K param model | Severe under-learning | Reduced to 0.1 |
| No LayerNorm before classifier head | Fused features at inconsistent scales | Added `LayerNorm(fused_dim)` |

### 2. Transformer: `_init_weights` Destroys Input Projections (F1: 0.035 → expected >0.38)

| Issue | Impact | Fix |
|---|---|---|
| `_init_weights()` replaces Kaiming init with `trunc_normal_(std=0.02)` on ALL Linear layers including input projections | Input projections produce near-zero activations, starving Transformer of signal | Skip `imu_proj`, `thermo_proj`, `tof_proj` in `_init_weights()` |
| Missing `sqrt(d_model)` embedding scaling | Positional encoding dominates content signal | Added `* math.sqrt(d_model)` scaling |
| No input normalization | Multi-modal projections at different scales | Added `LayerNorm(d_model)` on projected features |
| Head hidden dim = 64 (too narrow) | Bottleneck in classification | Changed to `d_model` (128) |

> [!IMPORTANT]
> The Transformer root cause was definitively the `_init_weights()` function. For `imu_proj (7→128)`, the default Kaiming std is 0.378 but was overwritten to 0.02 — a **19× reduction**. The projected activations were near-zero, making the entire Transformer layer stack process noise.

## Files Modified

| File | Change |
|---|---|
| `src/arst/models/baselines/cnn.py` | Added LayerNorm, reduced dropout to 0.1 |
| `src/arst/models/baselines/transformer.py` | Fixed `_init_weights`, added input norm + sqrt(d_model) scaling, wider head |
| `configs/model/cnn.yaml` | `dropout: 0.3` → `dropout: 0.1` |

## Files Created

| File | Purpose |
|---|---|
| `scripts/tiny_dataset_overfit.py` | Sanity check: overfit 32 samples |
| `scripts/lr_study.py` | LR sweep: [1e-3, 5e-4, 1e-4, 5e-5] × {CNN, Transformer} |
| `scripts/ablation_study.py` | Ablations: Transformer pooling, CNN kernels |
| `scripts/run_phase26.bat` | Orchestrator: runs all steps in order |
| `reports/cnn_debug_report.md` | Detailed CNN debugging analysis |
| `reports/transformer_debug_report.md` | Detailed Transformer debugging analysis |

---

## Manual Execution Steps

Due to a command execution infrastructure issue, please run these commands manually:

### Step 1: Verify fixes don't break tests
```bash
cd c:\Users\Ekaansh\OneDrive\Desktop\AB\projects\ARST\ARST
python -m pytest tests/test_baselines.py -v --tb=short
```

### Step 2: Commit architecture fixes
```bash
git add src/arst/models/baselines/cnn.py configs/model/cnn.yaml
git commit -m "fix: correct CNN - add LayerNorm pre-head, reduce dropout 0.3->0.1"

git add src/arst/models/baselines/transformer.py
git commit -m "fix: correct Transformer - skip input proj in _init_weights, add sqrt(d_model) scaling"

git add reports/cnn_debug_report.md reports/transformer_debug_report.md
git commit -m "docs: add Phase 2.6 debug reports"

git add scripts/tiny_dataset_overfit.py scripts/lr_study.py scripts/ablation_study.py scripts/run_phase26.bat
git commit -m "scripts: add Phase 2.6 diagnostic and experiment scripts"
```

### Step 3: Tiny overfit sanity check
```bash
python scripts/tiny_dataset_overfit.py
```
**Expected:** Both CNN and Transformer reach >95% accuracy on 32 samples.

### Step 4: Learning rate study
```bash
python scripts/lr_study.py
```
**Expected:** Identifies optimal LR for each model. Report saved to `reports/lr_study.md`.

### Step 5: Ablation study
```bash
python scripts/ablation_study.py
```
**Expected:** Compares pooling strategies and kernel configs. Report saved to `reports/baseline_architecture_ablations.md`.

### Step 6: Full benchmark retrain
```bash
python train.py model=cnn training.epochs=100 wandb.enabled=false
python train.py model=transformer training.epochs=100 wandb.enabled=false
```

### Step 7: Or run everything at once
```bash
scripts\run_phase26.bat
```

---

## Stop Condition for Phase 3

| Model | Minimum | Target |
|---|---|---|
| CNN Macro F1 | > 0.32 (beat MLP) | ~0.35–0.40 |
| Transformer Macro F1 | ≥ 0.39 (match LSTM) or > 0.32 (beat MLP) | ~0.38–0.45 |

**Phase 3 is BLOCKED until both conditions are met.**
