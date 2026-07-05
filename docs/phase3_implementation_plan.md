# Phase 3 Implementation Plan — Modality-Specific Encoders

## Goal

Replace the placeholder encoders with **production-quality, Phase-1-verified modality-specific encoders** for IMU (7 channels), Thermopile (5 channels), and ToF (320 channels + mask). Validate each encoder unimodally, then combine via simple concat fusion to establish the multimodal encoder baseline that Phase 4 (Reliability Module) will build upon.

---

## Background Context

### Current State

The existing encoder files in `src/arst/models/encoders/` are **pre-Phase-1 placeholders** with critical dimension mismatches:

| Encoder | Current `in_channels` | Required `in_channels` | Status |
|---|---|---|---|
| `imu_encoder.py` | `6` (docstring says acc + gyro) | **`7`** (acc + quaternion) | ❌ Wrong dim |
| `thermal_encoder.py` | `64` (8×8 grid) | **`5`** (linear array) | ❌ Wrong dim |
| `tof_encoder.py` | `64` (8×8 grid) | **`320`** (5 sensors × 64 pixels) | ❌ Wrong dim |
| `arst.py` (L78-80) | Hardcoded `in_channels=6,64,64` | **`7,5,320`** | ❌ Wrong dims |

Additionally, `arst_full.yaml` config has `in_channels: 6, 64, 64` instead of `7, 5, 320`.

Existing `test_arst.py` creates dummy batches with shapes `[B, T, 6]`, `[B, T, 64]`, `[B, T, 64]` — all wrong.

### Validated Baseline Performance (Phase 2.8)

| Model | Macro F1 | Notes |
|---|---|---|
| MLP | 0.3179 | Current best baseline |
| Majority | 0.2199 | Chance baseline |

**Phase 3 success criterion**: Multimodal encoder model must exceed MLP (Macro F1 > 0.32).

---

## Items Requiring Review

### Breaking Change to ARST Model
The `ARSTModel` class currently hardcodes `in_channels=6, 64, 64`. This will be updated to `7, 5, 320`. The existing `test_arst.py` will also be rewritten with correct dimensions.

### Embedding Dimension: D=128 vs D=256
The architecture docs specify `D=256`, but the RTX 3050 has 4GB VRAM. I propose starting with `D=128` for Phase 3 encoder validation, then scaling to `D=256` only after confirming memory fits in Phase 5. The config will be parameterised so this is a one-line change.

---

## Open Questions

### Q1: ToF Encoder — Per-Sensor vs. Flat Processing
The 320 ToF channels are structured as 5 sensors × 64 pixels. Two approaches:
- **Option A (Recommended)**: Per-sensor decomposition — reshape `[B,T,320]` → `[B,T,5,64]`, process each 64-pixel bank with a shared 1D-CNN, then aggregate across sensors. This respects the physical structure and reduces parameter count.
- **Option B**: Flat projection — treat all 320 channels as an unstructured vector, project directly to `D`. Simpler but ignores sensor structure.

I recommend **Option A** as it is more physically grounded. Does this seem right?

### Q2: Training Infrastructure
Phase 2 uses `build_csv_loaders()` (CSV-backed, includes Phase 2.8 normalization fix). Phase 3 can continue using this same loader since HDF5 preprocessing isn't built yet. I plan to reuse the existing CSV-backed infrastructure for Phase 3 training. Is this acceptable?

---

## Proposed Changes

### Component 1: IMU Encoder

**File**: `src/arst/models/encoders/imu_encoder.py`  
**Change**: `in_channels` default 6 → **7**, docstring "acc + gyro" → "acc + quaternion"

Architecture (keep existing, just fix dimension):
- Input `[B, T, 7]` → Linear projection → Multi-scale CNN (k=3,7,15) → LayerNorm → PositionalEncoding → TransformerEncoder(L=2, H=4) → Output `[B, T, D]`

**Justification**: Multi-scale CNN captures different motion frequencies (jerk, gesture trajectory, posture). Transformer adds temporal context.  
**Cost**: ~600K params at D=128; ~2M at D=256

---

### Component 2: Thermal Encoder

**File**: `src/arst/models/encoders/thermal_encoder.py`  
**Change**: `in_channels` default 64 → **5**, docstring "8×8 grid" → "5-channel linear array"

Architecture (keep existing, just fix dimension):
- Input `[B, T, 5]` → Linear(5→D) → LayerNorm → PositionalEncoding → TransformerEncoder(L=2, H=4) → Output `[B, T, D]`

**Justification**: Only 5 channels — linear projection + Transformer is the natural choice.  
**Cost**: ~400K params at D=128

---

### Component 3: ToF Encoder — Complete Redesign

**File**: `src/arst/models/encoders/tof_encoder.py`  
**Change**: Complete redesign from flat 64→320, per-sensor decomposition

New architecture:
```
Input: X_tof [B, T, 320], M_tof [B, T, 320]
  → Reshape to [B, T, 5, 64] (5 sensor banks)
  → Per-sensor: null embedding for invalid pixels + Linear(64 → D_sensor) [shared weights]
  → Validity-weighted aggregation across 5 sensors → [B, T, D_sensor]
  → Linear(D_sensor → D) + LayerNorm
  → PositionalEncoding → TransformerEncoder(L=2)
  → Output [B, T, D]
```

**Justification**: Respects physical 5×64 structure; shared weights reduce params; validity-weighted aggregation downweights noisy sensors.  
**Cost**: ~800K params at D=128 (D_sensor=32)

---

### Component 4: ARST Model Integration

**File**: `src/arst/models/arst.py`  
**Change**: Fix hardcoded `in_channels=6,64,64` → use `config.get("in_channels", 7/5/320)`. Update `num_classes` default 10→4.

---

### Component 5: Encoder Wrapper Models

**New file**: `src/arst/models/encoder_models.py`

Four wrapper models for unimodal/multimodal training:
- `IMUEncoderModel`: IMUEncoder + mean-pool + classification head → `[B, C]`
- `ThermalEncoderModel`: ThermalEncoder + mean-pool + classification head → `[B, C]`
- `ToFEncoderModel`: ToFEncoder + mean-pool + classification head → `[B, C]`
- `ConcatEncoderModel`: All three encoders + concat + classification head → `[B, C]`

All accept the standard batch format and return logits, compatible with existing `Trainer`.

**File**: `src/arst/models/registry.py`  
**Change**: Register 4 new models: `imu_encoder`, `thermal_encoder`, `tof_encoder`, `concat_encoder`

---

### Component 6: Configuration Files

**New files**:
- `configs/model/imu_encoder.yaml`
- `configs/model/thermal_encoder.yaml`
- `configs/model/tof_encoder.yaml`
- `configs/model/concat_encoder.yaml`

**Modified**: `configs/model/arst_full.yaml` — fix `in_channels` values

---

### Component 7: Tests

**New file**: `tests/models/test_encoders.py`
- Shape tests for all 3 encoders
- Gradient flow tests
- NaN/Inf guard tests
- ToF mask handling tests
- Unimodal wrapper model tests
- Concat model test

**Modified**: `tests/models/test_arst.py` — fix dummy batch dimensions

---

### Component 8: Phase 3 Training Script

**New file**: `scripts/train_phase3.py`

Orchestrates: unimodal training (E-IMU, E-THM, E-TOF) + multimodal concat (E-ALL-CONCAT), saves to `results/phase3/`.

---

### Component 9: Documentation & Reports

**Modified**: README.md, ARCHITECTURE.md, TASKS.md, DEVELOPMENT_ROADMAP.md  
**New**: `reports/phase3_encoder_report.md`

---

## Verification Plan

### Automated Tests
```bash
python -m pytest tests/models/test_encoders.py -v
python -m pytest tests/models/test_arst.py -v
python -m pytest tests/ -v
```

### Manual Verification
1. **Tiny overfit**: 32 samples, 100 epochs → ~100% train accuracy
2. **Unimodal training**: 100 epochs each → check val Macro F1
3. **Multimodal concat**: 100 epochs → must beat MLP (F1 > 0.32)
4. **Memory profiling**: Peak GPU memory on RTX 3050

### Expected Results

| Model | Expected Macro F1 | Rationale |
|---|---|---|
| E-IMU (unimodal) | 0.20–0.35 | IMU captures motion well |
| E-THM (unimodal) | 0.15–0.25 | Only 5 channels |
| E-TOF (unimodal) | 0.10–0.25 | ~59% invalid data |
| E-ALL-CONCAT | 0.30–0.45 | Multimodal advantage |
| MLP (Phase 2) | 0.3179 | Baseline to beat |

---

## Implementation Order

1. Fix encoder dimensions (IMU, Thermal, ToF)
2. Redesign ToF encoder with per-sensor decomposition
3. Create encoder wrapper models
4. Create Hydra configs
5. Fix `arst.py` hardcoded dims
6. Write unit tests
7. Run tests
8. Fix `test_arst.py`
9. Tiny overfit sanity tests
10. Full unimodal training runs
11. Multimodal concat training run
12. Generate reports
13. Update documentation
14. Commit and push
