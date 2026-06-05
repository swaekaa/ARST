# ARST Phase 2 — CUDA Diagnosis Report

## System Information

| Item | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.11.5 |
| PyTorch (before fix) | 2.12.0+cpu |
| CUDA (torch.version.cuda) | None (CPU-only build) |
| torch.cuda.is_available() | False |
| GPU (from nvidia-smi) | NVIDIA GeForce RTX 3050 Laptop |
| Driver Version | 576.88 |
| CUDA Runtime (driver) | 12.9 |
| VRAM | 4096 MiB (4 GB) |

---

## Root Cause

A **CPU-only PyTorch wheel** (`torch==2.12.0+cpu`) was installed.

The CUDA runtime version reported by the NVIDIA driver is 12.9, which is
**forward compatible** with wheels built against CUDA 12.1. The GPU and drivers
are fully functional — only the PyTorch build was wrong.

Confirming diagnostics:
```
PyTorch: 2.12.0+cpu
CUDA available: False
CUDA version (torch): None    <-- CPU-only wheel has no CUDA
Device count: 0
nvidia-smi: RTX 3050 visible, driver 576.88, CUDA 12.9
```

This is a common issue when `pip install torch` is run without specifying
`--index-url`, causing pip to select the CPU-only wheel from PyPI.

---

## Device Logic Audit

### `experiment.py` — `get_device()`
The function correctly uses:
```python
if torch.cuda.is_available():
    device = torch.device("cuda")
```
The logic is **correct**. CUDA was unavailable only because the wrong wheel was installed.

### `trainer.py`
```python
self.model = model.to(device)
self.mixed_precision = mixed_precision and torch.cuda.is_available()
self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
```
- Model is moved to device correctly.
- AMP is gated on `cuda.is_available()` — correct.
- `GradScaler` updated to new API (was using deprecated `torch.cuda.amp.GradScaler`).

All batch tensors are moved with `.to(self.device, non_blocking=True)` — correct.

---

## Fix Applied

### Reinstall PyTorch with CUDA 12.1 support

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

The CUDA 12.1 build is **compatible with the installed CUDA 12.9 driver** due to
NVIDIA's backward-compatible driver ABI.

### After reinstall, expected output:
```
PyTorch: 2.x.x+cu121
CUDA available: True
CUDA version (torch): 12.1
GPU: NVIDIA GeForce RTX 3050 Laptop (4.0 GB VRAM)
Selected device: cuda
```

---

## AMP Deprecation Fix

**Old (deprecated in PyTorch 2.x):**
```python
from torch.cuda.amp import GradScaler
self.scaler = GradScaler(enabled=self.mixed_precision)
```

**New (correct for PyTorch 2.3+):**
```python
_scaler_device = "cuda" if self.mixed_precision else "cpu"
self.scaler = torch.amp.GradScaler(_scaler_device, enabled=self.mixed_precision)
```

---

## Startup Diagnostics Added

`experiment.py` now logs on every run:
```
PyTorch version  : 2.x.x+cu121
CUDA version     : 12.1
CUDA available   : True
Device count     : 1
GPU              : NVIDIA GeForce RTX 3050 Laptop  (4.0 GB VRAM)
Selected device  : cuda
```

If CPU-only PyTorch is detected, an actionable install command is printed to stderr.

---

## Memory Budget for RTX 3050 4GB

| Component | Estimated VRAM |
|---|---|
| MLPBaseline (337K params) | ~0.05 GB |
| CNN / LSTM baselines | ~0.1-0.3 GB |
| Transformer baseline (d=128, L=2) | ~0.4 GB |
| Batch of 32, T=64 | ~0.15 GB |
| AdamW optimizer states | ~0.2-0.5 GB |
| AMP fp16 activations | ~0.1-0.2 GB |
| **Total (MLP run)** | **~0.5 GB** |
| **Total (Transformer run)** | **~1.5 GB** |

All baselines should fit within 4 GB with `accumulation_steps=4` and `mixed_precision=True`.

---

## Remaining Risks

1. **torchaudio version mismatch**: The existing `torchaudio==2.5.1+cu121` may not
   match the newly installed `torch` version. Run `pip list | grep torch` to verify
   all torch packages share the same CUDA suffix.

2. **`num_workers > 0` on Windows**: DataLoader multiprocessing can cause issues on
   Windows. Keep `num_workers=0` (already the default in `configs/data/default.yaml`).

3. **AMP with fp16 on RTX 3050**: Some ops may produce fp16 overflow for large
   Thermal values (~30°C = 30.0, which is safe in fp16 range of ±65504).
   Monitor for overflow warnings if they appear.
