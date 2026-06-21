# ARST Phase 2.5 — Model Efficiency Report

> Computational cost analysis for all baseline models.  
> Device: NVIDIA GPU (CUDA) / CPU fallback  
> Batch size: 32, Sequence length: T=64  

---

## Parameter Count

| Model | Trainable Params | Relative Size |
|---|---|---|
| Random | 0 (non-parametric) | — |
| Majority | 0 (non-parametric) | — |
| MLP | 337,028 | 1.00× MLP |
| CNN | 172,740 | 0.51× MLP |
| LSTM | 2,003,396 | 5.94× MLP |
| Transformer | 457,092 | 1.36× MLP |

---

## GPU Memory Usage (Peak VRAM)

| Model | Peak VRAM (MB) | Notes |
|---|---|---|
| Random | 0.0 MB | No GPU computation |
| Majority | 0.0 MB | No GPU computation |
| MLP | 13.2 MB |  |
| CNN | 14.0 MB |  |
| LSTM | 88.3 MB |  |
| Transformer | 46.9 MB |  |

---

## Training Time

| Model | Epoch Time (s) | Notes |
|---|---|---|
| Random | N/A (non-trainable) | — |
| Majority | N/A (non-trainable) | — |
| MLP | 5.5s | — |
| CNN | 3.9s | — |
| LSTM | 7.9s | — |
| Transformer | 10.1s | — |

---

## Inference Time

| Model | Inference Time (ms/batch) | Throughput (samples/s) |
|---|---|---|
| Random | N/A | N/A |
| Majority | N/A | N/A |
| MLP | 1.46 ms | 21918 samples/s |
| CNN | 3.59 ms | 8914 samples/s |
| LSTM | 7.85 ms | 4076 samples/s |
| Transformer | 5.39 ms | 5937 samples/s |

---

## Memory Budget Analysis (RTX 3060 4 GB)

| Model | VRAM (MB) | Within 4 GB Budget? |
|---|---|---|
| Random | — | — |
| Majority | — | — |
| MLP | 13 MB | ✅ Yes |
| CNN | 14 MB | ✅ Yes |
| LSTM | 88 MB | ✅ Yes |
| Transformer | 47 MB | ✅ Yes |
