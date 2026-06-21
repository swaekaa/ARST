# ARST Phase 2.5 — Baseline Benchmark Results

> **Status:** ✅ Complete  
> **Generated:** Phase 2.5 benchmark validation  
> **Primary metric:** Macro F1-Score (higher is better)  
> **Experimental seed:** 42  

---

## Dataset (Phase 1 Verified)

| Property | Value |
|---|---|
| Total sequences | 8,151 |
| Subjects | 81 |
| Window size (T) | 64 |
| IMU channels | 7 (acc_xyz + quaternion) |
| Thermal channels | 5 (linear array) |
| ToF channels | 320 (5 sensors × 64 pixels) |
| ToF invalidity | ~59.4% |
| Classes | 4 (3.79× imbalance) |
| Train / Val / Test | Subject-stratified 70/15/15 |

---

## Benchmark Leaderboard

| Model | Accuracy | Macro F1 | Weighted F1 | Params | Peak VRAM | Epoch Time |
|---|---|---|---|---|---|---|
| Random | 0.7850 | **0.2199** | 0.6904 | 0 | 0 MB | N/A |
| Majority | 0.7850 | **0.2199** | 0.6904 | 0 | 0 MB | N/A |
| MLP | 0.6410 | **0.3179** | 0.6933 | 337,028 | 13 MB | 5.5s |
| CNN | 0.1807 | **0.1894** | 0.2203 | 172,740 | 14 MB | 3.9s |
| LSTM | 0.0482 | **0.0390** | 0.0435 | 2,003,396 | 88 MB | 7.9s |
| Transformer | 0.2527 | **0.2007** | 0.3463 | 457,092 | 47 MB | 10.1s |

---

## Rankings (by Macro F1)

| Rank | Model | Macro F1 | Δ vs Random |
|---|---|---|---|
| 1 | **MLP** | 0.3179 | +0.0981 |
| 2 | **Random** | 0.2199 | +0.0000 |
| 3 | **Majority** | 0.2199 | +0.0000 |
| 4 | **Transformer** | 0.2007 | +-0.0192 |
| 5 | **CNN** | 0.1894 | +-0.0305 |
| 6 | **LSTM** | 0.0390 | +-0.1809 |

---

## Per-Class F1 Breakdown

| Model | Hand at target | Moves hand | Performs gesture | Relaxes + moves |
|---|---|---|---|---|
| Random | 0.0000 | 0.0000 | 0.8795 | 0.0000 |
| Majority | 0.0000 | 0.0000 | 0.8795 | 0.0000 |
| MLP | 0.0357 | 0.3882 | 0.7950 | 0.0529 |
| CNN | 0.0456 | 0.4899 | 0.1695 | 0.0525 |
| LSTM | 0.0584 | 0.0000 | 0.0526 | 0.0449 |
| Transformer | 0.0900 | 0.2812 | 0.3747 | 0.0567 |

---

## Training Protocol

All models trained with identical conditions:

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-2 |
| Scheduler | Cosine with warmup (5 epochs) |
| Loss | Focal Loss (γ=2.0, class-weighted) |
| Max epochs | 100 |
| Early stopping patience | 15 (val/f1_macro) |
| Batch size | 32 (effective 128 with accumulation) |
| Mixed precision | AMP enabled |
| Gradient clip | max_norm=1.0 |
| Seed | 42 |
