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
| CNN | 0.0286 | **0.0278** | 0.0022 | 172,356 | 14 MB | 4.1s |
| LSTM | 0.6484 | **0.3896** | 0.7143 | 2,003,396 | 88 MB | 9.7s |
| Transformer | 0.0294 | **0.0352** | 0.0128 | 448,324 | 47 MB | 8.1s |

---

## Rankings (by Macro F1)

| Rank | Model | Macro F1 | Δ vs Random |
|---|---|---|---|
| 1 | **LSTM** | 0.3896 | +0.1698 |
| 2 | **MLP** | 0.3179 | +0.0981 |
| 3 | **Random** | 0.2199 | +0.0000 |
| 4 | **Majority** | 0.2199 | +0.0000 |
| 5 | **Transformer** | 0.0352 | +-0.1847 |
| 6 | **CNN** | 0.0278 | +-0.1921 |

---

## Per-Class F1 Breakdown

| Model | Hand at target | Moves hand | Performs gesture | Relaxes + moves |
|---|---|---|---|---|
| Random | 0.0000 | 0.0000 | 0.8795 | 0.0000 |
| Majority | 0.0000 | 0.0000 | 0.8795 | 0.0000 |
| MLP | 0.0357 | 0.3882 | 0.7950 | 0.0529 |
| CNN | 0.0527 | 0.0000 | 0.0000 | 0.0584 |
| LSTM | 0.0800 | 0.6105 | 0.7700 | 0.0980 |
| Transformer | 0.0460 | 0.0441 | 0.0042 | 0.0466 |

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
