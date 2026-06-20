# Phase 2.6 — Baseline Architecture Ablations

> **Generated:** 2026-06-20 11:48
> **Epochs per run:** 30
> **Learning rate:** 0.0001

## Transformer Pooling Ablation

| Variant | Pool Type | Best Val F1 |
|---|---|---|
| transformer_cls | cls | **0.1168** |
| transformer_mean | mean | **0.0245** |

## CNN Kernel Ablation

| Variant | Kernels | Best Val F1 |
|---|---|---|
| cnn_k3_5_7 | k3_5_7 | **0.2562** |
| cnn_k3_7 | k3_7 | **0.1746** |
| cnn_k5 | k5 | **0.1262** |
| cnn_k3 | k3 | **0.1073** |
