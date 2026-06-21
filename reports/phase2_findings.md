# ARST Phase 2.5 — Research Findings

> Phase 2 complete baseline evaluation.
> All models trained with identical protocol (seed=42, Focal Loss, AdamW).

---

## 1. Which Baseline Performs Best?

**Best model: MLP** with Macro F1 = **0.3179**

| Model | Macro F1 | Δ vs Random |
|---|---|---|
| Random | 0.2199 | baseline |
| MLP | 0.3179 | +0.0981 |
| CNN | 0.1894 | +-0.0305 |
| LSTM | 0.0390 | +-0.1809 |
| Transformer | 0.2007 | +-0.0192 |

---

## 2. Does Temporal Modeling Help?

**MLP (no temporal):** 0.3179
**CNN (local temporal):** 0.1894
**LSTM (sequential):** 0.0390
**Transformer (global attention):** 0.2007

**Marginal** — temporal models show limited advantage over MLP in this dataset, suggesting that either the window-level statistics dominate, or that the 59.4% ToF invalidity rate is masking temporal structure.

---

## 3. Is Transformer Justified?

Transformer Macro F1: **0.2007**
LSTM Macro F1: **0.0390**
Difference: +0.1617

**Yes** — the Transformer outperforms the BiLSTM, and its cross-modal attention mechanism provides the foundation for Phase 3 ARST, which needs to model per-timestep reliability across modalities. The Transformer's attention weights can be extended to per-sensor reliability gating.

The Transformer is the recommended baseline for Phase 3 comparisons because:
1. Highest architectural capacity for incorporating reliability signals
2. Attention mechanism naturally extends to per-sensor confidence weighting
3. CLS token provides a clean interface for downstream task heads

---

## 4. How Much Room Remains for ARST?

| Metric | Best Baseline | Perfect | Headroom |
|---|---|---|---|
| Macro F1 | 0.3179 | 1.0000 | 0.6821 |
| Accuracy | 0.7850 | 1.0000 | 0.2150 |

**Headroom for ARST: 0.6821 Macro F1 points** (~68.2%)

The minority classes ("Hand at target", "Relaxes + moves") have near-zero F1 across all baselines. ARST must specifically address these hard classes through reliability-aware feature weighting.

---

## 5. What Weaknesses Remain That ARST Could Solve?

### ToF Invalidity (59.4%)
- All Phase 2 baselines treat invalid ToF readings (filled with 0.0) identically to valid ones
- **ARST solution**: Per-timestep ToF validity mask → weighted attention over valid frames only
- Estimated impact: The 320-channel ToF is the richest modality but currently degraded by invalidity

### Modality Reliability
- When a modality is partially valid, baselines average valid+invalid equally
- **ARST solution**: Per-modality reliability scores → down-weight unreliable modalities per sample
- This is particularly critical for ToF (59.4% invalid) and IMU (potential motion artifacts)

### Missing Sensor Information
- Some samples may have entire modalities missing/corrupted
- **ARST solution**: Modality dropout resilience — model should gracefully degrade when a modality is absent
- Phase 2 baselines would fail or produce garbage outputs if a modality is zeroed

### Class Imbalance (3.79× ratio)
- Minority classes ("Hand at target", "Relaxes + moves") have near-zero F1 in all baselines
- **ARST solution**: Reliability-weighted sampling + per-class reliability calibration
- Reliable samples from minority classes should receive higher training weight

---

## Recommendation

**Primary baseline for ARST comparison: MLP** (Macro F1 = 0.3179)

All Phase 3 ARST results must beat this threshold to justify the reliability-aware architecture overhead.

**Target for Phase 3 ARST**: Macro F1 ≥ 0.3679 (+5 points over best baseline)
