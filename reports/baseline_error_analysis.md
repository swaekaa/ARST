# ARST Phase 2.5 — Baseline Error Analysis

> Per-class performance breakdown and common failure patterns.
> Primary metric: per-class F1-Score.

---

## Random

**Overall:** Accuracy=0.7850  |  Macro F1=0.2199  |  Weighted F1=0.6904

**Class-wise F1:**

| Class | F1 | Performance |
|---|---|---|
| Hand at target | 0.0000 | ❌ Fails (F1=0) |
| Moves hand | 0.0000 | ❌ Fails (F1=0) |
| Performs gesture | 0.8795 | ✅ Strong |
| Relaxes + moves | 0.0000 | ❌ Fails (F1=0) |

**Strongest:** Performs gesture, Hand at target
**Weakest:** Moves hand, Relaxes + moves

**Most common confusions:**

- True **Moves hand** → Predicted **Performs gesture** (213 samples)
- True **Hand at target** → Predicted **Performs gesture** (34 samples)
- True **Relaxes + moves** → Predicted **Performs gesture** (16 samples)

## Majority

**Overall:** Accuracy=0.7850  |  Macro F1=0.2199  |  Weighted F1=0.6904

**Class-wise F1:**

| Class | F1 | Performance |
|---|---|---|
| Hand at target | 0.0000 | ❌ Fails (F1=0) |
| Moves hand | 0.0000 | ❌ Fails (F1=0) |
| Performs gesture | 0.8795 | ✅ Strong |
| Relaxes + moves | 0.0000 | ❌ Fails (F1=0) |

**Strongest:** Performs gesture, Hand at target
**Weakest:** Moves hand, Relaxes + moves

**Most common confusions:**

- True **Moves hand** → Predicted **Performs gesture** (213 samples)
- True **Hand at target** → Predicted **Performs gesture** (34 samples)
- True **Relaxes + moves** → Predicted **Performs gesture** (16 samples)

## MLP

**Overall:** Accuracy=0.6410  |  Macro F1=0.3179  |  Weighted F1=0.6933

**Class-wise F1:**

| Class | F1 | Performance |
|---|---|---|
| Hand at target | 0.0357 | ❌ Weak |
| Moves hand | 0.3882 | 🟡 Moderate |
| Performs gesture | 0.7950 | ✅ Strong |
| Relaxes + moves | 0.0529 | ❌ Weak |

**Strongest:** Performs gesture, Moves hand
**Weakest:** Relaxes + moves, Hand at target

**Most common confusions:**

- True **Performs gesture** → Predicted **Relaxes + moves** (154 samples)
- True **Performs gesture** → Predicted **Moves hand** (96 samples)
- True **Moves hand** → Predicted **Performs gesture** (79 samples)

## CNN

**Overall:** Accuracy=0.0286  |  Macro F1=0.0278  |  Weighted F1=0.0022

**Class-wise F1:**

| Class | F1 | Performance |
|---|---|---|
| Hand at target | 0.0527 | ❌ Weak |
| Moves hand | 0.0000 | ❌ Fails (F1=0) |
| Performs gesture | 0.0000 | ❌ Fails (F1=0) |
| Relaxes + moves | 0.0584 | ❌ Weak |

**Strongest:** Relaxes + moves, Hand at target
**Weakest:** Moves hand, Performs gesture

**Most common confusions:**

- True **Performs gesture** → Predicted **Hand at target** (625 samples)
- True **Performs gesture** → Predicted **Relaxes + moves** (335 samples)
- True **Moves hand** → Predicted **Relaxes + moves** (134 samples)

## LSTM

**Overall:** Accuracy=0.6484  |  Macro F1=0.3896  |  Weighted F1=0.7143

**Class-wise F1:**

| Class | F1 | Performance |
|---|---|---|
| Hand at target | 0.0800 | ❌ Weak |
| Moves hand | 0.6105 | ✅ Strong |
| Performs gesture | 0.7700 | ✅ Strong |
| Relaxes + moves | 0.0980 | ❌ Weak |

**Strongest:** Performs gesture, Moves hand
**Weakest:** Relaxes + moves, Hand at target

**Most common confusions:**

- True **Performs gesture** → Predicted **Moves hand** (155 samples)
- True **Performs gesture** → Predicted **Hand at target** (136 samples)
- True **Performs gesture** → Predicted **Relaxes + moves** (63 samples)

## Transformer

**Overall:** Accuracy=0.0294  |  Macro F1=0.0352  |  Weighted F1=0.0128

**Class-wise F1:**

| Class | F1 | Performance |
|---|---|---|
| Hand at target | 0.0460 | ❌ Weak |
| Moves hand | 0.0441 | ❌ Weak |
| Performs gesture | 0.0042 | ❌ Weak |
| Relaxes + moves | 0.0466 | ❌ Weak |

**Strongest:** Relaxes + moves, Hand at target
**Weakest:** Moves hand, Performs gesture

**Most common confusions:**

- True **Performs gesture** → Predicted **Hand at target** (734 samples)
- True **Performs gesture** → Predicted **Relaxes + moves** (216 samples)
- True **Moves hand** → Predicted **Hand at target** (118 samples)

---

## Cross-Model Patterns

### Class Difficulty

| Class | Description | Pattern |
|---|---|---|
| Performs gesture | Dominant class (44.5% of data) | All models learn this well |
| Moves hand | Second most common | Moderate performance across models |
| Hand at target | Minority class | Most models struggle severely |
| Relaxes + moves | Minority class | Most models struggle severely |

### Root Causes

1. **Class imbalance**: "Performs gesture" dominates training signal despite Focal Loss
2. **ToF invalidity**: ~59.4% invalid ToF readings reduce discriminative signal
3. **No temporal reliability**: Models cannot discount unreliable sensor windows
4. **Boundary confusion**: Transition classes (Moves hand, Relaxes) overlap temporally
