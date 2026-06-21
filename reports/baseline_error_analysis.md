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

**Overall:** Accuracy=0.1807  |  Macro F1=0.1894  |  Weighted F1=0.2203

**Class-wise F1:**

| Class | F1 | Performance |
|---|---|---|
| Hand at target | 0.0456 | ❌ Weak |
| Moves hand | 0.4899 | 🟡 Moderate |
| Performs gesture | 0.1695 | ❌ Weak |
| Relaxes + moves | 0.0525 | ❌ Weak |

**Strongest:** Moves hand, Performs gesture
**Weakest:** Relaxes + moves, Hand at target

**Most common confusions:**

- True **Performs gesture** → Predicted **Hand at target** (474 samples)
- True **Performs gesture** → Predicted **Relaxes + moves** (290 samples)
- True **Performs gesture** → Predicted **Moves hand** (107 samples)

## LSTM

**Overall:** Accuracy=0.0482  |  Macro F1=0.0390  |  Weighted F1=0.0435

**Class-wise F1:**

| Class | F1 | Performance |
|---|---|---|
| Hand at target | 0.0584 | ❌ Weak |
| Moves hand | 0.0000 | ❌ Fails (F1=0) |
| Performs gesture | 0.0526 | ❌ Weak |
| Relaxes + moves | 0.0449 | ❌ Weak |

**Strongest:** Hand at target, Performs gesture
**Weakest:** Relaxes + moves, Moves hand

**Most common confusions:**

- True **Performs gesture** → Predicted **Hand at target** (615 samples)
- True **Performs gesture** → Predicted **Relaxes + moves** (319 samples)
- True **Moves hand** → Predicted **Relaxes + moves** (133 samples)

## Transformer

**Overall:** Accuracy=0.2527  |  Macro F1=0.2007  |  Weighted F1=0.3463

**Class-wise F1:**

| Class | F1 | Performance |
|---|---|---|
| Hand at target | 0.0900 | ❌ Weak |
| Moves hand | 0.2812 | ❌ Weak |
| Performs gesture | 0.3747 | 🟡 Moderate |
| Relaxes + moves | 0.0567 | ❌ Weak |

**Strongest:** Performs gesture, Moves hand
**Weakest:** Hand at target, Relaxes + moves

**Most common confusions:**

- True **Performs gesture** → Predicted **Hand at target** (369 samples)
- True **Performs gesture** → Predicted **Relaxes + moves** (258 samples)
- True **Performs gesture** → Predicted **Moves hand** (111 samples)

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
