# Summary of Phases 2.6 and 2.7

This document outlines the systematic debugging, validation, and repair efforts undertaken during Phase 2.6 (Architecture Validation) and Phase 2.7 (Dataset & Pipeline Investigation) to resolve the catastrophic underperformance of the CNN and Transformer baselines.

---

## Phase 2.6: Architecture Validation and Repair

**Goal:** Verify if the expressive models (CNN, Transformer) were fundamentally broken architecturally or if the issue lay elsewhere.

### 1. Tiny Overfit Sanity Tests
- Created a `tiny_dataset_overfit.py` script to train the models on a heavily restricted 32-sample subset.
- **Result:** Both the CNN and Transformer successfully achieved 100% accuracy and >0.95 Macro F1 on the tiny dataset.
- **Conclusion:** The fundamental architectures were mathematically sound and capable of learning representations; the collapse during full training was an optimization or pipeline issue, not an architectural failure.

### 2. Architectural Refinements
While the architectures were structurally sound, several improvements and bug fixes were applied to increase robustness:
- **CNN Baseline:** Added `BatchNorm1d` and `Dropout` layers to regularize learning and stabilize hidden representations.
- **Transformer Baseline:** 
  - Fixed a critical bug in `_init_weights()` where the standard Xavier/Kaiming initialization of the initial embedding projection layer was being improperly overwritten with scaled-down standard deviations.
  - Adjusted the number of attention heads and dimensions to better suit the modality dimensions.
  - Ensured correct input normalization (`LayerNorm`) application before the multi-head attention blocks.

### 3. Experimental Infrastructure
- Created `lr_study.py` to test various learning rates dynamically.
- Created `ablation_study.py` to verify the impact of specific components (like dropout or positional encodings).
- Documented findings in `cnn_debug_report.md` and `transformer_debug_report.md`.

---

## Phase 2.7: Dataset and Training Pipeline Investigation

**Goal:** Identify why the CNN and Transformer architectures completely collapsed (F1 < 0.10) during full-scale training while the simpler MLP and LSTM baselines succeeded (F1 ~ 0.32-0.39).

### 1. Diagnostic Scripting (Parts 1-7)
Developed a suite of diagnostic scripts to investigate all aspects of the data pipeline:
- **`investigate_class_distribution.py`:** Verified that class distributions were consistent across raw CSVs and dataloader splits to rule out label skew.
- **`investigate_windowing.py`:** Inspected the 64-timestep window extraction logic, majority vote labeling, boundary effects, and generated visualizations of the multimodal sequences.
- **`investigate_normalization.py`:** Checked the standard deviation and mean of batches exiting the `DataLoader`.
- **`investigate_predictions.py`:** Analyzed training dynamics and loss curves to understand the mechanics of the class collapse.

### 2. Root Cause Discovery
The investigation revealed a critical bug in `src/arst/data/dataloader.py`:
- **The Bug:** The `build_csv_loaders` function extracted raw numpy arrays from the CSV and passed them directly to the PyTorch `DataLoader` without applying any form of feature scaling or normalization. 
- **Why MLP & LSTM Survived:** 
  - The MLP applied a global mean-pool across the temporal dimension, drastically smoothing out variance.
  - The LSTM used internal `tanh` and `sigmoid` gating, which implicitly squashes enormous inputs into a safe `[-1, 1]` range.
- **Why CNN & Transformer Collapsed:**
  - The CNN attempted to compute `BatchNorm1d` statistics over massive, highly variable raw sensor values, leading to extreme instability and NaN gradients.
  - The Transformer projected raw, unscaled values and fed them into scaled dot-product attention ($QK^T / \sqrt{d_{k}}$). Because the features were unnormalized, the dot products exploded to infinity. The `softmax` function instantly snapped to a single token, causing gradients to explode, trapping the model in permanent class collapse on Step 1.

### 3. Implementation of the Fix (Phase 2.8)
- Modified `dataloader.py` to include proper Z-score normalization.
- Computed the `mean` and `std` strictly on the **Training Split** arrays to prevent data leakage.
- Applied `(x - mean) / std` standard scaling sequentially to the Train, Validation, and Test arrays before wrapping them in the PyTorch Dataset.

### 4. UTF-8 Encoding Fixes
- Addressed cross-platform read/write errors that caused `UnicodeDecodeError` or garbled text by hardcoding `encoding="utf-8"` in:
  - `src/arst/evaluation/evaluate.py`
  - `scripts/generate_benchmark_report.py`

---

## Scientific Lessons Learned

The failure of the CNN and Transformer baselines initially suggested architectural flaws. However, tiny-overfit experiments demonstrated that both models possessed sufficient expressive capacity. A deeper investigation revealed that the real culprit was the absence of feature normalization in the dataloader pipeline.

This highlighted several important principles:

1. Successful tiny-overfit tests are strong evidence that architectures are fundamentally sound.

2. Complex models such as CNNs and Transformers are much more sensitive to feature scaling than MLPs and gated recurrent networks.

3. Debugging should proceed from:
   data → optimization → architecture

rather than assuming architectural failure.

4. Silent preprocessing bugs can dominate model performance and lead to misleading conclusions.

The normalization fix restored the intended behavior of expressive sequence models and enabled the project to proceed to the next stage.

## Conclusion
The completion of Phases 2.6 and 2.7 correctly identified and solved a silent data pipeline flaw. With the data normalization in place, the expressive capacities of the CNN and Transformer models are unlocked, clearing the path to execute the final Baseline Benchmarks and advance to Phase 3.
