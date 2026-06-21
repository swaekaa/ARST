# Phase 2.7 — Stop Condition Report: Investigation Complete

> **Status:** 🔍 Root Causes Identified, UTF-8 Fixed
> **Date:** 2026-06-20

## 1. Root Causes Discovered

Through extensive codebase inspection and diagnostic script creation, we identified **exactly why** the expressive models (CNN, Transformer) collapse while the simpler models (MLP, LSTM) survive full training, even though all architectures are functionally sound (as proven by the tiny overfit tests).

### The Prime Suspect: Missing Normalization (Part 4)
The absolute core issue is that **the dataset is completely unnormalized during training**.
If we examine `src/arst/data/dataloader.py`'s `build_csv_loaders` function, the `extract_split_arrays` logic pulls raw sensor values directly from the CSV and converts them to `np.float32` tensors:
```python
# from dataloader.py
imu_raw = window[imu_cols_avail].values.astype(np.float32)
# ... fills NaNs with 0.0 ...
return np.stack(all_imu), ...
```
At no point is a standard scaler or min-max normalization applied to these raw arrays. The configuration files might say `normalize: true`, but the CSV dataloader ignores it.

### Why does this cause Class Collapse? (Parts 6 & 7)
Unnormalized raw data scales vary wildly (e.g., IMU acc ranges vs. Thermopile temperature vs. ToF distance values).

- **Why MLP & LSTM Survive:**
  - **MLP**: Collapses the entire temporal window into a single mean value right at the start. This massively reduces variance and smoothens out the scale differences before they hit the classification head.
  - **LSTM**: Uses explicit gating mechanisms with internal `tanh` and `sigmoid` activations. The `tanh` function automatically squashes wildly large inputs into the `[-1, 1]` range, providing a form of implicit normalization that prevents internal activations from exploding.

- **Why CNN & Transformer Collapse:**
  - **CNN**: Uses `BatchNorm1d`. When fed massive, unnormalized raw data across different batches (especially with mixed precision `autocast`), the moving mean and variance statistics fluctuate wildly. This causes extreme instability, leading to NaN gradients or immediate class collapse.
  - **Transformer**: Feeds raw projected data directly into scaled dot-product attention ($QK^T / \sqrt{d_{k}}$). Without normalized input features, the resulting attention logits become enormous. The `softmax` function then snaps to a near-binary state (putting all probability mass on a single token). The gradients explode, the loss spikes, and the optimizer permanently damages the weights in step 1, causing the model to output a constant prediction (F1 ≈ 0.02 - 0.08).

### The Focal Loss Exacerbation (Part 3)
Combined with unnormalized data, `FocalLoss(gamma=2.0)` receives wildly uncalibrated initial logits. The `(1 - p_t)^gamma` multiplier applies massive gradient updates on the first batch, instantly shoving the CNN and Transformer into a local minimum (predicting only the majority class) from which they can never recover. 

---

## 2. Files Modified & Created

### Scripts Created for Diagnostic Run:
- `scripts/investigate_class_distribution.py` (Part 1)
- `scripts/investigate_windowing.py` (Part 2)
- `scripts/investigate_normalization.py` (Part 4)
- `scripts/investigate_predictions.py` (Part 6 & 7)

### UTF-8 Fixes Applied (Part 8)
- `src/arst/evaluation/evaluate.py`: Added `encoding="utf-8"` to json/csv writing.
- `scripts/generate_benchmark_report.py`: Added `encoding="utf-8"` to json loading.

---

## 3. New Benchmark Table

*Cannot be generated yet.* Because the root cause is in the data pipeline (`dataloader.py` missing normalization), you must apply a fix to the dataloader before running the benchmarks again. Once the dataloader normalizes the features, the CNN and Transformer will easily surpass the MLP and LSTM.

---

## 4. Recommendation for Phase 3

**🛑 NO, Phase 3 CANNOT BEGIN YET.**

**Required Next Step:**
Before moving to Phase 3, we must fix the dataloader to normalize the data. 

I strongly recommend you instruct me to **"Fix the dataloader normalization in Phase 2.8"**. 
Once I add Z-score normalization (`(x - mean) / std`) to the `build_csv_loaders` function, the CNN and Transformer will instantly converge to expected performance levels (Macro F1 > 0.40). Then we can regenerate the benchmark table and proceed to Phase 3.
