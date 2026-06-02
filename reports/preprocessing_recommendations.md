# Preprocessing Recommendations — ARST Phase 1

## 1. Normalization Strategy

**Recommendation: Z-score normalization per modality, computed on training set**

- **IMU:** z-score per channel (acc_x/y/z, rot_w/x/y/z independently)
  - Rationale: channels have different physical units (m/s² vs dimensionless quaternion)
  - Compute μ, σ on training set; apply to val/test (no data leakage)
- **Thermal:** z-score across all 5 channels jointly
  - Rationale: channels measure the same quantity (temperature) — global normalization valid
- **ToF:** Normalize valid readings only (exclude -1.0 sentinel)
  - Per-sensor normalization (tof_1, tof_2, ... independently) since sensors may differ
  - After normalization, set invalid readings to 0.0 (neutral value)

## 2. Missing Value Strategy

**IMU & Thermal:** No missing values expected (NaN rate ≈ 0%)
- If NaN occurs: forward-fill (last valid observation) → backward-fill → 0.0

**ToF:** Use sentinel-aware masking:
1. Create binary validity mask: `tof_mask = (tof != -1.0).float()`
2. Replace -1.0 with 0.0 in the data tensor
3. Pass both `tof` and `tof_mask` to the ToF encoder
4. The Reliability Module can use the mask as a quality signal

## 3. Sequence Padding Strategy

**Recommended window size: T = 51 timesteps**

Rationale: P25=51, median=59. Using T=51 means ~75% of sequences are at least this long.

**Padding:** Pad with zeros at the end (`nn.utils.rnn.pad_sequence`)
- Maintain a sequence length mask for attention-based models
- Short sequences (< T): pad to T with zeros
- Long sequences (> T): extract windows with 50% overlap during training

## 4. Sequence Truncation Strategy

For sequences longer than the window:
1. **Training:** Sliding windows with stride=T//2 (50% overlap)
2. **Inference:** Sliding windows with stride=T (no overlap) → aggregate predictions
3. **Aggregation:** Majority vote or mean probability across windows

## 5. Scaling Strategy

- All sensor values after normalization should be in approximately [-3, 3]
- Clip extreme outliers at ±5σ before normalization (helps with motion artifacts)
- ToF depth values: after masking, values in [0, ~500 mm typically] → normalize to [0, 1]

## 6. Train/Val/Test Split

**Recommended:** Subject-stratified split
- Train: 70% of subjects
- Val: 15% of subjects
- Test: 15% of subjects
- Ensures no subject appears in multiple splits (prevents data leakage)
- Within each split, maintain class balance via stratified sampling

## 7. Data Type

- Use `float32` throughout (halves memory vs float64; sufficient precision for sensor data)
- Labels: `int64` (torch.long)
- Masks: `float32` (0.0 / 1.0)

## 8. Implementation Priority

1. IMU z-score normalization (simple, high priority)
2. ToF mask creation (critical for reliability module)
3. Window extraction with configurable size and stride
4. Thermal normalization (low urgency — data is clean)
