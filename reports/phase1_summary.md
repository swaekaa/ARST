# Phase 1 Summary — Dataset Exploration

**Status:** ✅ Complete

## Key Findings

1. **Dataset format:** Flat CSV (not per-sequence parquet). 574,945 rows × 341 columns.
2. **Train size:** ~1069.59 MB on disk.
3. **Behavior classes:** 4 unique behaviors.
4. **Class imbalance:** 3.79× ratio (max/min class size).
5. **Sequences:** 8,151 total, 81 subjects.
6. **Sequence lengths:** min=29, max=700, mean=71.
7. **IMU features:** 7 channels (acc + quaternion, NOT acc+gyro as docs assumed).
8. **Thermal features:** 5 channels (5, NOT 64).
9. **ToF features:** 320 channels (5 sensors × 64 pixels each).
10. **ToF invalidity:** High rate of -1.0 sentinel values encoding failed distance measurements.

## Dataset Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Class imbalance (3.79×) | High | Weighted loss, stratified splits, Focal Loss |
| ToF invalid readings (high %) | High | Sentinel masking, validity mask channel, reliability module |
| Variable sequence lengths | Medium | Fixed-window extraction with padding |
| IMU: quaternion not gyroscope | Medium | Update architecture — quaternion has different properties than gyro |
| Flat CSV (not parquet) | Low | Chunked reading implemented; HDF5 conversion recommended |
| 1.1 GB training file | Medium | Never load fully; use chunked pipeline or convert to HDF5 |

## Memory Considerations (RTX 3060 4GB)

- Full CSV in RAM: ~1.64 GB — **DO NOT load all at once**
- Recommended: Convert to HDF5 (`preprocessor.py`) → windowed, compressed storage
- Window size T=51: each sample = (T×7 + T×5 + T×320) float32 ≈ 66.1 KB
- At batch_size=32: ~1 MB per batch (fits easily in VRAM)
- Full ARST model: ~2.5 GB VRAM (needs gradient checkpointing for T=256)

## Recommended Max Sequence Length

**T = 51 timesteps**

- P5 = 46 (all sequences have at least this length)
- P25 = 51 (75% of sequences are at least this long)
- With T=51, most sequences provide 1-2 windows; short ones are padded

## Recommended Preprocessing Pipeline

```
Raw train.csv
    │
    ▼ (chunked reading)
Modality separation: [IMU 7ch] [Thermal 5ch] [ToF 320ch]
    │
    ▼
ToF sentinel masking: -1.0 → 0.0, mask = (tof != -1.0)
    │
    ▼
Z-score normalization (per-channel, fit on train)
    │
    ▼
Window extraction: T=51, stride=T//2
    │
    ▼
Save to HDF5: /windows/imu, /windows/thermo, /windows/tof, /windows/tof_mask, /windows/labels
    │
    ▼
ARSTDataset → DataLoader → Training
```

## Recommendations for Phase 2 Baseline Models

1. **Architecture note:** IMU has quaternion (rot_w/x/y/z), not raw gyroscope. Treat as 7-channel signal.
2. **Thermal note:** Only 5 thermopile channels (not 64 as docs assumed). Linear projection head is appropriate.
3. **ToF note:** 320 total features (5 sensors × 64 pixels). Consider per-sensor processing.
4. **Baseline input:** After windowing, shape is `[B, T, 7]` (IMU), `[B, T, 5]` (thermal), `[B, T, 320]` (ToF).
5. **Loss:** Use Focal Loss or weighted CE given the class imbalance.
6. **Evaluation:** Macro F1 (not accuracy) is the primary metric.
7. **Start simple:** MLP on statistical features (mean, std, min, max per channel per window).
8. **Memory:** With T=256 and batch=32, batches are tiny — you can use larger batches than expected.

## Phase 1 Deliverables Checklist

- [x] `reports/dataset_inventory.md` — File inventory with sizes, row counts, schema
- [x] `reports/dataset_profile.md` — Chunked profiling of train.csv
- [x] `configs/sensor_groups.yaml` — Automatic sensor group identification
- [x] `reports/class_analysis.md` — Target analysis with class frequencies
- [x] `outputs/eda/class_distribution.png` — Class distribution chart
- [x] `reports/sequence_analysis.md` — Sequence length statistics
- [x] `outputs/eda/sequence_length_distribution.png` — Sequence length histogram
- [x] `reports/missing_data_analysis.md` — Missing data analysis
- [x] `outputs/eda/missing_values_heatmap.png` — Missing data heatmap
- [x] `outputs/eda/missing_values_report.csv` — Per-feature missing report
- [x] `outputs/eda/sensor_statistics.csv` — Per-feature statistics
- [x] `outputs/eda/examples/` — IMU, thermal, ToF visualizations per behavior
- [x] `reports/reliability_motivation.md` — Reliability module justification
- [x] `reports/preprocessing_recommendations.md` — Preprocessing strategy
- [x] `src/arst/data/dataset.py` — Updated PyTorch dataset (flat CSV aware)
- [x] `src/arst/data/dataloader.py` — Train/val/test loaders
- [x] `notebooks/phase1_dataset_exploration.ipynb` — Reproducible EDA notebook
- [x] `reports/phase1_summary.md` — This document
