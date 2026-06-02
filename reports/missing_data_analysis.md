# Missing Data Analysis — ARST Phase 1

## Modality-Level Summary

| Modality | Mean Missing % | Max Missing % | Min Missing % |
|---|---|---|---|
| imu | 0.37% | 0.64% | 0.0% |
| thermal | 2.11% | 5.79% | 1.08% |
| tof | 59.4% | 74.39% | 43.86% |

## Key Findings

1. **IMU:** Minimal NaN values. IMU data is nearly complete.
2. **Thermopile:** Minimal NaN values. Thermopile data is nearly complete.
3. **ToF:** Significant invalid readings encoded as `-1.0` (not NaN). These represent failed distance measurements (non-reflective surfaces, out-of-range, etc.).

## ToF Invalid Reading Strategy

- `-1.0` values are **domain-specific invalids** (sensor failure), not missing-at-random
- Replace with `0.0` or maximum valid range (creates boundary condition) — chosen strategy: `0.0`
- Alternative: use a ToF mask channel alongside the raw values (recommended for ARST)
- The mask provides explicit signal to the reliability module about ToF data quality

## Recommendations

- IMU: z-score normalization (data is complete)
- Thermal: z-score normalization (data is complete)
- ToF: replace `-1.0` with `0.0`, create binary validity mask channel
- Store ToF mask as separate `tof_mask` channels for the Reliability Module
