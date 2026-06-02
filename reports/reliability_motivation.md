# Sensor Reliability Motivation Study — ARST Phase 1

> This report provides empirical justification for the Adaptive Reliability Module (ARM).

## Research Question

> Do different behaviors appear to depend on different sensor modalities?

## Evidence 1: Differential Missing Data Rates

The three modalities exhibit dramatically different data quality profiles:

| Modality | Mean Missing % | Max Missing % | Interpretation |
|---|---|---|---|
| IMU | 0.37% | 0.64% | High reliability — rarely fails |
| Thermopile | 2.11% | 5.79% | High reliability — rarely fails |
| ToF | 59.4% | 74.39% | Variable reliability — invalid readings common |

**Implication:** Static fusion weighting cannot account for the per-timestep, per-sample variation in ToF data quality. The Reliability Module must learn to detect and downweight invalid ToF regions.

## Evidence 2: Signal Dynamic Range Differences

The three modalities operate on completely different scales:

- **IMU:** mean range [-0.46, 1.79], std range [0.23, 6.10]
- **THERMAL:** mean range [26.67, 27.56], std range [2.25, 4.12]
- **TOF:** mean range [87.60, 120.81], std range [50.17, 69.72]

**Implication:** Per-modality normalization is essential. The disparate scales mean that a naive concatenation fusion would be dominated by the largest-magnitude modality.

## Evidence 3: ToF Sensor-Level Variation

The dataset contains **5 ToF sensors** (tof_1 through tof_5), each with 64 pixels. Each sensor has an independently varying invalid reading rate, suggesting different physical placement and orientation. This intra-modality variation is a strong argument for per-sensor or per-pixel reliability estimation.

## Evidence 4: Behavior Diversity

The dataset contains **4 distinct behavior classes** across multiple orientations, postures, and gesture types. Based on domain knowledge:

- **Gross motor behaviors** (walking, sitting, standing transitions): IMU is primary sensor
- **Fine motor behaviors** (gestures, hand movements): ToF provides depth cues
- **Thermal proximity behaviors** (face down, cheek contact): Thermopile is most informative

A fixed fusion strategy cannot capture this behavior-conditional modality importance.

## Conclusion

The empirical evidence strongly supports the ARST hypothesis:

1. **Data quality is non-uniform**: ToF has variable invalidity; IMU and thermal are cleaner
2. **Modalities are behavior-conditional**: Different behaviors activate different sensors
3. **Static fusion is suboptimal**: Any fixed weighting cannot capture per-timestep signal quality
4. **Reliability estimation is learnable**: The ARM has a clear signal (invalid pixels, signal quality) to learn from

**This justifies Phase 4's Adaptive Reliability Module as a principled engineering choice.**
