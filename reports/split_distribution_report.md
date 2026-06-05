# ARST Split Distribution Report

This report summarizes the behavior class distribution across the Train, Validation, and Test splits. The split logic partitions at the subject level, avoiding data leakage.

## Dataset Audit Results

Encoder Mapping:
* `0`: Hand at target location
* `1`: Moves hand to target location
* `2`: Performs gesture
* `3`: Relaxes and moves hand to target location

### Train Split
Total Sequences: `5632`
* Class 0: `88` (1.6%)
* Class 1: `1263` (22.4%)
* Class 2: `4200` (74.6%)
* Class 3: `81` (1.4%)

### Validation Split
Total Sequences: `1167`
* Class 0: `16` (1.4%)
* Class 1: `272` (23.3%)
* Class 2: `862` (73.9%)
* Class 3: `17` (1.5%)

### Test Split
Total Sequences: `1223`
* Class 0: `34` (2.8%)
* Class 1: `213` (17.4%)
* Class 2: `960` (78.5%)
* Class 3: `16` (1.3%)

## Observations
* **Consistency:** The class distribution is extremely consistent across all three splits, meaning the subject-based split is robust.
* **Imbalance:** Class 2 ("Performs gesture") dominates the dataset, making up around ~75% of the samples across all splits. Classes 0 and 3 are minority classes (each < 3%).
* **Contiguity:** All 4 classes `[0, 1, 2, 3]` are now correctly present in every split.
