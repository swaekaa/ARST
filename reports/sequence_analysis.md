# Sequence Analysis — ARST Phase 1

## Identifiers

| Identifier | Column | Description |
|---|---|---|
| Sequence ID | `sequence_id` | Unique recording session identifier |
| Subject ID | `subject` | Participant identifier |
| Timestep | `sequence_counter` | Within-sequence timestep index (starts at 1) |

## Sequence Length Statistics

| Statistic | Value |
|---|---|
| Total Sequences | 8,151 |
| Total Subjects | 81 |
| Min Length | 29 |
| Max Length | 700 |
| Mean Length | 70.5 |
| Median Length | 59.0 |
| Std Dev | 35.4 |
| 5th Percentile | 46 |
| 25th Percentile | 51 |
| 75th Percentile | 78 |
| 95th Percentile | 127 |

## Recommendations

- **Recommended window size:** 51 timesteps (≤ P25 to keep most sequences intact)
- **Max safe window:** 46 timesteps (P5 — all sequences have at least this many rows)
- Use sliding windows with 50% overlap for training data augmentation
- Pad shorter sequences to window length with zeros (or last valid value)
- For sequences shorter than the window, use the full sequence with end-padding
