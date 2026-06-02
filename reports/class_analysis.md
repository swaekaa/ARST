# Class Analysis — ARST Phase 1

**Primary target column:** `behavior`

**Secondary label columns:** `phase`, `gesture`

**Number of behavior classes:** 4

**Total labeled rows:** 574,945

**Class imbalance ratio (max/min):** 3.79×

## Behavior Class Frequencies

| Rank | Behavior | Count | % |
|---|---|---|---|
| 1 | Performs gesture | 255,817 | 44.49% |
| 2 | Moves hand to target location | 156,474 | 27.22% |
| 3 | Hand at target location | 95,173 | 16.55% |
| 4 | Relaxes and moves hand to target location | 67,481 | 11.74% |

## Gesture Classes

Unique gestures: 18

| Gesture | Count |
|---|---|
| Text on phone | 58,462 |
| Neck - scratch | 56,619 |
| Eyebrow - pull hair | 44,305 |
| Forehead - scratch | 40,923 |
| Forehead - pull hairline | 40,802 |
| Above ear - pull hair | 40,560 |
| Neck - pinch skin | 40,507 |
| Eyelash - pull hair | 40,218 |
| Cheek - pinch skin | 40,124 |
| Wave hello | 34,356 |
| Write name in air | 31,267 |
| Pull air toward your face | 30,743 |
| Feel around in tray and pull out an object | 17,114 |
| Glasses on/off | 13,542 |
| Drink from bottle/cup | 13,093 |
| Scratch knee/leg skin | 12,328 |
| Write name on leg | 10,138 |
| Pinch knee/leg skin | 9,844 |

## Phase Distribution

| Phase | Count |
|---|---|
| Transition | 319,128 |
| Gesture | 255,817 |

## Imbalance Risk Assessment

With a 3.79× imbalance ratio, standard cross-entropy loss will be biased.
**Recommended mitigations:**
- Class-weighted cross-entropy (`torch.nn.CrossEntropyLoss(weight=class_weights)`)
- Focal Loss (γ=2.0) to down-weight easy majority-class samples
- Stratified train/val/test split (by behavior + subject)
- Macro F1 as primary evaluation metric (not accuracy)
