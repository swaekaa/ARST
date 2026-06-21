# Single Modality Baselines

Prior to training the Adaptive Reliability Module (ARM) in Phase 4, we establish the individual predictive power of each sensor modality.

These baselines are trained using the corrected pipeline (Phase 2.8) with the Transformer model, restricting the input to a single active modality.

| Modality | Accuracy | Macro F1 |
| -------- | -------- | -------- |
| IMU      | ?        | ?        |
| Thermal  | ?        | ?        |
| ToF      | ?        | ?        |

*(Note: Run `python train.py model=transformer model.active_modalities=[imu]` to generate these results.)*
