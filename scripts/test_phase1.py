"""Phase 1 smoke test — validates dataset and dataloader."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arst.data.dataloader import build_csv_loaders  # noqa: E402

print("Testing CSV-backed DataLoader (Phase 1)...")
train_loader, val_loader, test_loader, info = build_csv_loaders(
    csv_path=ROOT / "data" / "raw" / "train.csv",
    window_size=128,
    batch_size=8,
    max_rows=10_000,
    num_workers=0,
)

print("n_classes:", info["n_classes"])
print("n_train:", info["n_train"])
print("n_val:", info["n_val"])
print("modality_dims:", info["modality_dims"])

batch = next(iter(train_loader))

print("\n=== Batch Shapes ===")
for k, v in batch.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {v.shape} {v.dtype}")

print("\n=== Shape Assertions ===")
imu_ch = batch["imu"].shape[2]
thm_ch = batch["thermo"].shape[2]
tof_ch = batch["tof"].shape[2]
assert imu_ch == 7, f"IMU should be 7 channels, got {imu_ch}"
assert thm_ch == 5, f"Thermal should be 5 channels, got {thm_ch}"
assert tof_ch == 320, f"ToF should be 320 channels, got {tof_ch}"
assert batch["tof_mask"].shape == batch["tof"].shape, "ToF mask shape mismatch"
print("All shape assertions PASSED")

print("\nBehavior classes:", info["behavior_encoder"])
print("\nPhase 1 dataloader smoke test: PASSED")
