"""
Phase 2.7 - Part 4: Investigate Normalization
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from arst.data.dataloader import build_csv_loaders

def investigate_normalization():
    csv_path = Path("data/raw/train.csv")
    report_path = Path("reports/normalization_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    print("Building DataLoaders to inspect batch stats...")
    train_loader, _, _, _ = build_csv_loaders(
        csv_path=csv_path,
        window_size=64,
        batch_size=128,
        val_fraction=0.15,
        test_fraction=0.15,
        num_workers=0,
        seed=42
    )

    batch = next(iter(train_loader))
    
    imu = batch["imu"].numpy()        # [B, T, 7]
    thermo = batch["thermo"].numpy()  # [B, T, 5]
    tof = batch["tof"].numpy()        # [B, T, 320]

    def get_stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "has_nan": bool(np.isnan(arr).any()),
            "has_inf": bool(np.isinf(arr).any()),
            "zero_variance": bool(np.std(arr) == 0.0)
        }

    imu_stats = get_stats(imu)
    thermo_stats = get_stats(thermo)
    tof_stats = get_stats(tof)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2.7 - Normalization Report\n\n")
        f.write("## 1. Batch Statistics (Unnormalized?)\n")
        
        f.write("### IMU\n")
        for k, v in imu_stats.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n### Thermal\n")
        for k, v in thermo_stats.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n### ToF\n")
        for k, v in tof_stats.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## 2. Findings\n")
        if any(abs(s["mean"]) > 5.0 or s["std"] > 10.0 for s in [imu_stats, thermo_stats, tof_stats]):
            f.write("- **CRITICAL**: The data appears to be completely unnormalized. ")
            f.write("Large means and standard deviations indicate raw sensor values are being fed directly to the models.\n")
            f.write("- While MLP and LSTM can often tolerate unnormalized data (due to their layer structure and gating mechanisms), CNNs (with BatchNorm) and Transformers are highly sensitive to input scale.\n")
            f.write("- This is likely the root cause of the architecture collapse during full training.\n")
        else:
            f.write("- The data appears to be normalized correctly.\n")

    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    investigate_normalization()
