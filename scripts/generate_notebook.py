"""Script to generate phase1_dataset_exploration.ipynb"""

import json
from pathlib import Path


def code_cell(src):
    if isinstance(src, str):
        src = [s + "\n" for s in src.split("\n")]
        # Remove trailing newline from last line
        if src and src[-1] == "\n":
            src.pop()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def md_cell(src):
    if isinstance(src, str):
        src = [s + "\n" for s in src.split("\n")]
        if src and src[-1] == "\n":
            src.pop()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src,
    }


cells = []

# Title
cells.append(
    md_cell(
        """# ARST Phase 1 — Dataset Exploration
**Adaptive Reliability Sensor Transformer**
CMI — Detect Behavior with Sensor Data

This notebook reproduces all Phase 1 EDA findings.
Heavy computation (chunked CSV reads) was run by `scripts/phase1_eda.py`.
This notebook loads pre-computed outputs and adds interactive analysis."""
    )
)

# Setup
cells.append(
    code_cell(
        """from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("..")
DATA_RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
OUTPUTS_EDA = ROOT / "outputs" / "eda"
CONFIGS = ROOT / "configs"
EXAMPLES = OUTPUTS_EDA / "examples"

print("Root:", ROOT.resolve())
print("train.csv exists:", (DATA_RAW / "train.csv").exists())
print("EDA outputs exist:", OUTPUTS_EDA.exists())"""
    )
)

# 1. Dataset Inventory
cells.append(md_cell("## 1. Dataset Inventory"))
cells.append(
    code_cell("""print((REPORTS / "dataset_inventory.md").read_text(encoding="utf-8")[:3000])""")
)

# 2. Profile
cells.append(md_cell("## 2. Dataset Profile"))
cells.append(code_cell("""print((REPORTS / "dataset_profile.md").read_text(encoding="utf-8"))"""))

cells.append(
    code_cell(
        """# Quick peek (5 rows only — never load full 1.1 GB)
sample = pd.read_csv(DATA_RAW / "train.csv", nrows=5)
meta = ["row_id","sequence_type","sequence_id","sequence_counter",
        "subject","orientation","behavior","phase","gesture"]
print("Shape:", sample.shape)
display(sample[meta])"""
    )
)

# 3. Sensor groups
cells.append(md_cell("## 3. Sensor Group Identification"))
cells.append(
    code_cell(
        """with open(CONFIGS / "sensor_groups.yaml") as f:
    sensor_groups = yaml.safe_load(f)

for group, cols in sensor_groups.items():
    sample_cols = cols[:3] if len(cols) > 3 else cols
    print(f"{group.upper():12s}: {len(cols):3d} features  e.g. {sample_cols}")"""
    )
)

# 4. Target
cells.append(md_cell("## 4. Target Analysis — Behavior Classes"))
cells.append(
    code_cell("""print((REPORTS / "class_analysis.md").read_text(encoding="utf-8")[:3000])""")
)

cells.append(
    code_cell(
        """img = mpimg.imread(OUTPUTS_EDA / "class_distribution.png")
fig, ax = plt.subplots(figsize=(16, 8))
ax.imshow(img)
ax.axis("off")
ax.set_title("Behavior Class Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()"""
    )
)

# 5. Sequence analysis
cells.append(md_cell("## 5. Sequence Analysis"))
cells.append(code_cell("""print((REPORTS / "sequence_analysis.md").read_text(encoding="utf-8"))"""))

cells.append(
    code_cell(
        """img = mpimg.imread(OUTPUTS_EDA / "sequence_length_distribution.png")
fig, ax = plt.subplots(figsize=(14, 5))
ax.imshow(img)
ax.axis("off")
plt.tight_layout()
plt.show()"""
    )
)

# 6. Missing data
cells.append(md_cell("## 6. Missing Data Analysis"))
cells.append(
    code_cell(
        """missing_df = pd.read_csv(OUTPUTS_EDA / "missing_values_report.csv")
print("=== Missing data by modality ===")
summary = missing_df.groupby("modality")[[
    "missing_pct_total","missing_pct_nan","missing_pct_sentinel_minus1"
]].describe().round(2)
display(summary)"""
    )
)

cells.append(
    code_cell(
        """img = mpimg.imread(OUTPUTS_EDA / "missing_values_heatmap.png")
fig, ax = plt.subplots(figsize=(18, 5))
ax.imshow(img)
ax.axis("off")
ax.set_title("Missing Values Heatmap by Modality", fontsize=13)
plt.tight_layout()
plt.show()"""
    )
)

cells.append(
    code_cell(
        """# ToF invalidity breakdown
tof_missing = missing_df[missing_df["modality"] == "tof"]
print(f"ToF sentinel (-1.0) invalidity statistics:")
print(f"  Mean across 320 features: {tof_missing.missing_pct_sentinel_minus1.mean():.1f}%")
print(f"  Max: {tof_missing.missing_pct_sentinel_minus1.max():.1f}%")
print(f"  Min: {tof_missing.missing_pct_sentinel_minus1.min():.1f}%")
print()
print("This is the PRIMARY motivation for the ToF mask channel and Reliability Module.")
print("~59% of ToF readings are invalid — a static fusion would be heavily corrupted.")"""
    )
)

# 7. Sensor statistics
cells.append(md_cell("## 7. Sensor Statistics"))
cells.append(
    code_cell(
        """stats_df = pd.read_csv(OUTPUTS_EDA / "sensor_statistics.csv")
print("Summary by modality:")
display(stats_df.groupby("modality")[["mean","std","min","max","skewness"]].describe().round(3))"""
    )
)

cells.append(
    code_cell(
        """# IMU stats
imu = stats_df[stats_df["modality"] == "imu"]
print("IMU feature statistics:")
display(imu[["feature","mean","std","min","max","skewness"]].reset_index(drop=True))"""
    )
)

cells.append(
    code_cell(
        r"""# Visualize mean +/- std per modality
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Per-Feature Statistics by Modality", fontsize=13, fontweight="bold")
palette = {"imu": "#2196F3", "thermal": "#FF5722", "tof": "#4CAF50"}

for ax, mod in zip(axes, ["imu", "thermal", "tof"]):
    df = stats_df[stats_df["modality"] == mod].reset_index(drop=True)
    x = range(len(df))
    ax.bar(x, df["mean"], color=palette[mod], alpha=0.7)
    ax.errorbar(x, df["mean"], yerr=df["std"],
                fmt="none", color="black", capsize=3, linewidth=0.8)
    ax.set_title(f"{mod.upper()} (mean +/- std)")
    ax.set_xticks(list(x))
    labels = df["feature"].str.replace(r"tof_\d+_v", "v", regex=True)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()"""
    )
)

# 8. Modality visualizations
cells.append(md_cell("## 8. Modality Visualizations per Behavior"))
cells.append(
    code_cell(
        """examples = list(EXAMPLES.glob("*.png"))
print(f"Generated {len(examples)} signal visualization plots:")
for f in sorted(examples):
    print(f"  {f.name}")"""
    )
)

cells.append(
    code_cell(
        """# Show IMU signals per behavior
imu_files = sorted(EXAMPLES.glob("imu_*.png"))
for img_path in imu_files:
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(img_path.stem.replace("imu_", "IMU: ").replace("_", " "), fontsize=11)
    plt.tight_layout()
    plt.show()"""
    )
)

cells.append(
    code_cell(
        """# Show thermal signals per behavior
thm_files = sorted(EXAMPLES.glob("thermal_*.png"))
for img_path in thm_files:
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(img_path.stem.replace("thermal_", "Thermal: ").replace("_", " "), fontsize=11)
    plt.tight_layout()
    plt.show()"""
    )
)

cells.append(
    code_cell(
        """# Show ToF signals per behavior
tof_files = sorted(EXAMPLES.glob("tof_*.png"))
for img_path in tof_files:
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(img_path.stem.replace("tof_", "ToF: ").replace("_", " "), fontsize=11)
    plt.tight_layout()
    plt.show()"""
    )
)

# 9. Reliability motivation
cells.append(md_cell("## 9. Reliability Motivation Study"))
cells.append(
    code_cell("""print((REPORTS / "reliability_motivation.md").read_text(encoding="utf-8"))""")
)

cells.append(
    code_cell(
        """# Quantify modality reliability differences
mod_stats = missing_df.groupby("modality")["missing_pct_total"].agg(["mean","max","min"]).round(2)
print("Missing data by modality (combined NaN + -1 sentinel):")
display(mod_stats)

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(mod_stats))
bars = ax.bar(x, mod_stats["mean"], color=["#2196F3", "#FF5722", "#4CAF50"],
              width=0.6, alpha=0.85)
ax.errorbar(x, mod_stats["mean"],
            yerr=[mod_stats["mean"]-mod_stats["min"], mod_stats["max"]-mod_stats["mean"]],
            fmt="none", color="black", capsize=8, linewidth=1.5)
ax.set_xticks(list(x))
ax.set_xticklabels(mod_stats.index, fontsize=13)
ax.set_ylabel("Mean Missing % per Feature", fontsize=12)
ax.set_title("Reliability Motivation: Per-Modality Data Quality (higher = less reliable)", fontsize=12)
for bar, val in zip(bars, mod_stats["mean"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.show()"""
    )
)

# 10. Preprocessing
cells.append(md_cell("## 10. Preprocessing Recommendations"))
cells.append(
    code_cell(
        """print((REPORTS / "preprocessing_recommendations.md").read_text(encoding="utf-8"))"""
    )
)

# 11. Dataset smoke test
cells.append(md_cell("## 11. Dataset Class & DataLoader Smoke Test"))
cells.append(
    code_cell(
        """import sys
sys.path.insert(0, str(ROOT / "src"))

from arst.data.dataloader import build_csv_loaders

train_loader, val_loader, test_loader, info = build_csv_loaders(
    csv_path=DATA_RAW / "train.csv",
    window_size=128,
    batch_size=16,
    max_rows=20000,   # small subset for quick test
    num_workers=0,
)

print("DataLoader info:")
for k, v in info.items():
    if k not in ["behavior_encoder", "class_weights"]:
        print(f"  {k}: {v}")
print()
print("Behavior classes:", info["behavior_encoder"])"""
    )
)

cells.append(
    code_cell(
        """# Inspect one batch
batch = next(iter(train_loader))
print("Batch keys:", list(batch.keys()))
for key in ["imu", "thermo", "tof", "tof_mask", "label"]:
    print(f"  {key:10s}: shape={batch[key].shape}, dtype={batch[key].dtype}")
print()
print("Label sample:", batch["label"][:8])"""
    )
)

cells.append(
    code_cell(
        """# Verify ToF mask
tof_mask = batch["tof_mask"]
tof = batch["tof"]
print(f"ToF mask: unique values = {tof_mask.unique().tolist()}")
print(f"ToF valid fraction (mask=1): {tof_mask.mean().item():.3f}")
print(f"ToF min (after sentinel replace): {tof.min().item():.3f}")
print(f"ToF max: {tof.max().item():.3f}")"""
    )
)

# 12. Summary
cells.append(md_cell("## 12. Phase 1 Summary & Key Findings"))
cells.append(code_cell("""print((REPORTS / "phase1_summary.md").read_text(encoding="utf-8"))"""))

cells.append(
    md_cell(
        """## Conclusion

Phase 1 is **COMPLETE**. Key discoveries:

| Finding | Value | Impact on Architecture |
|---|---|---|
| Dataset format | Flat CSV (not parquet) | Chunked reading + HDF5 conversion required |
| Total rows | 574,945 | Never load all at once |
| Behavior classes | 4 | Multi-class classification |
| Class imbalance | Significant | Use Focal Loss + Macro F1 |
| IMU channels | 7 (acc_xyz + quaternion) | Not raw gyroscope |
| Thermal channels | 5 (not 64) | Simple linear head |
| ToF features | 320 (5 sensors x 64) | Per-sensor processing |
| **ToF invalidity** | **~59% avg** | **Critical: mask channel + Reliability Module** |
| Sequences | 8,151 | Variable length |
| Recommended T | 128 | Based on P25 sequence length |

**Ready for Phase 2: Baseline Models**"""
    )
)

# Build notebook
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = Path("notebooks/phase1_dataset_exploration.ipynb")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out}")
print(f"Cells: {len(cells)}")
